import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import bitsandbytes.optim as bnb_optim
import gc
import os
import copy
from flwr.common import Context 
from typing import Dict, List, Optional, Tuple, Union
from flwr.server.strategy import FedAvg
from flwr.common import (
    parameters_to_ndarrays, 
    ndarrays_to_parameters, 
    Parameters,
    FitRes, 
    FitIns,
    EvaluateIns
)
from torch.utils.data import DataLoader
from . import task

# --- CẤU HÌNH ---
NUM_CLIENTS = 20
UNLEARN_ROUNDS = 5 
LOCAL_EPOCHS_MEVAL = 1 
LOCAL_EPOCHS_MERASE = 1 
CLIENTS_PER_ROUND = 20
FRACTION_FIT = CLIENTS_PER_ROUND / NUM_CLIENTS 
PRETRAINED_WEIGHTS_FILE = "trained_model_weights1.5B.npz" 

# --- MODEL MEVAL (Mạng đánh giá bộ nhớ) ---
class MEvalNetwork(nn.Module):
    def __init__(self, input_dim):
        super(MEvalNetwork, self).__init__()
        # Giảm kích thước mạng MEval để tiết kiệm VRAM
        self.fc1 = nn.Linear(input_dim, 256) 
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1) 

    def forward(self, x):
        # Ép kiểu x về đúng dtype của mạng (bfloat16) để khớp
        x = x.to(self.fc1.weight.dtype)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

def extract_features_detached(model, batch):
    """
    Trích xuất đặc trưng dùng cho việc TRAIN MEVAL.
    Sử dụng no_grad() và detach() để không lưu computational graph, tiết kiệm VRAM.
    """
    with torch.no_grad():
        outputs = model(**batch, output_hidden_states=True)
    
    hidden_states = outputs.hidden_states
    
    # Lấy 3 tầng đại diện: Nông, Trung, Sâu
    h1 = hidden_states[0] 
    h2 = hidden_states[len(hidden_states) // 2] 
    h3 = hidden_states[-1] 
    
    # Mean pooling + Detach
    f1 = h1.mean(dim=1).detach()
    f2 = h2.mean(dim=1).detach()
    f3 = h3.mean(dim=1).detach()
    
    features = torch.cat([f1, f2, f3], dim=-1)
    # Trả về bfloat16 để nhẹ bộ nhớ
    return features.to(torch.bfloat16)

# --- CLIENT FEDME2 ---
class FedME2Client(fl.client.NumPyClient):
    def __init__(self, cid: int, num_partitions: int):
        self.cid = cid
        self.model = task.load_model() 
        self.model.to(task.DEVICE)
        
        self.forget_loader, _ = task.load_data(cid, num_partitions, mode="unlearn")
        self.retain_loader, self.test_loader = task.load_data(cid, num_partitions, mode="train")
        
        # Optimizer cho model chính (dùng 8-bit optimizer)
        self.optimizer_merase = bnb_optim.AdamW8bit(self.model.parameters(), lr=5e-6)
        
        # MEval setup
        hidden_dim = self.model.config.hidden_size
        self.meval_model = MEvalNetwork(input_dim=hidden_dim * 3).to(task.DEVICE)
        # Chuyển MEval sang bfloat16
        self.meval_model = self.meval_model.to(torch.bfloat16) 
        self.optimizer_meval = torch.optim.Adam(self.meval_model.parameters(), lr=1e-4)

    def get_parameters(self, config):
        return task.get_weights(self.model)

    def fit(self, parameters, config):
        task.set_weights(self.model, parameters)
        
        # Lấy số lượng mẫu retain để trả về (tránh lỗi chia cho 0)
        num_examples = len(self.retain_loader.dataset) if self.retain_loader else 1

        if self.forget_loader is None:
            print(f"[Client {self.cid}] Không có dữ liệu cần quên. Skip.")
            return task.get_weights(self.model), num_examples, {}

        # ================= GIAI ĐOẠN 1: MEval (Train mạng đánh giá) =================
        # Mục tiêu: MEval phân biệt được Retain (Label 1) và Test/Non-member (Label 0)
        print(f"[Client {self.cid}] Phase MEval...")
        self.meval_model.train()
        self.model.eval() 
        
        gc.collect()
        torch.cuda.empty_cache()

        for _ in range(LOCAL_EPOCHS_MEVAL):
            # 1.1 Train Positive (Retain Data)
            for i, batch in enumerate(self.retain_loader):
                if i > 5: break # Limit số batch để nhanh và đỡ tốn mem
                batch = {k: v.to(task.DEVICE) for k, v in batch.items()}
                
                # Extract features (Detached)
                features = extract_features_detached(self.model, batch)
                
                preds = self.meval_model(features)
                # Label = 1
                loss = F.binary_cross_entropy(preds.float(), torch.ones_like(preds).float())
                
                self.optimizer_meval.zero_grad()
                loss.backward()
                self.optimizer_meval.step()
                
                del batch, features, preds, loss
            
            # 1.2 Train Negative (Test Data giả lập Non-member)
            for i, batch in enumerate(self.test_loader):
                if i > 5: break 
                batch = {k: v.to(task.DEVICE) for k, v in batch.items()}
                
                features = extract_features_detached(self.model, batch)
                
                preds = self.meval_model(features)
                # Label = 0
                loss = F.binary_cross_entropy(preds.float(), torch.zeros_like(preds).float())
                
                self.optimizer_meval.zero_grad()
                loss.backward()
                self.optimizer_meval.step()
                
                del batch, features, preds, loss

        # ================= GIAI ĐOẠN 2: MErase (Xóa ký ức) =================
        # Mục tiêu: Train model chính sao cho giữ accuracy trên Retain nhưng lừa MEval trên Forget
        print(f"[Client {self.cid}] Phase MErase...")
        self.model.train()
        self.meval_model.eval() 
        
        iter_retain = iter(self.retain_loader)
        
        for _ in range(LOCAL_EPOCHS_MERASE):
            for i, batch_forget in enumerate(self.forget_loader):
                # Clean cache đầu vòng lặp để giải phóng VRAM
                torch.cuda.empty_cache()
                
                # Lấy batch retain tương ứng
                try:
                    batch_retain = next(iter_retain)
                except StopIteration:
                    iter_retain = iter(self.retain_loader)
                    batch_retain = next(iter_retain)

                # --- Bước 2.1: Tính L_local (Duy trì tri thức Retain) ---
                batch_retain = {k: v.to(task.DEVICE) for k, v in batch_retain.items()}
                outputs_retain = self.model(**batch_retain)
                l_local = outputs_retain.loss 
                
                # Backprop L_local ngay để giải phóng graph
                l_local.backward()
                del batch_retain, outputs_retain, l_local

                # --- Bước 2.2: Tính L_eval (Unlearn Forget) ---
                batch_forget = {k: v.to(task.DEVICE) for k, v in batch_forget.items()}
                
                # Cần forward pass để lấy hidden states có Gradient (để update model chính)
                # Dùng autocast bfloat16 để tiết kiệm bộ nhớ
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs_forget = self.model(**batch_forget, output_hidden_states=True)
                    h_states = outputs_forget.hidden_states
                    
                    # Manual features extraction (có kết nối Gradient)
                    f1 = h_states[0].mean(1)
                    f2 = h_states[len(h_states)//2].mean(1)
                    f3 = h_states[-1].mean(1)
                    f_cat = torch.cat([f1, f2, f3], dim=-1) # bfloat16
                
                preds_forget = self.meval_model(f_cat)
                
                # Mục tiêu: Ép MEval đoán output là 0 (Non-member) cho dữ liệu Forget
                l_eval = F.binary_cross_entropy(preds_forget.float(), torch.zeros_like(preds_forget).float())
                
                # Backprop L_eval
                l_eval.backward()
                
                # Update weights
                self.optimizer_merase.step()
                self.optimizer_merase.zero_grad()
                
                del batch_forget, outputs_forget, h_states, f_cat, preds_forget, l_eval
        
        gc.collect()
        torch.cuda.empty_cache()
        return task.get_weights(self.model), num_examples, {}

    def evaluate(self, parameters, config):
        if self.test_loader is None:
             return 0.0, 0, {}
        task.set_weights(self.model, parameters)
        loss, metrics = task.test(self.model, self.test_loader)
        return float(loss), len(self.test_loader.dataset), metrics

# FIX: Cập nhật hàm client_fn dùng Context
def client_fn(context: Context) -> fl.client.Client:
    partition_id = int(context.node_config["partition-id"])
    try:
        return FedME2Client(partition_id, NUM_CLIENTS).to_client()
    except Exception as e:
        print(f"Error init client {partition_id}: {e}")
        os._exit(1)

# --- SERVER SIDE ---
def fit_config(server_round: int):
    return {"round": str(server_round)}

def main():
    print(f"--- FedME2 Unlearning Framework (Multi-GPU Optimized) ---")
    
    if os.path.exists(PRETRAINED_WEIGHTS_FILE):
        print(f"Loading pretrained weights from {PRETRAINED_WEIGHTS_FILE}...")
        loaded = np.load(PRETRAINED_WEIGHTS_FILE)
        initial_weights = [loaded[key] for key in loaded.files]
        initial_parameters = ndarrays_to_parameters(initial_weights)
    else:
        print(f"Error: File {PRETRAINED_WEIGHTS_FILE} not found!")
        return

    strategy = FedAvg(
        fraction_fit=FRACTION_FIT,
        min_fit_clients=CLIENTS_PER_ROUND,
        min_available_clients=NUM_CLIENTS,
        initial_parameters=initial_parameters,
        on_fit_config_fn=fit_config,
    )

    # --- TỰ ĐỘNG CẤU HÌNH GPU ---
    num_gpus_available = torch.cuda.device_count()
    print(f"Detected {num_gpus_available} GPUs.")

    # Cấu hình tài nguyên cho mỗi Client
    # num_gpus=0.5 nghĩa là 1 GPU chạy được 2 clients song song.
    # Với 2 GPU, tổng cộng chạy được 4 clients song song.
    # Nếu vẫn bị OOM, hãy sửa thành 1.0 (mỗi GPU chỉ chạy 1 client).
    client_resources = {"num_cpus": 1, "num_gpus": 0.5}

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=UNLEARN_ROUNDS),
        strategy=strategy,
        client_resources=client_resources,
        # Ray sẽ quản lý toàn bộ GPU có sẵn
        ray_init_args={
            "num_gpus": num_gpus_available, 
            "include_dashboard": False
        } 
    )

    print("Unlearning complete.")
    
if __name__ == "__main__":
    main()
