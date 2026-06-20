import numpy as np
import glob
import os
import gc
import torch
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from flwr.server.strategy import FedAvg
from flwr.common import (
    parameters_to_ndarrays, 
    ndarrays_to_parameters, 
    Parameters,
    FitRes, 
    Scalar, 
    FitIns,
    EvaluateIns
)
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling

# Import file task.py
from . import task 

# --- CẤU HÌNH ---
VALID_ALGOS = ["GA", "GradDiff", "NPO", "SimNPO"]
NUM_CLIENTS = 20           
RECOVERY_ROUNDS = 2        
LOCAL_EPOCHS = 1           
CLIENTS_PER_ROUND = 10     
CLIENT_RESOURCES = {"num_cpus": 1, "num_gpus": 0.2} 
TOTAL_GPUS_SYSTEM = 2

# ID của client cần quên (phải khớp với task.py)
TARGET_CLIENT_ID = 0 

# --- HÀM HỖ TRỢ: LOAD RETAIN DATA CHO TARGET CLIENT ---
def get_retain_loader(partition_id, num_clients):
    """
    Hàm này tạo DataLoader đặc biệt cho Target Client.
    Nó LỌC BỎ dữ liệu cần quên (Target IDs) ra khỏi tập train.
    """
    full_ds = task.get_full_dataset()
    client_map = task.get_client_map(num_clients)
    
    if partition_id not in client_map:
        return None
        
    my_author_ids = client_map[partition_id]
    
    # Lấy danh sách ID cần quên
    target_unlearn_ids = task.get_target_unlearn_ids()

    # Lọc: Chỉ giữ lại author KHÔNG nằm trong target_unlearn_ids
    my_retain_indices = []
    for auth_id in my_author_ids:
        if auth_id in target_unlearn_ids:
            continue # BỎ QUA tác giả cần quên
            
        start = auth_id * task.SAMPLES_PER_AUTHOR
        end = start + task.SAMPLES_PER_AUTHOR
        my_retain_indices.extend(range(start, end))
    
    my_retain_indices.sort()
    
    if not my_retain_indices:
        return None

    retain_data = full_ds.select(my_retain_indices)

    # Tokenize và tạo Loader (giống task.py)
    def format_prompt(example):
        return {"text": f"Question: {example['question']}\nAnswer: {example['answer']}{task.tokenizer.eos_token}"}

    def tokenize(examples):
        return task.tokenizer(examples["text"], truncation=True, max_length=256)

    train_ds = retain_data.map(format_prompt).filter(lambda x: len(x['text'])>0).map(tokenize, batched=True, remove_columns=["question", "answer", "text"])
    
    collator = DataCollatorForLanguageModeling(tokenizer=task.tokenizer, mlm=False)
    # Batch size = 1 để tiết kiệm VRAM như task.py
    trainloader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collator)
    
    return trainloader

# --- SERVER STRATEGY ---
class NoEvalFedAvg(FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.final_parameters: Optional[Parameters] = None

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[Tuple[ClientProxy, FitIns]]:
        print(f"\n--- [Server] Bắt đầu Round {server_round}/{RECOVERY_ROUNDS} (Recovery) ---")
        return super().configure_fit(server_round, parameters, client_manager)

    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[Tuple[ClientProxy, EvaluateIns]]:
        return [] # Tắt Evaluate để tiết kiệm VRAM

    def aggregate_fit(self, server_round: int, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is not None:
            self.final_parameters = aggregated_parameters
        return aggregated_parameters, aggregated_metrics

# --- CLIENT ---
class RecoveryClient(fl.client.NumPyClient):
    def __init__(self, cid: str):
        self.cid = int(cid)

    def fit(self, parameters, config):
        gc.collect()
        torch.cuda.empty_cache()

        try:
            # 1. Load Model
            net = task.load_model(on_cpu=False) 
            task.set_weights(net, parameters)
            
            # 2. Load Data (LOGIC QUAN TRỌNG)
            if self.cid == TARGET_CLIENT_ID:
                # Nếu là Target Client -> Dùng hàm riêng để LỌC BỎ dữ liệu quên
                # print(f"Client {self.cid} (Target): Loading RETAIN set only...")
                trainloader = get_retain_loader(self.cid, NUM_CLIENTS)
            else:
                # Nếu là Client khác -> Load dữ liệu bình thường (Full)
                trainloader, _ = task.load_data(
                    partition_id=self.cid, 
                    num_clients=NUM_CLIENTS, 
                    shuffle_train=True, 
                    mode="train"
                )
            
            if trainloader is None:
                return task.get_weights(net), 0, {}

            # 3. Train
            optimizer = torch.optim.AdamW(net.parameters(), lr=1e-5)
            task.train(net, trainloader, epochs=LOCAL_EPOCHS, optimizer=optimizer)
            
            # 4. Return
            new_weights = task.get_weights(net)
            data_len = len(trainloader.dataset)

            # Cleanup
            del net, optimizer, trainloader
            torch.cuda.empty_cache() 
            gc.collect()

            return new_weights, data_len, {}

        except Exception as e:
            print(f"Client {self.cid} Error: {e}")
            torch.cuda.empty_cache()
            raise e

def client_fn(cid: str):
    return RecoveryClient(cid)

def fit_config(server_round: int):
    return {"local-epochs": str(LOCAL_EPOCHS)}

# --- MAIN ---
def process_algorithm_group(algo_name, file_list):
    print(f"\n=== XỬ LÝ THUẬT TOÁN: {algo_name} ===")

    # 1. [Server Side Simulation] Nhận model unlearned từ Client
    # Đây là bước "Server nhận model đã xóa" như bạn nói
    first_data = np.load(file_list[0])
    initial_weights = [first_data[key].astype(np.float32) for key in first_data]
    first_data.close()

    # Nếu có nhiều client cùng unlearn, ta average lại (Federated Unlearning aggregation)
    if len(file_list) > 1:
        for fpath in file_list[1:]:
            curr_data = np.load(fpath)
            curr_list = [curr_data[key] for key in curr_data]
            for i in range(len(initial_weights)):
                initial_weights[i] += curr_list[i]
            curr_data.close()
        for i in range(len(initial_weights)):
            initial_weights[i] /= len(file_list)
    
    initial_parameters = ndarrays_to_parameters(initial_weights)
    del initial_weights
    gc.collect()

    # 2. Bắt đầu Recovery Rounds
    # Từ đây trở đi, Target Client sẽ đóng vai như một client bình thường
    # nhưng train trên tập RETAIN của nó.
    strategy = NoEvalFedAvg(
        fraction_fit=CLIENTS_PER_ROUND / NUM_CLIENTS,
        min_fit_clients=CLIENTS_PER_ROUND,
        min_available_clients=NUM_CLIENTS,
        initial_parameters=initial_parameters, # Global Model khởi đầu = Unlearned Model
        on_fit_config_fn=fit_config,
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=RECOVERY_ROUNDS),
        strategy=strategy,
        client_resources=CLIENT_RESOURCES, 
        ray_init_args={"num_gpus": TOTAL_GPUS_SYSTEM, "include_dashboard": False}
    )

    # 3. Lưu kết quả
    if strategy.final_parameters:
        weights = parameters_to_ndarrays(strategy.final_parameters)
        output_filename = f"final_model_{algo_name}_recovered.npz"
        np.savez(output_filename, *weights)
        print(f">>> Đã lưu: {output_filename}")
    
    gc.collect()
    torch.cuda.empty_cache()

def main():
    all_files = glob.glob("unlearned_*_client_*.npz")
    if not all_files:
        print("Không tìm thấy file unlearned nào.")
        return

    algo_groups = defaultdict(list)
    for fpath in all_files:
        try:
            name_part = os.path.basename(fpath).split("_")[1]
            if name_part in VALID_ALGOS:
                algo_groups[name_part].append(fpath)
        except IndexError:
            continue

    for algo_name, file_list in algo_groups.items():
        if not file_list: continue
        process_algorithm_group(algo_name, file_list)

if __name__ == "__main__":
    main()
