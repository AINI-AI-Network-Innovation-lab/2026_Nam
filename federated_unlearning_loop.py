import numpy as np
import torch
import copy
import gc
import os
import torch.multiprocessing as mp
from collections import OrderedDict

# Import module
from . import task
from . import unlearning_oblivionis

# --- CẤU HÌNH ---
NUM_ROUNDS = 5                  # Số vòng lặp Federated (Global Rounds)
TOTAL_CLIENTS = 20              # Tổng số client
TARGET_CLIENT_ID = 0            # Client thực hiện unlearning
NORMAL_CLIENT_EPOCHS = 1        # Số epoch huấn luyện local
NORMAL_LR = 2e-5                # Learning rate

# Cấu hình GPU & Song song
GPUS = [0, 1]                   # Danh sách ID của GPU (cuda:0, cuda:1)
CLIENTS_PER_GPU = 4             # Số lượng client chạy cùng lúc trên 1 GPU

# Danh sách thuật toán
ALGORITHMS = ["GA", "GradDiff", "NPO", "SimNPO"]

# File trọng số ban đầu
INITIAL_WEIGHTS_PATH = "trained_model_weights1.5B.npz"

def aggregate_weights(weights_list):
    """FedAvg: Trung bình cộng trọng số."""
    if not weights_list: return None
    
    # Lấy mẫu trọng số đầu tiên để làm khung
    avg_weights = copy.deepcopy(weights_list[0])
    num_clients = len(weights_list)
    
    # Cộng dồn
    for i in range(1, num_clients):
        for w_idx in range(len(avg_weights)):
            avg_weights[w_idx] += weights_list[i][w_idx]
            
    # Chia trung bình
    for w_idx in range(len(avg_weights)):
        avg_weights[w_idx] /= num_clients
        
    return avg_weights

def run_single_client_process(queue, client_id, gpu_id, global_weights, algo_name, is_target):
    """
    Hàm này chạy trong một Process riêng biệt.
    """
    try:
        # 1. Thiết lập GPU cho process này
        device_str = f"cuda:{gpu_id}"
        device = torch.device(device_str)
        torch.cuda.set_device(device)
        
        # 2. "Monkey Patch": Ép biến DEVICE trong module task thành GPU của process này
        task.DEVICE = device
        
        # 3. Load Model
        model = task.load_model()
        task.set_weights(model, global_weights)
        model.to(device)

        if is_target:
            # --- TARGET CLIENT (UNLEARNING) ---
            print(f"[GPU {gpu_id}] Client {client_id} (Target): Unlearning {algo_name}...")
            
            forget_loader, _ = task.load_data(partition_id=client_id, num_clients=TOTAL_CLIENTS, mode="unlearn")
            retain_client_id = (client_id + 1) % TOTAL_CLIENTS
            retain_loader, _ = task.load_data(partition_id=retain_client_id, num_clients=TOTAL_CLIENTS, mode="train")
            
            # Unlearning (optimizer được tạo bên trong hàm này nên không cần quản lý ở đây)
            unlearning_oblivionis.run_unlearning(model, forget_loader, retain_loader, method=algo_name)
        
        else:
            # --- NORMAL CLIENT (TRAINING) ---
            print(f"[GPU {gpu_id}] Client {client_id}: Training...")
            
            train_loader, _ = task.load_data(partition_id=client_id, num_clients=TOTAL_CLIENTS, mode="train")
            
            # Tạo optimizer chỉ cho normal client
            optimizer = torch.optim.AdamW(model.parameters(), lr=NORMAL_LR)
            task.train(model, train_loader, epochs=NORMAL_CLIENT_EPOCHS, optimizer=optimizer)
            
            # Xóa optimizer ngay sau khi dùng xong
            del optimizer

        # 4. Lấy trọng số trả về
        weights = task.get_weights(model)
        queue.put(weights)
        
        # Dọn dẹp model chung
        del model
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"!!! Error on Client {client_id} (GPU {gpu_id}): {e}")
        queue.put(None)

def run_federated_experiment(algo_name):
    print(f"\n==================================================")
    print(f"STARTING FEDERATED EXPERIMENT (MULTI-GPU): {algo_name}")
    print(f"==================================================")
    
    if not os.path.exists(INITIAL_WEIGHTS_PATH):
        print(f"Error: Initial weights file '{INITIAL_WEIGHTS_PATH}' not found.")
        return

    # Load weights ban đầu
    loaded = np.load(INITIAL_WEIGHTS_PATH)
    global_weights = [loaded[key] for key in loaded]
    loaded.close()
    
    all_clients = list(range(TOTAL_CLIENTS))
    
    # Kích thước batch
    BATCH_SIZE = len(GPUS) * CLIENTS_PER_GPU
    
    for round_idx in range(1, NUM_ROUNDS + 1):
        print(f"\n--- Round {round_idx}/{NUM_ROUNDS} [{algo_name}] ---")
        
        local_weights_collected = []
        
        # Chia client thành từng batch
        for i in range(0, len(all_clients), BATCH_SIZE):
            batch_client_ids = all_clients[i : i + BATCH_SIZE]
            print(f"Processing batch: {batch_client_ids}")
            
            processes = []
            queue = mp.Queue()
            
            for idx, cid in enumerate(batch_client_ids):
                gpu_id = GPUS[idx % len(GPUS)]
                is_target = (cid == TARGET_CLIENT_ID)
                
                p = mp.Process(
                    target=run_single_client_process,
                    args=(queue, cid, gpu_id, global_weights, algo_name, is_target)
                )
                p.start()
                processes.append(p)
            
            for p in processes:
                p.join()
            
            while not queue.empty():
                w = queue.get()
                if w is not None:
                    local_weights_collected.append(w)
            
            gc.collect()

        print(f"Aggregating weights from {len(local_weights_collected)} clients...")
        if len(local_weights_collected) > 0:
            global_weights = aggregate_weights(local_weights_collected)
        
        del local_weights_collected
        gc.collect()

    output_filename = f"fed_unlearned_5rounds_{algo_name}.npz"
    np.savez(output_filename, *global_weights)
    print(f"Saved: {output_filename}")

def main():
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available!")

    for algo in ALGORITHMS:
        run_federated_experiment(algo)

if __name__ == "__main__":
    main()
