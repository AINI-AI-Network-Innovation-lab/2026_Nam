import numpy as np
from . import task
import torch
import glob
import os

# Cấu hình
TOTAL_CLIENTS = 20
OUTPUT_MODEL_NAME = "final_model_macforget_dirichlet.npz"

# --- Tinh chỉnh Server ---
# Đôi khi Mask Gradient tính ra ở Client hơi nhỏ so với tổng trọng số 1.5 tỷ tham số.
# Ta dùng hệ số này để "khuếch đại" tín hiệu quên.
MASK_SCALE_FACTOR = 10.0 

def evaluate_model(model, client_id: int):
    print(f"--- Evaluating Client {client_id} ---")
    _, testloader = task.load_data(partition_id=client_id, num_clients=TOTAL_CLIENTS, mode="train")
    if testloader is None: 
        print(f"Client {client_id}: No data.")
        return
    loss, metrics = task.test(model, testloader)
    print(f"Client {client_id}: Loss = {loss:.4f}, PPL = {metrics['perplexity']:.2f}, Acc = {metrics['accuracy']:.4f}")

def main():
    print("Loading original server weights (Theta_k)...")
    weights_server_old_npz = np.load("trained_model_weights1.5B_dirichlet.npz")
    list_server_old = [weights_server_old_npz[key] for key in weights_server_old_npz]

    search_pattern = "macforget_weights_client_*_dirichlet.npz"
    files = glob.glob(search_pattern)
    
    if not files: 
        print(f"No MacForget weights found. Please run macforget_client.py first.")
        return
    
    print(f"Found {len(files)} client updates.")
    
    total_mask_gradient = [np.zeros_like(w) for w in list_server_old]
    target_clients = [] 
    
    for fpath in files:
        try:
            clean_fpath = fpath.replace("_dirichlet", "")
            cid = int(clean_fpath.split("_")[-1].split(".")[0])
            target_clients.append(cid)
        except ValueError:
            continue
        
        print(f"Computing Mask Gradient from Client {cid}...")
        data_client = np.load(fpath)
        list_client_weights = [data_client[key] for key in data_client]
        
        # Mask = W_server - W_client
        for i, (w_server, w_client) in enumerate(zip(list_server_old, list_client_weights)):
            mask_i = w_server - w_client
            total_mask_gradient[i] += mask_i

    print(f"Applying Neuron Masking to global model (Scale Factor = {MASK_SCALE_FACTOR})...")
    weights_server_new = []
    
    for w_old, mask_sum in zip(list_server_old, total_mask_gradient):
        # Eq. 7: Theta_new = Theta_old - (Scale * Mask)
        # Scale giúp tín hiệu quên mạnh hơn
        w_updated = w_old - (mask_sum * MASK_SCALE_FACTOR)
        weights_server_new.append(w_updated)

    np.savez(OUTPUT_MODEL_NAME, *weights_server_new)
    print(f"Saved Final MacForget Model to: {OUTPUT_MODEL_NAME}")

    # Đánh giá nhanh
    print("\n" + "="*30)
    print("EVALUATION REPORT: MACFORGET ALGORITHM")
    print("="*30)
    
    model = task.load_model()
    task.set_weights(model, weights_server_new)

    print("\n--- Target Clients ---")
    for cid in target_clients:
        evaluate_model(model, cid)

    print("\n--- Other Clients ---")
    count = 0
    for cid in range(TOTAL_CLIENTS):
        if cid not in target_clients:
            evaluate_model(model, cid)
            count += 1
            if count >= 2: break 

if __name__ == "__main__":
    main()
