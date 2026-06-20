import numpy as np
from . import task
import torch
import glob
import os

TOTAL_CLIENTS = 20
OUTPUT_MODEL_NAME = "final_model_rapid_retrain_dirichlet.npz"

def evaluate_model(model, client_id: int):
    print(f"--- Evaluating Client {client_id} ---")
    _, testloader = task.load_data(partition_id=client_id, num_clients=TOTAL_CLIENTS, mode="train")
    if testloader is None: 
        return
    loss, metrics = task.test(model, testloader)
    print(f"Client {client_id}: Loss = {loss:.4f}, PPL = {metrics['perplexity']:.2f}, Acc = {metrics['accuracy']:.4f}")

def main():
    search_pattern = "rapid_retrain_weights_client_*_dirichlet.npz"
    files = glob.glob(search_pattern)
    
    if len(files) < TOTAL_CLIENTS:
        print(f"Warning: Found {len(files)}/{TOTAL_CLIENTS} client updates.")
    else:
        print(f"Found updates from all {len(files)} clients.")
    
    if not files: return
    
    print("Aggregating models (FedAvg)...")
    
    first_file_data = np.load(files[0])
    weights_sum = [first_file_data[key] for key in first_file_data]
    count = 1
    
    for fpath in files[1:]:
        data_client = np.load(fpath)
        list_client_weights = [data_client[key] for key in data_client]
        
        for i, w in enumerate(list_client_weights):
            weights_sum[i] += w
        count += 1
        
    weights_avg = [w / count for w in weights_sum]
    print(f"Aggregated global model from {count} clients.")

    np.savez(OUTPUT_MODEL_NAME, *weights_avg)
    print(f"Saved Final Rapid Retrain Model to: {OUTPUT_MODEL_NAME}")

    print("\n" + "="*40)
    print("EVALUATION REPORT: BASELINE RAPID RETRAINING")
    print("="*40)
    
    model = task.load_model()
    task.set_weights(model, weights_avg)

    target_clients = [0]

    print("\n--- Unlearned Client (Forget Set) ---")
    _, forget_loader = task.load_data(0, TOTAL_CLIENTS, mode="unlearn")
    if forget_loader:
        loss, metric = task.test(model, forget_loader)
        print(f"Client 0 (Forget Set): Loss={loss:.4f}, Acc={metric['accuracy']:.4f}")

    print("\n--- Normal Clients (Utility) ---")
    count_eval = 0
    for cid in range(TOTAL_CLIENTS):
        if cid not in target_clients:
            evaluate_model(model, cid)
            count_eval += 1
            if count_eval >= 2: break 

if __name__ == "__main__":
    main()
