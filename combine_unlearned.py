import numpy as np
from . import task
import torch
import glob

LAMBDA = 0.9
TOTAL_CLIENTS = 20

def evaluate_model(model, client_id: int):
    _, testloader = task.load_data(partition_id=client_id, num_clients=TOTAL_CLIENTS, mode="train")
    if testloader is None: return
    loss, metrics = task.test(model, testloader)
    print(f"Client {client_id}: Loss = {loss:.4f}, PPL = {metrics['perplexity']:.2f}")

def main():
    weights_server_old_npz = np.load("trained_model_weights1.5B_dirichlet.npz")
    list_server_old = [weights_server_old_npz[key] for key in weights_server_old_npz]

    files = glob.glob("unlearned_weights_client_*_dirichlet.npz")
    if not files: return
    
    accumulated_delta = [np.zeros_like(w) for w in list_server_old]
    target_clients = [] 
    
    for fpath in files:
        clean_fpath = fpath.replace("_dirichlet", "")
        cid = int(clean_fpath.split("_")[-1].split(".")[0])
        target_clients.append(cid)
        
        data_new = np.load(fpath)
        list_new = [data_new[key] for key in data_new]
        
        for i, (w_old, w_new) in enumerate(zip(list_server_old, list_new)):
            delta = w_new - w_old
            accumulated_delta[i] += delta

    weights_server_new = []
    for w_old, delta_sum in zip(list_server_old, accumulated_delta):
        w_updated = w_old + LAMBDA * delta_sum
        weights_server_new.append(w_updated)

    np.savez("final_model_after_unlearning_dirichlet.npz", *weights_server_new)

    model = task.load_model()
    task.set_weights(model, weights_server_new)

    for cid in target_clients:
        evaluate_model(model, cid)

    count = 0
    for cid in range(TOTAL_CLIENTS):
        if cid not in target_clients:
            evaluate_model(model, cid)
            count += 1
            if count >= 2: break

if __name__ == "__main__":
    main()
