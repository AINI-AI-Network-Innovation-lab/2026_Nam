import numpy as np
from . import task 
import torch 
import os
import gc
import sys

TOTAL_CLIENTS = 20
RETRAIN_EPOCHS = 1
LEARNING_RATE = 0.001
TARGET_CLIENT_ID = 0
OUTPUT_MODEL_NAME = "final_model_rapid_retrain_dirichlet.npz"

def main():
    print("=== STARTING FEDERATED RAPID RETRAINING SIMULATION (ALGORITHM 1) ===")
    
    print("\n[Server] Loading initial global model (w*)...")
    try:
        loaded_weights = np.load("trained_model_weights1.5B_dirichlet.npz")
        initial_weights = [loaded_weights[key] for key in loaded_weights]
    except FileNotFoundError:
        print("[Error] File 'trained_model_weights1.5B.npz' not found.")
        return

    for cid in range(TOTAL_CLIENTS):
        print(f"\n--- Processing Client {cid} / {TOTAL_CLIENTS - 1} ---")
        
        try:
            client_model = task.load_model()
            task.set_weights(client_model, initial_weights)
            
            if cid == TARGET_CLIENT_ID:
                mode = "retain"
                print(f"[Role] Unlearned Client")
            else:
                mode = "train"
                print(f"[Role] Normal Client")

            loader, _ = task.load_data(
                partition_id=cid,
                num_clients=TOTAL_CLIENTS,
                shuffle_train=True,
                mode=mode 
            )

            if loader is None:
                print(f"[Skip] Client {cid} no data.")
                del client_model
                continue

            print(f"[Action] Executing Rapid Retraining Solver...")
            task.rapid_retraining_solver(
                client_model,
                loader,
                epochs=RETRAIN_EPOCHS,
                lr=LEARNING_RATE,
                beta1=0.9,
                beta2=0.999
            )

            w_updated = task.get_weights(client_model)
            fname = f"rapid_retrain_weights_client_{cid}_dirichlet.npz"
            np.savez(fname, *w_updated)
            print(f"[Done] Client {cid} saved.")

            del client_model, loader, w_updated
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"[FATAL] Error Client {cid}: {e}")
            gc.collect()
            torch.cuda.empty_cache()
            continue

    print("\nSimulation complete. Run 'combine_rapid_retrain.py' to aggregate.")

if __name__ == "__main__":
    main()
