import numpy as np
from . import task 
import torch 
import os

TOTAL_CLIENTS = 20
STRENGTHEN_EPOCHS = 10
UNLEARN_EPOCHS = 5
TARGET_CLIENT_ID = 0
ALPHA_RETAIN = 1.0

def main():
    torch.cuda.empty_cache()
    
    loaded_weights = np.load("trained_model_weights1.5B_dirichlet.npz")
    weights = [loaded_weights[key] for key in loaded_weights]
    global_model = task.load_model()
    task.set_weights(global_model, weights)

    # 1. Load Forget Data
    forget_loader, _ = task.load_data(
        partition_id=TARGET_CLIENT_ID,
        num_clients=TOTAL_CLIENTS,
        shuffle_train=False,
        mode="unlearn" 
    )
    
    # 2. [MỚI] Load Retain Data (Dữ liệu cần giữ lại của Client 0)
    retain_loader, _ = task.load_data(
        partition_id=TARGET_CLIENT_ID,
        num_clients=TOTAL_CLIENTS,
        shuffle_train=True, # Shuffle để mô hình học ngẫu nhiên
        mode="retain"       # Gọi mode mới vừa thêm ở task.py
    )

    if forget_loader is None: 
        return

    # 3. Truyền vào hàm unlearn
    task.unlearn(
        global_model, 
        forget_loader, 
        epochs=UNLEARN_EPOCHS, 
        strengthen_epochs=STRENGTHEN_EPOCHS,
        retain_loader=retain_loader,  # [MỚI]
        alpha_retain=ALPHA_RETAIN     # [MỚI]
    )

    w_new = task.get_weights(global_model)
    fname = f"unlearned_weights_client_{TARGET_CLIENT_ID}_dirichlet.npz"
    np.savez(fname, *w_new)

if __name__ == "__main__":
    main()
