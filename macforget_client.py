import numpy as np
from . import task 
import torch 
import os

# Cấu hình MacForget
TOTAL_CLIENTS = 20
MAC_FORGET_EPOCHS = 6      # Bài báo gợi ý số epoch nhỏ vì hội tụ nhanh [cite: 377]
LAMBDA_PENALTY = 0.01       # Hệ số phạt L1 (lambda trong Eq 4) [cite: 415]
TARGET_CLIENT_ID = 0

def main():
    torch.cuda.empty_cache()
    
    # 1. Load trọng số Global Model ban đầu
    print("Loading global model...")
    loaded_weights = np.load("trained_model_weights1.5B_dirichlet.npz")
    weights = [loaded_weights[key] for key in loaded_weights]
    global_model = task.load_model()
    task.set_weights(global_model, weights)

    # 2. Load dữ liệu cần unlearn của Client 0
    # Algorithm 2 (MacForget): Client tự lấy mẫu dữ liệu và thực hiện local unlearning [cite: 228-234]
    print(f"Loading unlearning data for client {TARGET_CLIENT_ID}...")
    forget_loader, _ = task.load_data(
        partition_id=TARGET_CLIENT_ID,
        num_clients=TOTAL_CLIENTS,
        shuffle_train=True,
        mode="unlearn" 
    )

    if forget_loader is None: 
        print("No target data found for unlearning.")
        return

    # 3. Thực hiện thuật toán MacForget (Algorithm 1)
    # Quá trình này tính toán 'mask gradient' được nhúng trực tiếp vào trọng số
    print("Executing MacForget algorithm...")
    task.macforget_solver(
        global_model, 
        forget_loader, 
        epochs=MAC_FORGET_EPOCHS, 
        lambda_penalty=LAMBDA_PENALTY
    )

    # 4. Lưu trọng số đã thay đổi (Masked Weights)
    # File combine_unlearned.py sẽ tính delta = w_new - w_old 
    # Điều này tương đương với việc Server thực hiện: theta_{k+1} = theta_k - mu [cite: 244]
    print("Saving unlearned weights...")
    w_new = task.get_weights(global_model)
    fname = f"macforget_weights_client_{TARGET_CLIENT_ID}_dirichlet.npz"
    np.savez(fname, *w_new)
    print(f"Done. Saved to {fname}")

if __name__ == "__main__":
    main()
