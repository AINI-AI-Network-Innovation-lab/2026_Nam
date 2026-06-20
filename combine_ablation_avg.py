import numpy as np
import glob
import os

TOTAL_CLIENTS = 20

def main():
    print("Bắt đầu thực hiện Ablation Study: Standard Parameter Averaging...")
    
    # 1. Load global model gốc trước khi unlearning
    if not os.path.exists("trained_model_weights1.5B.npz"):
        print("Lỗi: Không tìm thấy trained_model_weights1.5B.npz")
        return
        
    weights_server_old_npz = np.load("trained_model_weights1.5B.npz")
    list_server_old = [weights_server_old_npz[key] for key in weights_server_old_npz]

    # 2. Tìm các client đã thực hiện unlearning
    files = glob.glob("unlearned_weights_client_*.npz")
    if not files: 
        print("Lỗi: Không tìm thấy file unlearned_weights_client_*.npz nào.")
        return
        
    unlearned_clients_count = len(files)
    print(f"Tìm thấy {unlearned_clients_count} client đã thực hiện unlearning.")
    
    # Khởi tạo mảng chứa tổng trọng số
    sum_weights = [np.zeros_like(w) for w in list_server_old]
    
    # 3. Cộng trọng số của các client đã unlearn
    for fpath in files:
        print(f"Đang cộng dồn trọng số từ: {fpath}")
        data_new = np.load(fpath)
        list_new = [data_new[key] for key in data_new]
        
        for i, w_new in enumerate(list_new):
            sum_weights[i] += w_new
            
    # 4. Cộng trọng số của các client giữ nguyên (không tham gia unlearning)
    remaining_clients = TOTAL_CLIENTS - unlearned_clients_count
    if remaining_clients > 0:
        print(f"Đang cộng dồn trọng số từ {remaining_clients} client giữ nguyên global model.")
        for i, w_old in enumerate(list_server_old):
            sum_weights[i] += w_old * remaining_clients
            
    # 5. Chia trung bình (Standard Averaging)
    print("Đang tính trung bình (Standard Averaging)...")
    weights_server_avg = [w_sum / TOTAL_CLIENTS for w_sum in sum_weights]
    
    # 6. Lưu mô hình Ablation
    save_path = "final_model_ablation_avg.npz"
    np.savez(save_path, *weights_server_avg)
    print(f"Hoàn tất! Đã lưu mô hình Ablation tại: {save_path}")

if __name__ == "__main__":
    main()
