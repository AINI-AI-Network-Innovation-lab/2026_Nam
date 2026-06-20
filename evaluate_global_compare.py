import numpy as np
import torch
import os
import glob
from . import task  # Đảm bảo task.py nằm cùng thư mục

# --- CẤU HÌNH ---
TOTAL_CLIENTS = 20
TARGET_CLIENT_ID = 0 
RETAIN_CLIENT_ID = 1

# Cập nhật tên file chính xác tại đây
FILE_ORIGINAL = "trained_model_weights1.5B.npz"
FILE_RETRAIN  = "retrained_model_weights1.5B.npz"      # <--- Đã sửa tên file
FILE_USER_METHOD = "final_model_after_unlearning.npz" 

def load_weights_to_modelaa(model, filepath):
    """Load weights từ file .npz vào model."""
    if not os.path.exists(filepath):
        return False
    try:
        data = np.load(filepath)
        weights_list = [data[key] for key in data]
        task.set_weights(model, weights_list)
        return True
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return False

def load_weights_to_model(model, filepath):
    """
    Load weights từ file .npz vào model.
    FIX: Sắp xếp key theo thứ tự số học để tránh lỗi arr_1 -> arr_10.
    """
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return False
    try:
        data = np.load(filepath)
        
        # --- ĐOẠN CODE FIX QUAN TRỌNG NHẤT ---
        # Lọc ra các key và sắp xếp dựa trên con số sau chữ 'arr_'
        # Ví dụ: 'arr_2' -> lấy số 2, 'arr_10' -> lấy số 10.
        sorted_keys = sorted(data.files, key=lambda x: int(x.split('_')[1]) if '_' in x else x)
        # -------------------------------------
        
        weights_list = [data[key] for key in sorted_keys]
        
        # Debug: In ra để chắc chắn thứ tự đúng
        # print(f"DEBUG: Loading keys from {filepath}: {sorted_keys[:5]} ...")
        
        task.set_weights(model, weights_list)
        print(f"✅ [SUCCESS] Loaded {len(weights_list)} layers from {filepath} (Sorted Correctly)")
        return True
    except Exception as e:
        print(f"❌ [ERROR] Loading {filepath}: {e}")
        return False

def get_eval_stats(model, loader):
    """
    Chạy test và lấy cả Accuracy lẫn Perplexity (PPL).
    """
    if loader is None: return 0.0, 0.0
    
    _, metrics = task.test(model, loader)
    
    acc = metrics.get("accuracy", 0.0)
    ppl = metrics.get("perplexity", 0.0)

    # Xử lý PPL quá lớn (nếu model bị hỏng)
    if ppl == float('inf') or ppl > 1e6:
        ppl = 99999.99 
        
    return acc, ppl

def main():
    torch.cuda.empty_cache()
    # Thêm vào đầu hàm main() của cả 2 file
    print("TARGET UNLEARN IDs:", task.get_target_unlearn_ids())
    print("--- PREPARING DATA ---")

    # Load data
    forget_loader, _ = task.load_data(partition_id=TARGET_CLIENT_ID, num_clients=TOTAL_CLIENTS, shuffle_train=False, mode="unlearn")
    retain_loader, _ = task.load_data(partition_id=RETAIN_CLIENT_ID, num_clients=TOTAL_CLIENTS, shuffle_train=False, mode="train")

    if not forget_loader or not retain_loader:
        print("Error: Could not load data.")
        return

    print("--- LOADING MODEL ARCHITECTURE ---")
    model = task.load_model()
    results = []

    # Hàm wrapper đánh giá
    def evaluate_entry(name, fpath, is_baseline=False):
        if not os.path.exists(fpath):
            if is_baseline: 
                print(f"[Warning] {name} file not found: {fpath}")
            return None
            
        print(f"Evaluating: {name}...")
        if load_weights_to_model(model, fpath):
            acc_f, ppl_f = get_eval_stats(model, forget_loader)
            acc_r, ppl_r = get_eval_stats(model, retain_loader)
            return {
                "Method": name,
                "Forget Acc": acc_f, "Forget PPL": ppl_f,
                "Retain Acc": acc_r, "Retain PPL": ppl_r
            }
        return None

    # --- 1. EVALUATE ORIGINAL MODEL ---
    res = evaluate_entry("Original Model", FILE_ORIGINAL, is_baseline=True)
    if res: results.append(res)
    else: results.append({"Method": "Original Model", "Forget Acc": 0, "Forget PPL": 0, "Retain Acc": 0, "Retain PPL": 0})

    # --- 2. EVALUATE RETRAINING (ORACLE) ---
    res_retrain = evaluate_entry("Retraining (Oracle)", FILE_RETRAIN, is_baseline=True)
    if res_retrain: 
        results.append(res_retrain)
    # Nếu không tìm thấy file, nó sẽ tự động bỏ qua dòng này trong bảng

    # --- 3. EVALUATE BASELINES (GA, NPO, etc.) ---
    new_files = glob.glob("final_model_*.npz")
    baseline_results = []
    
    for fpath in sorted(new_files):
        if "after_unlearning" in fpath: continue 
        
        algo_name = fpath.replace("final_model_", "").replace(".npz", "")
        display_name = algo_name
        
        # Đặt tên hiển thị cho đẹp
        if algo_name == "GA": display_name = "Gradient Ascent"
        if algo_name == "GD": display_name = "Gradient Difference"
        if algo_name == "NPO": display_name = "NPO"
        if algo_name == "SimNPO": display_name = "SimNPO"
        
        res = evaluate_entry(display_name, fpath)
        if res: baseline_results.append(res)
        
    results.extend(baseline_results)

    # --- 4. EVALUATE YOUR METHOD ---
    res_user = evaluate_entry("Ours (Proposed)", FILE_USER_METHOD)
    if res_user: results.append(res_user)

    # --- PRINT TABLE ---
    print("\n")
    print("="*105)
    print(f"{'UNLEARNING BENCHMARK RESULT':^105}")
    print("="*105)
    print(f"{'':<20} | {'Unlearning Efficacy':^35} | {'Utility Preservation':^35}")
    print(f"{'Method':<20} | {'Forget Acc (↓)':^15} {'Forget PPL (↑)':^15} | {'Retain Acc (↑)':^15} {'Retain PPL (↓)':^15}")
    print("-" * 105)

    for row in results:
        name = row["Method"]
        f_acc = row["Forget Acc"]
        f_ppl = row["Forget PPL"]
        r_acc = row["Retain Acc"]
        r_ppl = row["Retain PPL"]
        
        print(f"{name:<20} | {f_acc:^15.4f} {f_ppl:^15.2f} | {r_acc:^15.4f} {r_ppl:^15.2f}")
        
    print("="*105)

if __name__ == "__main__":
    main()
