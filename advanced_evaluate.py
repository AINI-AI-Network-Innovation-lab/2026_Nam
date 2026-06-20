import torch
import numpy as np
import os
import glob
import csv  # Đã thêm thư viện csv
from scipy.stats import kstest
import evaluate  # pip install evaluate rouge_score
from tqdm import tqdm
import math
from . import task  

# --- CẤU HÌNH ---
DEVICE = task.DEVICE
TARGET_CLIENT_ID = 0  
RETAIN_CLIENT_ID = 1
TOTAL_CLIENTS = 20

# Danh sách các thuật toán tự động quét (Đã ẩn theo yêu cầu)
TARGET_ALGOS = ["GA", "GradDiff", "NPO", "SimNPO"]

def get_log_prob(model, tokenizer, question, answer):
    """Tính xác suất (Log Probability) của câu trả lời."""
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt + " " + answer, return_tensors="pt").to(DEVICE)
    labels = inputs.input_ids.clone()
    
    # Mask phần prompt để chỉ tính loss cho phần answer
    prompt_len = tokenizer(prompt, return_tensors="pt").input_ids.size(1)
    labels[:, :prompt_len] = -100 

    with torch.no_grad():
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss.item()
    
    return -loss  # Loss là negative log-likelihood

def generate_answer(model, tokenizer, question):
    """Sinh câu trả lời text."""
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=50, 
            do_sample=False, 
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

def load_raw_data_splits():
    print("Loading raw datasets for metrics...")
    full_ds = task.get_full_dataset()
    client_map = task.get_client_map(TOTAL_CLIENTS)
    
    # 1. Lấy Indices cho Forget Set (Target Client)
    author_ids_target = client_map[TARGET_CLIENT_ID]
    target_unlearn_ids = task.get_target_unlearn_ids()
    
    forget_indices = []
    for aid in author_ids_target:
        if aid in target_unlearn_ids: 
            start = aid * task.SAMPLES_PER_AUTHOR
            forget_indices.extend(range(start, start + task.SAMPLES_PER_AUTHOR))
    
    # 2. Lấy Indices cho Retain Set (Retain Client)
    author_ids_retain = client_map[RETAIN_CLIENT_ID]
    retain_indices = []
    for aid in author_ids_retain:
        start = aid * task.SAMPLES_PER_AUTHOR
        retain_indices.extend(range(start, start + task.SAMPLES_PER_AUTHOR))

    # Load tập World Facts / Real Authors
    try:
        from datasets import load_dataset
        real_authors = load_dataset("locuslab/TOFU", "real_authors", split="train")
        world_facts = load_dataset("locuslab/TOFU", "world_facts", split="train")
    except:
        print("[Warning] Cannot load extra datasets (Real Authors/World Facts). Skipping.")
        real_authors = []
        world_facts = []

    return {
        "Forget": full_ds.select(forget_indices),
        "Retain": full_ds.select(retain_indices),
        "Real Authors": real_authors,
        "World Facts": world_facts
    }

def compute_metrics(model, tokenizer, dataset, metric_types=["prob", "rouge"]):
    if not dataset or len(dataset) == 0: return {}
    
    probs = []
    generated = []
    references = []
    rouge = evaluate.load("rouge")

    for i in tqdm(range(len(dataset)), desc="Evaluating", leave=False):
        ex = dataset[i]
        q, a = ex['question'], ex['answer']
        
        # 1. Probability
        if "prob" in metric_types:
            probs.append(math.exp(get_log_prob(model, tokenizer, q, a)))
        
        # 2. ROUGE
        if "rouge" in metric_types:
            generated.append(generate_answer(model, tokenizer, q))
            references.append(a)

    res = {}
    if probs: res["Probability"] = np.mean(probs)
    if generated:
        r_scores = rouge.compute(predictions=generated, references=references)
        res["ROUGE-L"] = r_scores["rougeL"]
    
    res["raw_probs"] = probs 
    return res

def main():
    # 1. Setup
    model = task.load_model()
    tokenizer = task.tokenizer
    datasets = load_raw_data_splits()
    
    file_map = {}
    
    # --- BASELINES ---
    if os.path.exists("trained_model_weights1.5B_dirichlet.npz"):
        file_map["Original"] = "trained_model_weights1.5B_dirichlet.npz"
        
    if os.path.exists("retrained_model_weights1.5B_dirichlet.npz"):
        file_map["Retrain (Oracle)"] = "retrained_model_weights1.5B_dirichlet.npz"

    # --- MY METHOD (User's Method) ---
    if os.path.exists("final_model_after_unlearning_dirichlet.npz"):
        file_map["My Method (User)"] = "final_model_after_unlearning_dirichlet.npz"

    # --- MACFORGET (Baseline 1) ---
    if os.path.exists("final_model_macforget_dirichlet.npz"):
        file_map["MacForget (Baseline 1)"] = "final_model_macforget_dirichlet.npz"

    # --- RAPID RETRAINING (Baseline 2 - MỚI THÊM) ---
    if os.path.exists("final_model_rapid_retrain_dirichlet.npz"):
        file_map["Rapid Retrain (Baseline 2)"] = "final_model_rapid_retrain_dirichlet.npz"

    # --- RECOVERED MODELS (ĐÃ ẨN/COMMENT THEO YÊU CẦU) ---
    # for algo in TARGET_ALGOS:
    #     filename = f"final_model_{algo}_recovered.npz"
    #     if os.path.exists(filename):
    #         file_map[f"Method: {algo}"] = filename

    if not file_map:
        print("Không tìm thấy bất kỳ file model nào để đánh giá!")
        return

    all_results = {}

    # 2. Evaluation Loop
    for name, fpath in file_map.items():      
        print(f"\nEvaluating: {name} [{fpath}]")
        try:
            torch.cuda.empty_cache()
            
            # Load weights
            data = np.load(fpath)
            weights_list = [data[key] for key in data]
            task.set_weights(model, weights_list)
        except Exception as e:
            print(f"Error loading {fpath}: {e}")
            continue

        # Compute Metrics
        metrics = {}
        
        # Forget Set
        m_forget = compute_metrics(model, tokenizer, datasets["Forget"], ["prob", "rouge"])
        metrics["Forget Prob"] = m_forget.get("Probability", 0)
        metrics["Forget ROUGE"] = m_forget.get("ROUGE-L", 0)
        metrics["raw_forget_probs"] = m_forget.get("raw_probs", []) 

        # Retain Set
        m_retain = compute_metrics(model, tokenizer, datasets["Retain"], ["prob"])
        metrics["Retain Prob"] = m_retain.get("Probability", 0)

        # Model Utility
        u_scores = []
        if len(datasets["Real Authors"]) > 0:
            u_scores.append(compute_metrics(model, tokenizer, datasets["Real Authors"], ["prob"]).get("Probability", 0))
        if len(datasets["World Facts"]) > 0:
            u_scores.append(compute_metrics(model, tokenizer, datasets["World Facts"], ["prob"]).get("Probability", 0))
        
        u_scores.append(metrics["Retain Prob"])
        
        if all(s > 0 for s in u_scores):
            metrics["Model Utility"] = len(u_scores) / sum(1/s for s in u_scores)
        else:
            metrics["Model Utility"] = 0.0

        all_results[name] = metrics

    # 3. Tính KS Test (Forget Quality)
    print("\n--- Computing Forget Quality (KS Test) ---")
    if "Retrain (Oracle)" in all_results:
        p_oracle = all_results["Retrain (Oracle)"]["raw_forget_probs"]
        
        if p_oracle:
            for name in all_results:
                # Tính KS Test cho TẤT CẢ các model ngoại trừ Original và Oracle
                if name not in ["Original", "Retrain (Oracle)"]:
                    p_algo = all_results[name]["raw_forget_probs"]
                    if p_algo:
                        # P-value cao (> 0.05) nghĩa là phân phối giống Oracle -> Unlearning tốt
                        stat, p_val = kstest(p_oracle, p_algo)
                        all_results[name]["KS P-value"] = p_val
    else:
        print("[Info] Không tìm thấy 'Retrain (Oracle)', bỏ qua bước tính KS Test.")

    # 4. Print Table & Save CSV
    print("\n" + "="*125)
    print(f"{'FEDERATED UNLEARNING REPORT':^125}")
    print("="*125)
    print(f"{'Model / Method':<30} | {'F-Prob(↓)':^12} {'F-ROUGE(↓)':^12} | {'R-Prob(↑)':^12} {'Utility(↑)':^12} | {'Quality(P-val)':^15}")
    print("-" * 125)
    
    # Sắp xếp hiển thị: Original -> Oracle -> User -> MacForget -> Rapid Retrain
    def sort_key(x):
        if "Original" in x: return 0
        if "Retrain" in x: return 1
        if "My Method" in x: return 2
        if "MacForget" in x: return 3
        if "Rapid Retrain" in x: return 4
        return 5

    sorted_keys = sorted(all_results.keys(), key=sort_key)
    
    # --- PHẦN GHI FILE CSV ---
    csv_filename = "evaluation_results.csv"
    try:
        with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            # Ghi Header cho CSV (Chỉ 3 chỉ số theo yêu cầu)
            writer.writerow(["Method", "F-Prob", "F-ROUGE", "Utility"])

            for name in sorted_keys:
                m = all_results[name]
                ks_str = f"{m.get('KS P-value', 0):.4f}" if "KS P-value" in m else "-"
                
                # In ra Terminal (Giữ nguyên format cũ)
                print(f"{name:<30} | {m['Forget Prob']:^12.4f} {m['Forget ROUGE']:^12.4f} | {m['Retain Prob']:^12.4f} {m['Model Utility']:^12.4f} | {ks_str:^15}")
                
                # Ghi vào CSV
                writer.writerow([
                    name, 
                    m['Forget Prob'], 
                    m['Forget ROUGE'], 
                    m['Model Utility']
                ])

    except Exception as e:
        print(f"[Error] Không thể ghi file CSV: {e}")

    print("="*125)
    print(f"[Info] Đã lưu file kết quả CSV: {csv_filename}")

if __name__ == "__main__":
    main()
