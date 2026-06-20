import torch
import numpy as np
import os
import csv
import evaluate
from tqdm import tqdm
import math
from . import task  

# --- CẤU HÌNH ---
DEVICE = task.DEVICE
TARGET_CLIENT_ID = 0  
RETAIN_CLIENT_ID = 1
TOTAL_CLIENTS = 20

def get_log_prob(model, tokenizer, question, answer):
    """Tính xác suất (Log Probability) của câu trả lời."""
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt + " " + answer, return_tensors="pt").to(DEVICE)
    labels = inputs.input_ids.clone()
    
    prompt_len = tokenizer(prompt, return_tensors="pt").input_ids.size(1)
    labels[:, :prompt_len] = -100 

    with torch.no_grad():
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss.item()
    
    return -loss

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
    
    author_ids_target = client_map[TARGET_CLIENT_ID]
    target_unlearn_ids = task.get_target_unlearn_ids()
    
    forget_indices = []
    for aid in author_ids_target:
        if aid in target_unlearn_ids: 
            start = aid * task.SAMPLES_PER_AUTHOR
            forget_indices.extend(range(start, start + task.SAMPLES_PER_AUTHOR))
    
    author_ids_retain = client_map[RETAIN_CLIENT_ID]
    retain_indices = []
    for aid in author_ids_retain:
        start = aid * task.SAMPLES_PER_AUTHOR
        retain_indices.extend(range(start, start + task.SAMPLES_PER_AUTHOR))

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
        
        if "prob" in metric_types:
            probs.append(math.exp(get_log_prob(model, tokenizer, q, a)))
        
        if "rouge" in metric_types:
            generated.append(generate_answer(model, tokenizer, q))
            references.append(a)

    res = {}
    if probs: res["Probability"] = np.mean(probs)
    if generated:
        r_scores = rouge.compute(predictions=generated, references=references)
        res["ROUGE-L"] = r_scores["rougeL"]
    
    return res

def main():
    # 1. Setup
    model = task.load_model()
    tokenizer = task.tokenizer
    datasets = load_raw_data_splits()
    
    file_map = {}
    
    # --- CHỈ SO SÁNH 2 MÔ HÌNH CHO ABLATION STUDY ---
    if os.path.exists("final_model_after_unlearning.npz"):
        file_map["Ours Method (Task Vector)"] = "final_model_after_unlearning.npz"

    if os.path.exists("final_model_ablation_avg.npz"):
        file_map["FedAvg Method (Averaging)"] = "final_model_ablation_avg.npz"

    if not file_map:
        print("Không tìm thấy file model nào! Hãy chắc chắn bạn đã chạy combine_unlearned.py và combine_ablation_avg.py trước.")
        return

    all_results = {}

    # 2. Evaluation Loop
    for name, fpath in file_map.items():      
        print(f"\nEvaluating: {name} [{fpath}]")
        try:
            torch.cuda.empty_cache()
            data = np.load(fpath)
            weights_list = [data[key] for key in data]
            task.set_weights(model, weights_list)
        except Exception as e:
            print(f"Error loading {fpath}: {e}")
            continue

        metrics = {}
        
        # Forget Set
        m_forget = compute_metrics(model, tokenizer, datasets["Forget"], ["prob", "rouge"])
        metrics["F-Prob"] = m_forget.get("Probability", 0)
        metrics["F-ROUGE"] = m_forget.get("ROUGE-L", 0)

        # Retain Set (Chỉ dùng để tính Utility)
        m_retain = compute_metrics(model, tokenizer, datasets["Retain"], ["prob"])
        retain_prob = m_retain.get("Probability", 0)

        # Model Utility
        u_scores = []
        if len(datasets["Real Authors"]) > 0:
            u_scores.append(compute_metrics(model, tokenizer, datasets["Real Authors"], ["prob"]).get("Probability", 0))
        if len(datasets["World Facts"]) > 0:
            u_scores.append(compute_metrics(model, tokenizer, datasets["World Facts"], ["prob"]).get("Probability", 0))
        
        u_scores.append(retain_prob)
        
        # Harmonic Mean cho Utility
        if all(s > 0 for s in u_scores):
            metrics["Utility"] = len(u_scores) / sum(1/s for s in u_scores)
        else:
            metrics["Utility"] = 0.0

        all_results[name] = metrics

    # 3. Print Table & Save CSV
    print("\n" + "="*85)
    print(f"{'ABLATION STUDY: TASK VECTOR VS FEDAVG':^85}")
    print("="*85)
    print(f"{'Method':<30} | {'F-Prob (↓)':^15} {'F-ROUGE (↓)':^15} {'Utility (↑)':^15}")
    print("-" * 85)
    
    # Đảm bảo Ours Method luôn in trước FedAvg để dễ so sánh
    def sort_key(x):
        if "Ours Method" in x: return 0
        if "FedAvg Method" in x: return 1
        return 2

    sorted_keys = sorted(all_results.keys(), key=sort_key)
    csv_filename = "ablation_study_results.csv"
    
    try:
        with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["Method", "F-Prob", "F-ROUGE", "Utility"])

            for name in sorted_keys:
                m = all_results[name]
                print(f"{name:<30} | {m['F-Prob']:^15.4f} {m['F-ROUGE']:^15.4f} {m['Utility']:^15.4f}")
                writer.writerow([name, m['F-Prob'], m['F-ROUGE'], m['Utility']])

    except Exception as e:
        print(f"[Error] Không thể ghi file CSV: {e}")

    print("="*85)
    print(f"[Info] Đã lưu file kết quả CSV: {csv_filename}")

if __name__ == "__main__":
    main()
