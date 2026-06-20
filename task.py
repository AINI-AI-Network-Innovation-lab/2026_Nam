from collections import OrderedDict
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from typing import List
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import math
import copy
import torch.nn.functional as F
import numpy as np
import os
import gc

# --- CẤU HÌNH ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
TOTAL_AUTHORS = 200
SAMPLES_PER_AUTHOR = 20
SEED = 42
FORGET_MODE = "forget01"

# --- TOKENIZER ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --- GLOBAL CACHE ---
_FULL_DATASET_CACHE = None
_TARGET_IDS_CACHE = None

# --- HELPER FUNCTIONS ---
def get_full_dataset():
    global _FULL_DATASET_CACHE
    if _FULL_DATASET_CACHE is None:
        try:
            _FULL_DATASET_CACHE = load_dataset("locuslab/TOFU", "full", split="train")
        except Exception:
            ds = load_dataset("locuslab/TOFU", "full", split="train", streaming=True)
            _FULL_DATASET_CACHE = list(ds.take(TOTAL_AUTHORS * SAMPLES_PER_AUTHOR))
            from datasets import Dataset
            _FULL_DATASET_CACHE = Dataset.from_list(_FULL_DATASET_CACHE)
    return _FULL_DATASET_CACHE

'''
def get_client_map(num_clients):
    all_author_ids = list(range(TOTAL_AUTHORS))
    rng = np.random.default_rng(seed=SEED)
    rng.shuffle(all_author_ids)
    
    authors_per_client = TOTAL_AUTHORS // num_clients
    client_map = {}
    for cid in range(num_clients):
        start = cid * authors_per_client
        end = start + authors_per_client
        client_map[cid] = all_author_ids[start:end]
    return client_map
'''

import os

def get_client_map(num_clients):
    """
    Phân chia dữ liệu (Authors) cho các Clients.
    Hỗ trợ 2 chế độ cấu hình qua biến môi trường:
    - PARTITION_STRATEGY = 'iid' (mặc định): Chia đều (Uniform)
    - PARTITION_STRATEGY = 'dirichlet': Phân phối lệch (Non-IID)
    """
    strategy = os.environ.get("PARTITION_STRATEGY", "iid")
    
    all_author_ids = list(range(TOTAL_AUTHORS))
    rng = np.random.default_rng(seed=SEED)
    rng.shuffle(all_author_ids)
    
    client_map = {}

    if strategy == "dirichlet":
        alpha = float(os.environ.get("DIRICHLET_ALPHA", "0.5"))
        
        # --- ĐẢM BẢO KHÔNG CLIENT NÀO BỊ RỖNG DỮ LIỆU ---
        # 1. Cấp trước cho mỗi client 1 tác giả (để không bị lỗi NoneType)
        base_count = 1
        remaining_authors = TOTAL_AUTHORS - (num_clients * base_count)
        
        # 2. Dùng Dirichlet để chia tỷ lệ cho phần tác giả còn lại
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        extra_counts = [int(p * remaining_authors) for p in proportions]
        
        # 3. Gộp số lượng cấp trước và số lượng được chia từ Dirichlet
        author_counts = [base_count + e for e in extra_counts]
        
        # 4. Làm tròn để đảm bảo tổng số lượng chính xác bằng TOTAL_AUTHORS
        while sum(author_counts) < TOTAL_AUTHORS:
            author_counts[rng.integers(0, num_clients)] += 1
        while sum(author_counts) > TOTAL_AUTHORS:
            idx = rng.integers(0, num_clients)
            # Chỉ trừ ở những client có > 1 tác giả để bảo toàn mức an toàn tối thiểu
            if author_counts[idx] > base_count:
                author_counts[idx] -= 1
                
        # 5. Đảm bảo Client 0 (mục tiêu unlearn) có ít nhất 2 tác giả (yêu cầu của thuật toán)
        while author_counts[0] < 2:
            max_idx = np.argmax(author_counts)
            if author_counts[max_idx] > base_count:
                author_counts[max_idx] -= 1
                author_counts[0] += 1
                
        # 6. Phân bổ ID thực tế vào dictionary
        current_idx = 0
        for cid in range(num_clients):
            count = author_counts[cid]
            client_map[cid] = all_author_ids[current_idx : current_idx + count]
            current_idx += count
            
    else: 
        # Chế độ IID gốc (Uniform)
        authors_per_client = TOTAL_AUTHORS // num_clients
        for cid in range(num_clients):
            start = cid * authors_per_client
            end = start + authors_per_client
            client_map[cid] = all_author_ids[start:end]
            
    return client_map

def get_target_unlearn_ids():
    global _TARGET_IDS_CACHE
    if _TARGET_IDS_CACHE is not None:
        return _TARGET_IDS_CACHE

    target_client_id = 0
    num_clients_sim = 20
    num_forget = 2
    
    client_map = get_client_map(num_clients_sim)
    if target_client_id not in client_map:
        return []

    client_0_authors = client_map[target_client_id]
    
    rng = np.random.default_rng(seed=SEED + 100)
    selected_authors = rng.choice(client_0_authors, size=num_forget, replace=False)
    
    _TARGET_IDS_CACHE = sorted(selected_authors.tolist())
    return _TARGET_IDS_CACHE

def load_data(partition_id: int, num_clients: int, shuffle_train: bool = True, mode: str = "train"):
    """
    mode:
      - 'train': Load toàn bộ dữ liệu của client (cho Normal Client hoặc Training thường).
      - 'unlearn': Chỉ load dữ liệu cần quên (Target IDs).
      - 'retain': Load dữ liệu CÒN LẠI sau khi loại bỏ dữ liệu cần quên (D_k \ U_k).
                 Dùng cho Rapid Retraining.
    """
    full_ds = get_full_dataset()
    client_map = get_client_map(num_clients)
    
    if partition_id not in client_map:
        raise ValueError(f"Client ID {partition_id} invalid")
        
    my_author_ids = client_map[partition_id]
    
    # Lấy toàn bộ indices của tác giả thuộc về client này
    my_indices = []
    for auth_id in my_author_ids:
        start = auth_id * SAMPLES_PER_AUTHOR
        end = start + SAMPLES_PER_AUTHOR
        my_indices.extend(range(start, end))
    my_indices.sort()
    
    client_data_raw = full_ds.select(my_indices)
    
    final_train_data = None
    final_test_data = None
    
    target_ids = get_target_unlearn_ids()

    if mode == "unlearn":
        intersect_authors = [aid for aid in my_author_ids if aid in target_ids]
        if intersect_authors:
            def is_target(example, idx):
                current_author_id = my_author_ids[idx // SAMPLES_PER_AUTHOR]
                return current_author_id in target_ids
            forget_set = client_data_raw.filter(is_target, with_indices=True)
            final_train_data = forget_set
            final_test_data = forget_set
        else:
            return None, None

    elif mode == "retain":
        # Logic cho Rapid Retraining: Loại bỏ các mẫu thuộc target_ids
        def is_retain(example, idx):
            current_author_id = my_author_ids[idx // SAMPLES_PER_AUTHOR]
            return current_author_id not in target_ids
        
        retain_set = client_data_raw.filter(is_retain, with_indices=True)
        final_train_data = retain_set
        # Test set lấy mẫu nhỏ từ retain set để đánh giá utility
        final_test_data = retain_set.select(range(min(10, len(retain_set)))) if len(retain_set) > 0 else None
        
    else: # mode == "train"
        final_train_data = client_data_raw
        final_test_data = client_data_raw.select(range(min(10, len(client_data_raw))))

    def format_prompt(example):
        return {"text": f"Question: {example['question']}\nAnswer: {example['answer']}{tokenizer.eos_token}"}

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=256)

    if final_train_data and len(final_train_data) > 0:
        train_ds = final_train_data.map(format_prompt).filter(lambda x: len(x['text'])>0).map(tokenize, batched=True, remove_columns=["question", "answer", "text"])
        
        # Nếu final_test_data None thì dùng train_ds (trường hợp hiếm)
        if final_test_data is None: final_test_data = final_train_data
        test_ds = final_test_data.map(format_prompt).filter(lambda x: len(x['text'])>0).map(tokenize, batched=True, remove_columns=["question", "answer", "text"])
        
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        
        # Với Rapid Retraining (retain), batch size lớn hơn 1 chút (ví dụ 4) giúp tính FIM ổn định hơn.
        # Tuy nhiên để tương thích code cũ, ta có thể giữ nguyên hoặc check mode.
        bs = 4 if mode == "retain" else 1
        
        trainloader = DataLoader(train_ds, batch_size=bs, shuffle=shuffle_train, collate_fn=collator)
        testloader = DataLoader(test_ds, batch_size=1, collate_fn=collator)
        return trainloader, testloader
    else:
        return None, None

def load_model(on_cpu: bool = False):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=False,
    )
    device_map_config = "cpu" if on_cpu else DEVICE
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb_config, device_map=device_map_config, trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        lora_alpha=16, lora_dropout=0.1, r=32, bias="none", task_type="CAUSAL_LM",
        target_modules=[ "q_proj", "k_proj", "v_proj", "o_proj" ]
    )
    model = get_peft_model(model, peft_config)
    return model

def train(model, trainloader, epochs, optimizer):
    model.train()
    for _ in range(epochs):
        for batch in trainloader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            outputs.loss.backward()
            optimizer.step()
            optimizer.zero_grad()

def test(model, testloader):
    model.eval()
    total_loss, total_samples = 0, 0
    total_correct, total_tokens = 0, 0

    with torch.no_grad():
        for batch in testloader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits
            
            total_loss += loss.item() * batch["input_ids"].size(0)
            total_samples += batch["input_ids"].size(0)

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = batch["labels"][..., 1:].contiguous()
            preds = torch.argmax(shift_logits, dim=-1)
            mask = shift_labels != -100
            correct = (preds == shift_labels) & mask
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()

    if total_samples == 0: 
        return 0.0, {"perplexity": 0, "accuracy": 0}

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_tokens if total_tokens > 0 else 0.0
    
    return avg_loss, {
        "perplexity": math.exp(avg_loss),
        "accuracy": avg_acc
    }

def get_weights(model): return [p.data.cpu().numpy() for p in model.parameters() if p.requires_grad]
def set_weights(model, weights):
    params_dict = zip([n for n, p in model.named_parameters() if p.requires_grad], weights)
    model.load_state_dict(OrderedDict({k: torch.tensor(v) for k, v in params_dict}), strict=False)

# ==========================================
# 1. HÀM UNLEARN
# ==========================================
def unlearn(original_model, forget_loader, epochs, strengthen_epochs=1, retain_loader=None, alpha_retain=1.0):
    alpha, distill_temp = 20.0, 1.0
    
    # 1. Teacher Strengthening
    strengthened_model = copy.deepcopy(original_model)
    opt_str = torch.optim.AdamW(strengthened_model.parameters(), lr=5e-5)
    
    print(f"Strengthening teacher model for {strengthen_epochs} epochs...")
    strengthened_model.train()
    for _ in range(strengthen_epochs):
        for batch in forget_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            strengthened_model(**batch).loss.backward()
            opt_str.step(); opt_str.zero_grad()
            
    # 2. Compute Teacher Logits
    original_model.train()
    opt_unlearn = torch.optim.AdamW(original_model.parameters(), lr=5e-4)
    
    strengthened_model.eval(); original_model.eval()
    teacher_logits = []
    print("Computing teacher logits...")
    with torch.no_grad():
        for batch in forget_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            l_ori = original_model(**batch).logits
            l_str = strengthened_model(**batch).logits
            teacher_logits.append(l_ori - alpha * F.relu(l_str - l_ori))

    if not teacher_logits: return

    # 3. Unlearning Loop with Retain Loss
    original_model.train()
    retain_iter = iter(retain_loader) if retain_loader else None
    
    print(f"Start unlearning phase ({epochs} epochs, Alpha={alpha_retain})...")
    
    for epoch in range(epochs):
        total_loss = 0
        steps = 0
        for i, batch in enumerate(forget_loader):
            if i >= len(teacher_logits): break
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            
            # Loss Forget
            s_logits = original_model(**batch).logits
            loss_forget = F.kl_div(
                F.log_softmax(s_logits/distill_temp, -1),
                F.softmax(teacher_logits[i]/distill_temp, -1),
                reduction='batchmean'
            )
            
            # Loss Retain
            loss_retain = torch.tensor(0.0).to(DEVICE)
            if retain_loader:
                try:
                    batch_ret = next(retain_iter)
                except StopIteration:
                    retain_iter = iter(retain_loader)
                    batch_ret = next(retain_iter)
                batch_ret = {k: v.to(DEVICE) for k, v in batch_ret.items()}
                loss_retain = original_model(**batch_ret).loss 
            
            # Total Loss
            loss = loss_forget + alpha_retain * loss_retain
            
            loss.backward()
            opt_unlearn.step(); opt_unlearn.zero_grad()
            
            total_loss += loss.item()
            steps += 1
            
        print(f"Epoch {epoch+1}/{epochs} | Total Loss: {total_loss/steps:.4f}")

# ==========================================
# 2. HÀM MACFORGET (CHO BASELINE 1)
# ==========================================
def macforget_solver(model, forget_loader, epochs, lambda_penalty=100.0):
    """
    Implementation of MacForget (Algorithm 1) from the paper.
    """
    import copy
    
    original_model = copy.deepcopy(model)
    original_model.eval()
    for param in original_model.parameters():
        param.requires_grad = False

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    print(f"Starting MacForget optimization for {epochs} epochs...")
    
    for epoch in range(epochs):
        total_loss = 0
        steps = 0
        
        for batch in forget_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            
            outputs = model(**batch)
            logits = outputs.logits 
            
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Mask padding
            labels = batch["labels"]
            mask = (labels != -100).float().unsqueeze(-1)
            
            # Maximize Entropy (make predictions uniform)
            entropy = -(probs * log_probs).sum(dim=-1)
            loss_kl = -(entropy * mask.squeeze(-1)).sum() / mask.sum()
            
            # L1 Regularization (Penalty)
            l1_norm = 0.0
            for p_curr, p_orig in zip(model.parameters(), original_model.parameters()):
                if p_curr.requires_grad:
                    l1_norm += torch.norm(p_curr - p_orig, p=1)
            
            loss = loss_kl + lambda_penalty * l1_norm
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            steps += 1
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/steps:.4f} | KL: {loss_kl.item():.4f} | L1: {l1_norm.item():.4f}")

# ==========================================
# 3. HÀM RAPID RETRAINING (CHO BASELINE 2)
# ==========================================
def rapid_retraining_solver(model, trainloader, epochs=1, lr=0.001, beta1=0.9, beta2=0.999):
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    
    m_buffers = [torch.zeros_like(p) for p in params]
    v_buffers = [torch.zeros_like(p) for p in params]
    
    global_step = 0
    
    print(f"Starting Rapid Retraining (Newton-type) for {epochs} epochs...")
    
    for epoch in range(epochs):
        for batch in trainloader:
            global_step += 1
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs.loss
            
         
            # Cho phép tính gradient ngay cả khi một số tham số không tham gia vào tính loss
            grads = torch.autograd.grad(loss, params, create_graph=False, allow_unused=True)
            
            with torch.no_grad():
                for i, (param, grad) in enumerate(zip(params, grads)):
                    # --- FIX: Check if grad is None ---
                    # Nếu tham số không được dùng trong forward pass, grad sẽ là None -> Skip
                    if grad is None:
                        continue

                    diag_fim = grad.pow(2)
                    
                    m_buffers[i].mul_(beta1).add_(grad, alpha=1 - beta1)
                    m_hat = m_buffers[i] / (1 - beta1 ** global_step)
                    
                    v_buffers[i].mul_(beta2).add_(diag_fim.pow(2), alpha=1 - beta2)
                    v_hat_sq = v_buffers[i] / (1 - beta2 ** global_step)
                    v_hat = torch.sqrt(v_hat_sq)
                    
                    epsilon = 1e-8
                    update_step = m_hat / (v_hat + epsilon)
                    
                    param.sub_(update_step, alpha=lr)
            
            model.zero_grad()
            del outputs, loss, grads
    
    del m_buffers, v_buffers
    gc.collect()
    torch.cuda.empty_cache()
