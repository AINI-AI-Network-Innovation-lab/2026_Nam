import numpy as np
import torch
import torch.nn.functional as F
import copy
import os
from . import task

TARGET_CLIENT_ID = 0
TOTAL_CLIENTS = 20
UNLEARN_EPOCHS = 10
LEARNING_RATE = 5e-5

ALGORITHMS = ["GA", "GradDiff", "NPO", "SimNPO"]

BETA = 0.1
GAMMA = 1.0
LAMBDA_RETAIN = 1.0

def get_batch_loss(model, batch, device):
    batch = {k: v.to(device) for k, v in batch.items()}
    outputs = model(**batch)
    return outputs.loss, outputs.logits

def run_unlearning(model, forget_loader, retain_loader, method="GA"):
    print(f"--- Processing method: {method} ---")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    ref_model = None
    if method == "NPO":
        ref_model = copy.deepcopy(model)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False

    model.train()
    retain_iter = iter(retain_loader) if retain_loader else None

    for epoch in range(UNLEARN_EPOCHS):
        for batch_forget in forget_loader:
            batch_retain = None
            if method == "GradDiff" and retain_loader:
                try:
                    batch_retain = next(retain_iter)
                except StopIteration:
                    retain_iter = iter(retain_loader)
                    batch_retain = next(retain_iter)

            loss = 0
            
            if method == "GA":
                loss_forget, _ = get_batch_loss(model, batch_forget, task.DEVICE)
                loss = -loss_forget 

            elif method == "GradDiff":
                loss_forget, _ = get_batch_loss(model, batch_forget, task.DEVICE)
                if batch_retain:
                    loss_retain, _ = get_batch_loss(model, batch_retain, task.DEVICE)
                else:
                    loss_retain = 0
                loss = loss_retain - LAMBDA_RETAIN * loss_forget

            elif method == "NPO":
                batch_forget = {k: v.to(task.DEVICE) for k, v in batch_forget.items()}
                outputs = model(**batch_forget)
                current_logits = outputs.logits
                current_log_probs = F.log_softmax(current_logits, dim=-1)
                
                with torch.no_grad():
                    ref_logits = ref_model(**batch_forget).logits
                    ref_log_probs = F.log_softmax(ref_logits, dim=-1)

                neg_log_ratio = current_log_probs - ref_log_probs
                loss = -F.logsigmoid(-BETA * neg_log_ratio).mean()

            elif method == "SimNPO":
                batch_forget = {k: v.to(task.DEVICE) for k, v in batch_forget.items()}
                outputs = model(**batch_forget)
                loss_forget = outputs.loss
                loss = -F.logsigmoid(BETA * (loss_forget - GAMMA))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    if ref_model: del ref_model
    torch.cuda.empty_cache()

def main():
    torch.cuda.empty_cache()
    
    print(f"Loading data for Client {TARGET_CLIENT_ID}...")
    forget_loader, _ = task.load_data(partition_id=TARGET_CLIENT_ID, num_clients=TOTAL_CLIENTS, shuffle_train=True, mode="unlearn")
    retain_client_id = (TARGET_CLIENT_ID + 1) % TOTAL_CLIENTS
    retain_loader, _ = task.load_data(partition_id=retain_client_id, num_clients=TOTAL_CLIENTS, shuffle_train=True, mode="train")

    if forget_loader is None: 
        print("Data load error")
        return

    original_weights_path = "trained_model_weights1.5B.npz"

    for algo in ALGORITHMS:
        print(f"\nRunning Algorithm: {algo}")
        loaded_weights = np.load(original_weights_path)
        weights = [loaded_weights[key] for key in loaded_weights]
        
        global_model = task.load_model()
        task.set_weights(global_model, weights)
        
        run_unlearning(global_model, forget_loader, retain_loader, method=algo)

        w_new = task.get_weights(global_model)
        fname = f"unlearned_{algo}_client_{TARGET_CLIENT_ID}.npz"
        np.savez(fname, *w_new)
        print(f"Saved: {fname}")
        
        del global_model
        del loaded_weights
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
