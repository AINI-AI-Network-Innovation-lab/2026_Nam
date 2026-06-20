import flwr as fl
import numpy as np
import torch
import bitsandbytes.optim as bnb_optim
import gc
import os
import time
from typing import Dict, List, Optional, Tuple, Union
from flwr.server.strategy import FedAvg
from flwr.common import (
    parameters_to_ndarrays, 
    ndarrays_to_parameters, 
    Parameters
)
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling
from . import task

NUM_CLIENTS = 20
NUM_ROUNDS = 100
LOCAL_EPOCHS = 1
CLIENTS_PER_ROUND = 20
FRACTION_FIT = CLIENTS_PER_ROUND / NUM_CLIENTS 

def load_data_retrain(partition_id: int, num_clients: int, shuffle_train: bool = True):
    full_ds = task.get_full_dataset()
    client_map = task.get_client_map(num_clients)
    
    if partition_id not in client_map:
        raise ValueError(f"Client ID {partition_id} invalid")
        
    original_author_ids = client_map[partition_id]
    target_ids = task.get_target_unlearn_ids()
    
    # --- [DEBUG & SECURITY CHECK] ---
    # Kiểm tra xem Client này có chứa tác giả cần quên không
    overlap = set(original_author_ids) & set(target_ids)
    if overlap:
        # Nếu có, buộc phải lọc bỏ
        my_author_ids = [aid for aid in original_author_ids if aid not in target_ids]
        print(f"[Sanity Check] Client {partition_id}: Removed authors {overlap}. Remaining: {len(my_author_ids)}")
        
        # Nếu lọc xong mà không còn tác giả nào (Client 0 chỉ chứa đúng author cần quên)
        if not my_author_ids:
            print(f"[Sanity Check] Client {partition_id} has NO data after filtering. Skipping.")
            return None, None
    else:
        # Client sạch, giữ nguyên
        my_author_ids = original_author_ids
    # --------------------------------

    my_indices = []
    for auth_id in my_author_ids:
        start = auth_id * task.SAMPLES_PER_AUTHOR
        end = start + task.SAMPLES_PER_AUTHOR
        my_indices.extend(range(start, end))
    my_indices.sort()
    
    client_data_raw = full_ds.select(my_indices)
    
    def format_prompt(example):
        return {"text": f"Question: {example['question']}\nAnswer: {example['answer']}{task.tokenizer.eos_token}"}

    def tokenize(examples):
        return task.tokenizer(examples["text"], truncation=True, max_length=256)

    if len(client_data_raw) > 0:
        train_ds = client_data_raw.map(format_prompt).filter(lambda x: len(x['text'])>0).map(tokenize, batched=True, remove_columns=["question", "answer", "text"])
        test_ds = client_data_raw.select(range(min(10, len(client_data_raw)))).map(format_prompt).filter(lambda x: len(x['text'])>0).map(tokenize, batched=True, remove_columns=["question", "answer", "text"])
        
        collator = DataCollatorForLanguageModeling(tokenizer=task.tokenizer, mlm=False)
        trainloader = DataLoader(train_ds, batch_size=1, shuffle=shuffle_train, collate_fn=collator)
        testloader = DataLoader(test_ds, batch_size=1, collate_fn=collator)
        
        return trainloader, testloader
    else:
        return None, None

class RetrainFlowerClient(fl.client.NumPyClient):
    def __init__(self, cid: int, num_partitions: int):
        self.cid = cid
        self.num_partitions = num_partitions
        self.model = task.load_model() # Load Base Model (Fresh)
        self.model.to(task.DEVICE)
        
        self.optimizer = bnb_optim.AdamW8bit(self.model.parameters(), lr=2e-4)
        
        self.trainloader, self.testloader = load_data_retrain(
            partition_id=self.cid,
            num_clients=self.num_partitions
        )

    def get_parameters(self, config):
        return task.get_weights(self.model)

    def fit(self, parameters, config):
        # Nếu không có dữ liệu (đã bị lọc hết), trả về weight cũ và không train
        if self.trainloader is None:
             print(f"[Client {self.cid}] Skipping fit (No data).")
             return task.get_weights(self.model), 0, {}

        try:
            gc.collect(); torch.cuda.empty_cache()
            local_epochs = int(config["local-epochs"])      
            task.set_weights(self.model, parameters)
            
            task.train(self.model, self.trainloader, epochs=local_epochs, optimizer=self.optimizer)
            
            gc.collect(); torch.cuda.empty_cache()
            return task.get_weights(self.model), len(self.trainloader.dataset), {}

        except Exception as e:
            print(f"Client {self.cid} Error: {e}")
            os._exit(1)

    def evaluate(self, parameters, config):
        if self.testloader is None: return 0.0, 0, {}
        try:
            task.set_weights(self.model, parameters)
            loss, metrics = task.test(self.model, self.testloader)
            return float(loss), len(self.testloader.dataset), metrics
        except Exception: return 0.0, 0, {}

def client_fn_retrain(cid: str) -> fl.client.Client:
    return RetrainFlowerClient(int(cid), NUM_CLIENTS).to_client()

def fit_config(server_round: int):
    return {"local-epochs": str(LOCAL_EPOCHS)}

class SaveableFedAvg(FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.final_parameters: Optional[Parameters] = None

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is not None:
            self.final_parameters = aggregated_parameters
        return aggregated_parameters, aggregated_metrics

def main():
    # Thêm vào đầu hàm main() của cả 2 file
    print("TARGET UNLEARN IDs:", task.get_target_unlearn_ids())
    torch.cuda.empty_cache()
    # Load Base Model (Qwen Fresh) 
    initial_model = task.load_model(on_cpu=True)
    initial_parameters = ndarrays_to_parameters(task.get_weights(initial_model))
    del initial_model 
    
    strategy = SaveableFedAvg(
        fraction_fit=FRACTION_FIT,
        min_fit_clients=CLIENTS_PER_ROUND, 
        min_available_clients=NUM_CLIENTS,
        initial_parameters=initial_parameters,
        on_fit_config_fn=fit_config,
    )

    fl.simulation.start_simulation(
        client_fn=client_fn_retrain,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.5}, 
        ray_init_args={"num_gpus": 2}
    )

    if strategy.final_parameters:
        weights = parameters_to_ndarrays(strategy.final_parameters)
        
        output_file = "retrained_model_weights3B.npz"
        np.savez(output_file, *weights)
        print(f"\n[SUCCESS] Saved Retrained Model to '{output_file}'")

if __name__ == "__main__":
    main()
