import flwr as fl
from . import task 
import torch
import bitsandbytes.optim as bnb_optim
import gc
import os
import sys


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid: int, num_partitions: int):
        self.cid = cid
        self.num_partitions = num_partitions
        self.model = task.load_model()
        self.model.to(task.DEVICE)
        
        self.optimizer = bnb_optim.AdamW8bit(self.model.parameters(), lr=2e-4)
        
        self.trainloader, self.testloader = task.load_data(
            partition_id=self.cid,
            num_clients=self.num_partitions
        )

    def get_parameters(self, config):
        return task.get_weights(self.model)

    def fit(self, parameters, config):
        try:
        
            gc.collect()
            torch.cuda.empty_cache()
            
            
            local_epochs = int(config["local-epochs"])      
            task.set_weights(self.model, parameters)
            task.train(self.model, self.trainloader, epochs=local_epochs, optimizer=self.optimizer)
            
            
            gc.collect()
            torch.cuda.empty_cache()
            
            return task.get_weights(self.model), len(self.trainloader.dataset), {}

        except torch.OutOfMemoryError as e:
            print(f"\n[FATAL, Client {self.cid}] GẶP LỖI OOM KHI TRAIN. ÉP SẬP TIẾN TRÌNH ĐỂ GIẢI PHÓNG VRAM.")
            print(f"Lỗi: {e}")
            # Dùng os._exit(1) để ép tiến trình (actor) tự sát ngay lập tức
            # OS sẽ thu hồi VRAM, Ray sẽ khởi động actor mới cho client sau.
            os._exit(1)
        except Exception as e:
            print(f"\n[FATAL, Client {self.cid}] Gặp lỗi không xác định khi train: {e}")
            os._exit(1)


    def evaluate(self, parameters, config):
        try:
            task.set_weights(self.model, parameters)
            loss, metrics = task.test(self.model, self.testloader)
            return float(loss), len(self.testloader.dataset), metrics
        
        except torch.OutOfMemoryError as e:
            print(f"\n[FATAL, Client {self.cid}] GẶP LỖI OOM KHI EVALUATE. Ép sập tiến trình.")
            print(f"Lỗi: {e}")
            os._exit(1)
        except Exception as e:
            print(f"\n[FATAL, Client {self.cid}] Gặp lỗi không xác định khi evaluate: {e}")
            os._exit(1)


def client_fn(cid: str) -> fl.client.Client:

    partition_id = int(cid)
    num_clients = 20    
    print(f"[Client {partition_id}] Bắt đầu khởi tạo cho pool {num_clients} clients.")

    try:
        client = FlowerClient(
            cid=partition_id,
            num_partitions=num_clients
        )
        torch.cuda.empty_cache()
        return client.to_client()

    except torch.OutOfMemoryError as e:
        # Lỗi OOM ngay khi init (do lỗi tiếp diễn)
        print(f"\n[FATAL, Client {partition_id}] GẶP LỖI OOM KHI KHỞI TẠO (INIT).")
        print(f"Lỗi: {e}")
        print("Tiến trình này có thể đang chạy trên một GPU bị rò rỉ VRAM từ client trước.")
        print("Ép sập tiến trình để giải phóng VRAM.")
        os._exit(1)
        
    except Exception as e:
        print(f"\n[FATAL, Client {partition_id}] Gặp lỗi không xác định khi KHỞI TẠO: {e}")
        os._exit(1) 
