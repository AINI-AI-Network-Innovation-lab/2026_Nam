import flwr as fl
import numpy as np
from .client_app import client_fn
from flwr.server.strategy import FedAvg
from flwr.common import (
    parameters_to_ndarrays, 
    ndarrays_to_parameters, 
    Parameters,
    FitRes, 
    Scalar, 
    FitIns,
    EvaluateIns
)
from . import task 
import torch
from typing import Dict, List, Optional, Tuple, Union
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
import time 

NUM_CLIENTS = 20
NUM_ROUNDS = 100
LOCAL_EPOCHS = 1
CLIENTS_PER_ROUND = 20
FRACTION_FIT = CLIENTS_PER_ROUND / NUM_CLIENTS 

def fit_config(server_round: int):
    config = {
        "local-epochs": str(LOCAL_EPOCHS),
        "num-clients": str(NUM_CLIENTS), 
    }
    return config

class SaveableFedAvg(FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.final_parameters: Optional[Parameters] = None
        self.round_start_time: Optional[float] = None

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        print(f"\n--- [Server] Bắt đầu Round {server_round}/{NUM_ROUNDS} ---")
        self.round_start_time = time.time()
        return super().configure_fit(server_round, parameters, client_manager)

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        print(f"--- [Server] Bỏ qua (Tắt) Giai đoạn Evaluate (Round {server_round}) ---")
        return []

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        
        if self.round_start_time is not None:
            round_duration = time.time() - self.round_start_time
            print(f"--- [Server] Round {server_round} hoàn tất. Thời gian: {round_duration:.2f} giây ---")
            self.round_start_time = None
        
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        
        if aggregated_parameters is not None:
            self.final_parameters = aggregated_parameters
            
        return aggregated_parameters, aggregated_metrics

def main():
    torch.cuda.empty_cache()
    print("[Server] Đang tải mô hình ban đầu...")
    
    initial_model = task.load_model(on_cpu=True)
    
    initial_parameters = ndarrays_to_parameters(task.get_weights(initial_model))
    
    del initial_model 
    torch.cuda.empty_cache()
    print("[Server] Tải mô hình ban đầu hoàn tất.")

    strategy = SaveableFedAvg(
        fraction_fit=FRACTION_FIT,
        min_fit_clients=CLIENTS_PER_ROUND, 
        min_available_clients=NUM_CLIENTS,
        initial_parameters=initial_parameters,
        on_fit_config_fn=fit_config,
    )

    client_resources = {"num_cpus": 1, "num_gpus": 0.2}

    print(f"\nBắt đầu Giai đoạn HỌC TẬP")
    print(f"Cấu hình: {NUM_CLIENTS} clients total, {CLIENTS_PER_ROUND} clients/round, {NUM_ROUNDS} rounds, {LOCAL_EPOCHS} local epochs.")

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources=client_resources, 
        ray_init_args={"num_gpus": 2}
    )

    final_parameters = strategy.final_parameters 
    
    if final_parameters is None:
        print("Lỗi: Không thể lấy được trọng số cuối cùng từ strategy.")
        return

    weights = parameters_to_ndarrays(final_parameters)

    np.savez("trained_model_weights.npz", *weights)
    print("\nGiai đoạn HỌC TẬP hoàn tất. Trọng số đã được lưu vào 'trained_model_weights.npz'")
    print("Lịch sử loss (distributed):", history.losses_distributed)

if __name__ == "__main__":
    main()
