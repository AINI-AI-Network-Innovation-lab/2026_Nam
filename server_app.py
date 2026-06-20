# server_app.py

import flwr as fl
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg

from rkdl import task

def server_fn(context: Context):
    """Hàm khởi tạo server cho Flower."""
    
    # Đọc cấu hình từ run_config
    num_rounds = int(context.run_config.get("num-server-rounds", 3))
    num_clients = int(context.run_config.get("num-clients", 2))
    fraction_fit = float(context.run_config.get("fraction-fit", 1.0))
    
    print(f"[Server] Đang khởi tạo cho {num_rounds} vòng và đợi ít nhất {int(num_clients * fraction_fit)} clients.")

    # Khởi tạo tham số ban đầu của mô hình
    initial_model = task.load_model()
    initial_parameters = ndarrays_to_parameters(task.get_weights(initial_model))

    # Định nghĩa chiến lược (strategy)
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        min_fit_clients=int(num_clients * fraction_fit),
        min_available_clients=num_clients,
        initial_parameters=initial_parameters,
    )
    
    # Trả về các thành phần của server app
    return fl.server.ServerAppComponents(
        strategy=strategy,
        config=ServerConfig(num_rounds=num_rounds)
    )

# Tạo ServerApp
app = ServerApp(server_fn=server_fn)
