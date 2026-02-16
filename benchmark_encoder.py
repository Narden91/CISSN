import torch
import time
from cissn.models.encoder import DisentangledStateEncoder

def measure_performance(model, x, name="Model"):
    # Warmup
    print(f"Warming up {name}...")
    for _ in range(10):
        _ = model(x)
        
    start_time = time.time()
    n_iters = 100
    for _ in range(n_iters):
        _ = model(x)
        
    if x.device.type == 'cuda':
        torch.cuda.synchronize()
        
    end_time = time.time()
    avg_time = (end_time - start_time) / n_iters
    
    print(f"[{name}] Avg Time: {avg_time*1000:.2f} ms | Throughput: {x.size(0) / avg_time:.1f} samples/sec")

def benchmark_encoder():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Benchmarking on {device}")
    
    batch_size = 64
    seq_len = 96
    input_dim = 1
    
    x = torch.randn(batch_size, seq_len, input_dim).to(device)
    
    # 1. Baseline
    model = DisentangledStateEncoder(input_dim=input_dim).to(device)
    model.eval()
    measure_performance(model, x, "Eager Mode")
    
    # 2. Compile
    if int(torch.__version__.split('.')[0]) >= 2:
        print("Compiling model with torch.compile...")
        try:
            compiled_model = torch.compile(model)
            measure_performance(compiled_model, x, "torch.compile")
        except Exception as e:
            print(f"Compilation failed: {e}")
    else:
        print("Skipping torch.compile (Requires PyTorch 2.0+)")

if __name__ == "__main__":
    benchmark_encoder()
