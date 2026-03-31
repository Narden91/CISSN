import os
import shutil
import torch
import time
from cissn.models.encoder import DisentangledStateEncoder

def _synchronize_if_needed(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)

def _can_try_torch_compile():
    if hasattr(torch, 'compile') and torch.__version__.split('.')[0].isdigit() and int(torch.__version__.split('.')[0]) >= 2:
        if os.name == 'nt' and shutil.which('cl') is None:
            return False, "Skipping torch.compile: cl.exe was not found on PATH. Install Visual Studio Build Tools to benchmark compiled mode on Windows."
        return True, ""
    return False, "Skipping torch.compile (Requires PyTorch 2.0+)"

def measure_performance(model, x, name="Model", return_all_states=False, warmup_iters=10, measure_iters=100):
    print(f"Warming up {name}...")
    with torch.inference_mode():
        for _ in range(warmup_iters):
            _ = model(x, return_all_states=return_all_states)

        _synchronize_if_needed(x.device)
        start_time = time.perf_counter()
        for _ in range(measure_iters):
            _ = model(x, return_all_states=return_all_states)
        _synchronize_if_needed(x.device)
    end_time = time.perf_counter()
    avg_time = (end_time - start_time) / measure_iters
    mode = "all states" if return_all_states else "final state"
    
    print(
        f"[{name} | {mode}] Avg Time: {avg_time*1000:.2f} ms | "
        f"Throughput: {x.size(0) / avg_time:.1f} samples/sec"
    )

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
    measure_performance(model, x, "Eager Mode", return_all_states=False)
    measure_performance(model, x, "Eager Mode", return_all_states=True)
    
    # 2. Compile
    can_compile, reason = _can_try_torch_compile()
    if can_compile:
        print("Compiling model with torch.compile...")
        try:
            compiled_model = torch.compile(model, mode="reduce-overhead")
            measure_performance(compiled_model, x, "torch.compile", return_all_states=False)
            measure_performance(compiled_model, x, "torch.compile", return_all_states=True)
        except Exception as e:
            print(f"Compilation failed: {e}")
    else:
        print(reason)

if __name__ == "__main__":
    benchmark_encoder()
