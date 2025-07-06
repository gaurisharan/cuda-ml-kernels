import subprocess
import time
import torch
import sys
import os
import re

# ========== CONFIG ==========

exe_suffix = ".exe" if sys.platform == "win32" else ""
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

KERNELS = {
    "matrix_multiply": {
        "path": os.path.normpath(os.path.join(BASE_DIR, "../kernels/matrix_multiply" + exe_suffix)),
        "pytorch_fn": lambda: torch.matmul(
            torch.rand(1024, 1024, device='cuda'),
            torch.rand(1024, 1024, device='cuda')
        )
    },
    "vector_add": {
        "path": os.path.normpath(os.path.join(BASE_DIR, "../kernels/vector_add" + exe_suffix)),
        "pytorch_fn": lambda: torch.add(
            torch.rand(1000000, device='cuda'),
            torch.rand(1000000, device='cuda')
        )
    },
    "relu": {
        "path": os.path.normpath(os.path.join(BASE_DIR, "../kernels/relu" + exe_suffix)),
        "pytorch_fn": lambda: torch.relu(
            torch.randn(1000000, device='cuda')
        )
    },
    "dot_product": {
        "path": os.path.normpath(os.path.join(BASE_DIR, "../kernels/dot_product" + exe_suffix)),
        "pytorch_fn": lambda: torch.dot(
            torch.rand(1000000, device='cuda'),
            torch.rand(1000000, device='cuda')
        )
    },
    "intro": {
        "path": os.path.normpath(os.path.join(BASE_DIR, "../kernels/intro" + exe_suffix)),
        "pytorch_fn": lambda: torch.matmul(
            torch.arange(1, 17, dtype=torch.float32, device='cuda').reshape(4,4),
            (torch.arange(1, 17, dtype=torch.float32, device='cuda') * 2).reshape(4,4)
        )
    },
}

# ========== FUNCTIONS ==========

def benchmark_pytorch(fn, runs=10):
    torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        start = time.time()
        fn()
        torch.cuda.synchronize()
        end = time.time()
        times.append((end - start) * 1000)
    avg_time = sum(times) / len(times)
    return avg_time

def run_cuda_kernel(path):
    try:
        result = subprocess.run(path, capture_output=True, text=True, check=True)
        output = result.stdout
        print("Custom CUDA kernel output:")
        print(output)
        match = re.search(r'kernel execution time: ([\d.]+) ms', output)
        kernel_time = float(match.group(1)) if match else None
        return kernel_time
    except subprocess.CalledProcessError as e:
        print(f"Error running CUDA kernel at {path}")
        print(e)
        return None

# ========== MAIN ==========

if __name__ == "__main__":
    for name, cfg in KERNELS.items():
        print(f"\n=== Benchmarking: {name} ===")
        
        # PyTorch benchmark
        if cfg["pytorch_fn"]:
            torch_time = benchmark_pytorch(cfg["pytorch_fn"])
            print(f"--- PyTorch ---")
            print(f"PyTorch {name} time: {torch_time:.4f} ms")
        else:
            torch_time = None
            print("No PyTorch equivalent defined.")
        
        # Custom CUDA kernel benchmark
        print(f"--- Custom CUDA Kernel ---")
        cuda_time = run_cuda_kernel(cfg["path"])
        if cuda_time:
            if torch_time:
                speedup = torch_time / cuda_time
                print(f"Speedup: {speedup:.2f}x")
        else:
            print("CUDA kernel time not captured.")

# This script benchmarks both PyTorch and custom CUDA kernels for various operations.
# It runs each operation multiple times to get an average execution time and prints the results.
# The script also calculates the speedup of the custom CUDA kernel compared to the PyTorch implementation.

# =================== OBTAINED OUTPUT ====================
# === Benchmarking: matrix_multiply ===
# --- PyTorch ---
# PyTorch matrix_multiply time: 14.2845 ms
# --- Custom CUDA Kernel ---
# Custom CUDA kernel output:
# Matrix multiplication kernel execution time: 6.94618 ms
# Output[0]: 256.023

# Speedup: 2.06x

# === Benchmarking: vector_add ===
# --- PyTorch ---
# PyTorch vector_add time: 2.3942 ms
# --- Custom CUDA Kernel ---
# Custom CUDA kernel output:
# 0 3 6 9 12 15 18 21 24 27 
# vector_add kernel execution time: 1.35475 ms

# Speedup: 1.77x

# === Benchmarking: relu ===
# --- PyTorch ---
# PyTorch relu time: 2.9719 ms
# --- Custom CUDA Kernel ---
# Custom CUDA kernel output:
# 0 0 0 0.5 1 0 3 0 5 0 
# relu kernel execution time: 0.555008 ms

# Speedup: 5.35x

# === Benchmarking: dot_product ===
# --- PyTorch ---
# PyTorch dot_product time: 0.0000 ms
# --- Custom CUDA Kernel ---
# Custom CUDA kernel output:
# Dot Product: 2048
# dot_product kernel execution time: 1.32608 ms


# === Benchmarking: intro ===
# --- PyTorch ---
# PyTorch intro time: 2.2467 ms
# --- Custom CUDA Kernel ---
# Custom CUDA kernel output:
# intro kernel execution time: 1.07622 ms
# Result matrix:
# 180 200 220 240
# 404 456 508 560
# 628 712 796 880
# 852 968 1084 1200

# Speedup: 2.09x
