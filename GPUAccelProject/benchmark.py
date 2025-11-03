"""
GPU vs CPU Benchmarking Module
Compares matrix operations on CPU (NumPy) vs GPU (CuPy/PyTorch)
"""

import time
import numpy as np
import torch

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    print("CuPy not available, using PyTorch for GPU operations")


def benchmark_cpu_matmul(size=2048, iterations=3):
    """Benchmark CPU matrix multiplication using NumPy"""
    times = []
    
    # Generate random matrices
    A = np.random.randn(size, size).astype(np.float32)
    B = np.random.randn(size, size).astype(np.float32)
    
    # Warmup
    _ = np.matmul(A, B)
    
    # Benchmark
    for _ in range(iterations):
        start = time.perf_counter()
        C = np.matmul(A, B)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    avg_time = np.mean(times)
    return avg_time, C.shape


def benchmark_gpu_cupy(size=2048, iterations=3):
    """Benchmark GPU matrix multiplication using CuPy"""
    if not CUPY_AVAILABLE:
        return None, None
    
    times = []
    
    # Generate random matrices on GPU
    A = cp.random.randn(size, size, dtype=cp.float32)
    B = cp.random.randn(size, size, dtype=cp.float32)
    
    # Warmup
    _ = cp.matmul(A, B)
    cp.cuda.Stream.null.synchronize()
    
    # Benchmark
    for _ in range(iterations):
        start = time.perf_counter()
        C = cp.matmul(A, B)
        cp.cuda.Stream.null.synchronize()  # Wait for GPU to finish
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    avg_time = np.mean(times)
    return avg_time, C.shape


def benchmark_gpu_pytorch(size=2048, iterations=3):
    """Benchmark GPU matrix multiplication using PyTorch"""
    if not torch.cuda.is_available():
        return None, None
    
    times = []
    device = torch.device('cuda')
    
    # Generate random matrices on GPU
    A = torch.randn(size, size, dtype=torch.float32, device=device)
    B = torch.randn(size, size, dtype=torch.float32, device=device)
    
    # Warmup
    _ = torch.matmul(A, B)
    torch.cuda.synchronize()
    
    # Benchmark
    for _ in range(iterations):
        start = time.perf_counter()
        C = torch.matmul(A, B)
        torch.cuda.synchronize()  # Wait for GPU to finish
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    avg_time = np.mean(times)
    return avg_time, tuple(C.shape)


def run_benchmark(size=2048, iterations=3):
    """Run complete benchmark suite and return results"""
    print(f"\n{'='*60}")
    print(f"Running Benchmark: {size}x{size} matrix multiplication")
    print(f"Iterations: {iterations}")
    print(f"{'='*60}\n")
    
    # CPU Benchmark
    print("CPU (NumPy)...")
    cpu_time, cpu_shape = benchmark_cpu_matmul(size, iterations)
    print(f"   Time: {cpu_time:.2f} ms")
    
    # GPU Benchmark (try CuPy first, then PyTorch)
    gpu_time = None
    gpu_backend = None
    
    if CUPY_AVAILABLE:
        print("GPU (CuPy)...")
        gpu_time, gpu_shape = benchmark_gpu_cupy(size, iterations)
        if gpu_time:
            gpu_backend = "CuPy"
            print(f"   Time: {gpu_time:.2f} ms")
    
    if gpu_time is None and torch.cuda.is_available():
        print("GPU (PyTorch)...")
        gpu_time, gpu_shape = benchmark_gpu_pytorch(size, iterations)
        if gpu_time:
            gpu_backend = "PyTorch"
            print(f"   Time: {gpu_time:.2f} ms")
    
    # Calculate speedup
    if gpu_time and gpu_time > 0:
        speedup = cpu_time / gpu_time
    else:
        speedup = None
        print("GPU not available or benchmark failed")
    
    results = {
        "size": size,
        "iterations": iterations,
        "cpu_time_ms": round(cpu_time, 2),
        "gpu_time_ms": round(gpu_time, 2) if gpu_time else None,
        "speedup": round(speedup, 2) if speedup else None,
        "gpu_backend": gpu_backend
    }
    
    print(f"\n{'='*60}")
    if speedup:
        print(f" Speedup: {speedup:.2f}x (GPU is {speedup:.2f}x faster)")
    print(f"{'='*60}\n")
    
    return results


if __name__ == "__main__":
    # Test different matrix sizes
    sizes = [1024, 2048, 4096]
    
    for size in sizes:
        results = run_benchmark(size=size, iterations=3)
        print(results)