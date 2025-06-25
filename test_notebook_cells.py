#!/usr/bin/env python
"""ノートブックのセルを個別にテスト"""

print("=== Testing notebook cells ===")

# Cell 1: Basic imports
print("\nCell 1: Basic imports...")
try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import time
    from pathlib import Path
    import sys
    sys.path.append('..')
    from gpu_can_decoder import GPUCANDecoder
    from cpu_can_decoder import CPUCANDecoder
    print("✓ Success")
except Exception as e:
    print(f"✗ Error: {e}")

# Cell 2: GPU monitoring tools
print("\nCell 2: GPU monitoring tools...")
try:
    import pynvml
    import cupy as cp
    import rmm
    from rmm.statistics import ProfilerRecords, statistics
    import threading
    from collections import deque
    import gc
    import psutil
    
    pynvml.nvmlInit()
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    
    gpu_name_raw = pynvml.nvmlDeviceGetName(gpu_handle)
    gpu_name = gpu_name_raw.decode('utf-8') if isinstance(gpu_name_raw, bytes) else gpu_name_raw
    gpu_mem_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
    print(f"GPU: {gpu_name}")
    print(f"GPU Memory: {gpu_mem_info.total / (1024**3):.1f} GB")
    
    rmm.statistics.enable_statistics()
    rmm.reinitialize(
        managed_memory=False,
        pool_allocator=True,
        initial_pool_size=2<<30,
        maximum_pool_size=22<<30,
        logging=True
    )
    print("✓ Success")
except Exception as e:
    print(f"✗ Error: {e}")

# Cell 3: GPUMonitor class
print("\nCell 3: GPUMonitor class...")
try:
    class GPUMonitor:
        def __init__(self, interval=0.1):
            self.interval = interval
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.monitoring = False
            self.timestamps = deque(maxlen=10000)
            self.gpu_utils = deque(maxlen=10000)
            self.mem_utils = deque(maxlen=10000)
            self.mem_used = deque(maxlen=10000)
            self.thread = None
        
        def _monitor_loop(self):
            start_time = time.time()
            while self.monitoring:
                current_time = time.time() - start_time
                util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                self.gpu_utils.append(util.gpu)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                self.mem_utils.append(100 * mem_info.used / mem_info.total)
                self.mem_used.append(mem_info.used / (1024**3))
                self.timestamps.append(current_time)
                time.sleep(self.interval)
        
        def start(self):
            self.monitoring = True
            self.thread = threading.Thread(target=self._monitor_loop)
            self.thread.start()
        
        def stop(self):
            self.monitoring = False
            if self.thread:
                self.thread.join()
        
        def get_data(self):
            return {
                'timestamps': list(self.timestamps),
                'gpu_utils': list(self.gpu_utils),
                'mem_utils': list(self.mem_utils),
                'mem_used_gb': list(self.mem_used)
            }
        
        def plot(self, title="GPU Monitoring Results"):
            pass  # Skip plotting in test

    def print_rmm_statistics():
        try:
            stats = rmm.statistics.get_statistics()
            if stats:
                print("\nRMM Memory Statistics:")
                print(f"  Current allocated: {stats.current_bytes / (1024**3):.2f} GB")
                print(f"  Peak allocated: {stats.peak_bytes / (1024**3):.2f} GB")
                print(f"  Total allocations: {stats.n_allocations}")
                print(f"  Total deallocations: {stats.n_deallocations}")
            else:
                print("\nRMM Memory Statistics: No statistics available")
        except Exception as e:
            print(f"\nRMM Memory Statistics: Error - {e}")

    def print_cupy_memory_info():
        mempool = cp.get_default_memory_pool()
        print("\nCuPy Memory Pool:")
        print(f"  Used: {mempool.used_bytes() / (1024**3):.2f} GB")
        print(f"  Total: {mempool.total_bytes() / (1024**3):.2f} GB")

    monitor = GPUMonitor(interval=0.05)
    print("✓ Success - GPU Monitor initialized")
except Exception as e:
    print(f"✗ Error: {e}")

# Cell 4: generate_synthetic_can_data function
print("\nCell 4: generate_synthetic_can_data function...")
try:
    def generate_synthetic_can_data(n_messages):
        address_distribution = {
            170: 0.037,
            37: 0.037,
            36: 0.037,
            740: 0.044,
            608: 0.022,
            180: 0.018,
        }
        
        addresses = []
        for addr, prob in address_distribution.items():
            count = int(n_messages * prob)
            addresses.extend([addr] * count)
        
        remaining = n_messages - len(addresses)
        other_addresses = np.random.choice([452, 466, 467, 705, 321, 562], remaining)
        addresses.extend(other_addresses)
        
        np.random.shuffle(addresses)
        addresses = np.array(addresses[:n_messages], dtype=np.int64)
        
        timestamps = np.linspace(46408.0, 46468.0, n_messages)
        
        data_bytes = np.zeros((n_messages, 8), dtype=np.uint8)
        
        for i in range(n_messages):
            if addresses[i] == 170:
                for j in range(4):
                    speed_kmh = np.random.uniform(55, 65)
                    raw_value = int((speed_kmh + 67.67) / 0.01)
                    data_bytes[i, j*2] = (raw_value >> 8) & 0xFF
                    data_bytes[i, j*2 + 1] = raw_value & 0xFF
            elif addresses[i] == 37:
                data_bytes[i] = [0x00, 0x00, 0x10, 0x00, 0xC0, 0x00, 0x00, 0xFD]
            else:
                data_bytes[i] = np.random.randint(0, 256, 8, dtype=np.uint8)
        
        return timestamps, addresses, data_bytes
    
    # Test the function
    test_t, test_a, test_d = generate_synthetic_can_data(100)
    print(f"✓ Success - Generated {len(test_t)} messages")
except Exception as e:
    print(f"✗ Error: {e}")

# Cell 5: Test sizes
print("\nCell 5: Test sizes...")
try:
    test_sizes = [
        10_000,
        50_000,
        100_000,
        500_000,
        1_000_000,
        5_000_000,
        10_000_000,
    ]
    print(f"✓ Success - Test sizes: {test_sizes}")
except Exception as e:
    print(f"✗ Error: {e}")

# Cell 6: Sample data generation test
print("\nCell 6: Sample data generation...")
try:
    n_sample = 5
    sample_timestamps, sample_addresses, sample_data_bytes = generate_synthetic_can_data(n_sample)
    print(f"✓ Success - Generated sample data")
    print(f"  Timestamps shape: {sample_timestamps.shape}")
    print(f"  Addresses shape: {sample_addresses.shape}")
    print(f"  Data bytes shape: {sample_data_bytes.shape}")
except Exception as e:
    print(f"✗ Error: {e}")

# Cell 7: Initial memory state
print("\nCell 7: Initial memory state...")
try:
    print_rmm_statistics()
    print_cupy_memory_info()
    
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
    print(f"\nGPU Memory (NVML):")
    print(f"  Total: {mem_info.total / (1024**3):.1f} GB")
    print(f"  Used: {mem_info.used / (1024**3):.2f} GB")
    print(f"  Free: {mem_info.free / (1024**3):.2f} GB")
    
    print(f"\nSystem RAM:")
    print(f"  Available: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    print(f"  Total: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print("✓ Success")
except Exception as e:
    print(f"✗ Error: {e}")

# Cell 8: Create decoders
print("\nCell 8: Create decoders...")
try:
    gpu_decoder = GPUCANDecoder(batch_size=500_000)
    cpu_decoder = CPUCANDecoder(batch_size=100_000)
    print("✓ Success - Decoders created")
except Exception as e:
    print(f"✗ Error: {e}")

# Cell 9: Small benchmark test
print("\nCell 9: Small benchmark test...")
try:
    # Test with just 1000 messages
    test_n = 1000
    timestamps, addresses, data_bytes = generate_synthetic_can_data(test_n)
    
    # GPU処理
    gpu_start = time.time()
    gpu_results = gpu_decoder.decode_batch(timestamps, addresses, data_bytes)
    cp.cuda.Stream.null.synchronize()
    gpu_time = time.time() - gpu_start
    
    # CPU処理
    cpu_start = time.time()
    cpu_results = cpu_decoder.decode_batch(timestamps, addresses, data_bytes)
    cpu_time = time.time() - cpu_start
    
    print(f"✓ Success - Benchmark completed")
    print(f"  GPU time: {gpu_time:.4f}s")
    print(f"  CPU time: {cpu_time:.4f}s")
    print(f"  Speedup: {cpu_time/gpu_time:.2f}x")
except Exception as e:
    print(f"✗ Error: {e}")

print("\n=== Test completed ===")