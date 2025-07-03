#!/usr/bin/env python3
"""ベンチマークの最小実行テスト"""

import numpy as np
import time
from gpu_can_decoder_optimized import OptimizedGPUCANDecoder
from gpu_can_decoder_optimized_v3 import OptimizedGPUCANDecoderV3
from cpu_can_decoder import CPUCANDecoder

def generate_synthetic_can_data(n_messages):
    """合成CANデータの生成（OpenPilot DBCファイルに準拠）"""
    address_distribution = {
        170: 0.037,  # 4輪速度
        37: 0.037,   # ステアリング
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

print("=== ベンチマーク最小実行テスト ===\n")

# デコーダー初期化
print("デコーダーを初期化中...")
optimized_gpu_decoder = OptimizedGPUCANDecoder(batch_size=500_000, chunk_size=1)
optimized_gpu_decoder_v3 = OptimizedGPUCANDecoderV3(batch_size=500_000, chunk_size=1)
cpu_decoder = CPUCANDecoder(batch_size=100_000)
print("初期化完了\n")

# ウォームアップ
print("CUDAカーネルのウォームアップ中...")
warmup_data = generate_synthetic_can_data(10_000)
for i in range(2):
    _ = optimized_gpu_decoder.decode_batch_for_benchmark(*warmup_data)
    _ = optimized_gpu_decoder_v3.decode_batch_for_benchmark(*warmup_data)
print("ウォームアップ完了！\n")

# テスト実行
n_messages = 1_000_000
print(f"--- {n_messages:,} メッセージの処理 ---")

timestamps, addresses, data_bytes = generate_synthetic_can_data(n_messages)
data_size_mb = (timestamps.nbytes + addresses.nbytes + data_bytes.nbytes) / (1024**2)
target_mask = (addresses == 170) | (addresses == 37)
target_size_mb = (timestamps[target_mask].nbytes + addresses[target_mask].nbytes + data_bytes[target_mask].nbytes) / (1024**2)

print(f"全データサイズ: {data_size_mb:.1f} MB（内抽出対象データサイズ: {target_size_mb:.1f} MB）\n")

# GPU処理（最適化）
print("GPU処理（最適化）:")
gpu_start = time.time()
gpu_results = optimized_gpu_decoder.decode_batch_for_benchmark(timestamps, addresses, data_bytes)
gpu_time = time.time() - gpu_start
print(f"処理時間: {gpu_time:.4f}秒 ({n_messages / gpu_time / 1e6:.1f} Mmsg/s)\n")

# GPU処理（V3）
print("GPU処理（V3）:")
gpu_v3_start = time.time()
gpu_v3_results = optimized_gpu_decoder_v3.decode_batch_for_benchmark(timestamps, addresses, data_bytes)
gpu_v3_time = time.time() - gpu_v3_start
print(f"処理時間: {gpu_v3_time:.4f}秒 ({n_messages / gpu_v3_time / 1e6:.1f} Mmsg/s)\n")

# CPU処理
print("CPU処理:")
cpu_start = time.time()
cpu_results = cpu_decoder.decode_batch(timestamps, addresses, data_bytes, debug_timing=True)
cpu_time = time.time() - cpu_start

if '_timing' in cpu_results:
    timing = cpu_results['_timing']
    data_extract_time = timing.get('index_search', 0) + timing.get('array_allocation', 0)
    physical_convert_time = (timing.get('decode_loop', 0) + 
                           timing.get('wheel_df_creation', 0) + 
                           timing.get('wheel_sort', 0))
    
    print(f"  === CPU処理の詳細 ===")
    print(f"  データ抽出: {data_extract_time:.4f}秒")
    print(f"  物理値変換: {physical_convert_time:.4f}秒")
    print(f"  総処理時間: {data_extract_time + physical_convert_time:.4f}秒")

print(f"\nCPU処理時間: {cpu_time:.4f}秒 ({n_messages / cpu_time / 1e6:.1f} Mmsg/s)")
print(f"\n高速化率（最適化）: {cpu_time / gpu_time:.1f}x")
print(f"高速化率（V3）: {cpu_time / gpu_v3_time:.1f}x")