import numpy as np
import time
from gpu_can_decoder_optimized import OptimizedGPUCANDecoder
from gpu_can_decoder_optimized_v2 import OptimizedGPUCANDecoderV2
from cpu_can_decoder import CPUCANDecoder

def generate_synthetic_can_data(n_messages):
    """合成CANデータの生成（OpenPilot DBCファイルに準拠）"""
    # リアルなCANデータ分布を模倣
    address_distribution = {
        170: 0.037,  # 4輪速度
        37: 0.037,   # ステアリング
        36: 0.037,
        740: 0.044,
        608: 0.022,
        180: 0.018,
    }
    
    # アドレスを生成
    addresses = []
    for addr, prob in address_distribution.items():
        count = int(n_messages * prob)
        addresses.extend([addr] * count)
    
    # 残りはランダムなアドレス
    remaining = n_messages - len(addresses)
    other_addresses = np.random.choice([452, 466, 467, 705, 321, 562], remaining)
    addresses.extend(other_addresses)
    
    # シャッフル
    np.random.shuffle(addresses)
    addresses = np.array(addresses[:n_messages], dtype=np.int64)
    
    # タイムスタンプ（実データと同じ範囲）
    timestamps = np.linspace(46408.0, 46468.0, n_messages)
    
    # データバイト
    data_bytes = np.zeros((n_messages, 8), dtype=np.uint8)
    
    for i in range(n_messages):
        if addresses[i] == 170:  # 4輪速度
            # OpenPilot DBC: (0.01,-67.67) "kph" for Toyota RAV4
            for j in range(4):
                speed_kmh = np.random.uniform(55, 65)  # 55-65 km/h
                raw_value = int((speed_kmh + 67.67) / 0.01)
                data_bytes[i, j*2] = (raw_value >> 8) & 0xFF
                data_bytes[i, j*2 + 1] = raw_value & 0xFF
        elif addresses[i] == 37:  # ステアリング
            # 固定値パターン（実データと同じ）
            data_bytes[i] = [0x00, 0x00, 0x10, 0x00, 0xC0, 0x00, 0x00, 0xFD]
        else:
            # その他はランダム
            data_bytes[i] = np.random.randint(0, 256, 8, dtype=np.uint8)
    
    return timestamps, addresses, data_bytes

# テスト実行
print("=== GPU性能テスト開始 ===")

# 複数のデータサイズでテスト
test_sizes = [1_000_000, 5_000_000, 10_000_000]

for n_messages in test_sizes:
    print(f"\n{'='*60}")
    print(f"テストデータ: {n_messages:,} メッセージ")

# データ生成
timestamps, addresses, data_bytes = generate_synthetic_can_data(n_messages)
data_size_mb = (timestamps.nbytes + addresses.nbytes + data_bytes.nbytes) / (1024**2)
print(f"データサイズ: {data_size_mb:.1f} MB")

# GPU処理（最適化版V1）
print("\n--- GPU処理（CUDAカーネル版V1）---")
gpu_decoder = OptimizedGPUCANDecoder(batch_size=500_000, chunk_size=1)
gpu_start = time.time()
gpu_results = gpu_decoder.decode_batch_for_benchmark(timestamps, addresses, data_bytes)
gpu_time = time.time() - gpu_start
gpu_throughput = n_messages / gpu_time / 1e6
print(f"GPU V1処理時間: {gpu_time:.4f}秒")
print(f"GPU V1スループット: {gpu_throughput:.1f} Mmsg/s")

# GPU処理（高速化版V2）
print("\n--- GPU処理（高速化版V2）---")
gpu_decoder_v2 = OptimizedGPUCANDecoderV2(batch_size=500_000, chunk_size=1)
gpu_start_v2 = time.time()
gpu_results_v2 = gpu_decoder_v2.decode_batch_for_benchmark(timestamps, addresses, data_bytes)
gpu_time_v2 = time.time() - gpu_start_v2
gpu_throughput_v2 = n_messages / gpu_time_v2 / 1e6
print(f"GPU V2処理時間: {gpu_time_v2:.4f}秒")
print(f"GPU V2スループット: {gpu_throughput_v2:.1f} Mmsg/s")

# CPU処理
print("\n--- CPU処理 ---")
cpu_decoder = CPUCANDecoder(batch_size=100_000)
cpu_start = time.time()
cpu_results = cpu_decoder.decode_batch(timestamps, addresses, data_bytes)
cpu_time = time.time() - cpu_start
cpu_throughput = n_messages / cpu_time / 1e6
print(f"CPU処理時間: {cpu_time:.4f}秒")
print(f"CPUスループット: {cpu_throughput:.1f} Mmsg/s")

print(f"\n高速化率（V1）: {cpu_time / gpu_time:.1f}x")
print(f"高速化率（V2）: {cpu_time / gpu_time_v2:.1f}x")
print(f"\n目標: 100+ Mmsg/s")
print(f"V1: {gpu_throughput:.1f} Mmsg/s")
print(f"V2: {gpu_throughput_v2:.1f} Mmsg/s")