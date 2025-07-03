import numpy as np
import time
from gpu_can_decoder_optimized import OptimizedGPUCANDecoder
from gpu_can_decoder_optimized_v3 import OptimizedGPUCANDecoderV3
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
print("=== GPU性能テスト（V3実装）===")

# ウォームアップ
print("\n--- ウォームアップ中 ---")
warmup_data = generate_synthetic_can_data(10_000)
gpu_decoder = OptimizedGPUCANDecoder(batch_size=500_000, chunk_size=1)
gpu_decoder_v3 = OptimizedGPUCANDecoderV3(batch_size=500_000, chunk_size=1)
cpu_decoder = CPUCANDecoder(batch_size=100_000)

# ウォームアップ実行
for i in range(2):
    print(f"ウォームアップ {i+1}/2...")
    _ = gpu_decoder.decode_batch_for_benchmark(*warmup_data)
    _ = gpu_decoder_v3.decode_batch_for_benchmark(*warmup_data)

print("ウォームアップ完了！\n")

# テストサイズ
n_messages = 1_000_000
print(f"テストデータ: {n_messages:,} メッセージ")

# データ生成
timestamps, addresses, data_bytes = generate_synthetic_can_data(n_messages)
data_size_mb = (timestamps.nbytes + addresses.nbytes + data_bytes.nbytes) / (1024**2)

# 抽出対象データサイズの計算
target_mask = (addresses == 170) | (addresses == 37)
target_size_mb = (timestamps[target_mask].nbytes + addresses[target_mask].nbytes + data_bytes[target_mask].nbytes) / (1024**2)

print(f"全データサイズ: {data_size_mb:.1f} MB")
print(f"抽出対象データサイズ: {target_size_mb:.1f} MB ({target_size_mb/data_size_mb*100:.1f}%)")

# マスク作成時間の計測（共通処理）
mask_start = time.time()
target_mask = (addresses == 170) | (addresses == 37)
mask_time = time.time() - mask_start
print(f"\nマスク作成時間（共通処理）: {mask_time:.4f}秒")

# GPU処理（修正版）
print("\n--- GPU処理（修正版）---")
gpu_start = time.time()
gpu_results = gpu_decoder.decode_batch_for_benchmark(timestamps, addresses, data_bytes)
gpu_time = time.time() - gpu_start
gpu_throughput = n_messages / gpu_time / 1e6
print(f"GPU処理時間: {gpu_time:.4f}秒")
print(f"GPUスループット: {gpu_throughput:.1f} Mmsg/s")

# GPU処理（V3版）
print("\n--- GPU処理（V3版）---")
gpu_v3_start = time.time()
gpu_v3_results = gpu_decoder_v3.decode_batch_for_benchmark(timestamps, addresses, data_bytes)
gpu_v3_time = time.time() - gpu_v3_start
gpu_v3_throughput = n_messages / gpu_v3_time / 1e6
print(f"GPU V3処理時間: {gpu_v3_time:.4f}秒")
print(f"GPU V3スループット: {gpu_v3_throughput:.1f} Mmsg/s")

# CPU処理
print("\n--- CPU処理 ---")
cpu_start = time.time()
cpu_results = cpu_decoder.decode_batch(timestamps, addresses, data_bytes, debug_timing=True)
cpu_time = time.time() - cpu_start

# CPU処理の詳細時間を表示
if '_timing' in cpu_results:
    timing = cpu_results['_timing']
    # データ抽出（インデックス検索と配列割り当て）
    data_extract_time = timing.get('index_search', 0) + timing.get('array_allocation', 0)
    # 物理値変換（デコードループ、DataFrame作成、ソート）
    physical_convert_time = (timing.get('decode_loop', 0) + 
                           timing.get('wheel_df_creation', 0) + 
                           timing.get('wheel_sort', 0))
    
    print(f"\n  === CPU処理の詳細 ===")
    print(f"  データ抽出: {data_extract_time:.4f}秒")
    print(f"  物理値変換: {physical_convert_time:.4f}秒")
    print(f"  総処理時間: {data_extract_time + physical_convert_time:.4f}秒")

cpu_throughput = n_messages / cpu_time / 1e6
print(f"\nCPU処理時間: {cpu_time:.4f}秒")
print(f"CPUスループット: {cpu_throughput:.1f} Mmsg/s")

print(f"\n高速化率（修正版）: {cpu_time / gpu_time:.1f}x")
print(f"高速化率（V3版）: {cpu_time / gpu_v3_time:.1f}x")
print(f"\n目標: 100+ Mmsg/s")
print(f"修正版: {gpu_throughput:.1f} Mmsg/s")
print(f"V3版: {gpu_v3_throughput:.1f} Mmsg/s")