#!/usr/bin/env python3
"""
最適化されたCANデータGPU処理ベンチマーク
24GB GPU用のRMM設定で大規模データを処理
"""

import os
import gc
import time
import numpy as np
import cupy as cp
import cudf
import rmm
import psutil
from gpu_can_decoder import GPUCANDecoder

# 最適化されたRMM設定（24GB GPU用）
print("Initializing RMM with optimized settings for 24GB GPU...")
rmm.reinitialize(
    managed_memory=True,
    pool_allocator=True,
    initial_pool_size=8<<30,   # 8GB initial pool
    maximum_pool_size=22<<30   # 22GB max pool (24GBの約92%)
)
print("✓ RMM initialized successfully")

def print_memory_status():
    """メモリ状況を表示"""
    # システムメモリ
    mem = psutil.virtual_memory()
    print(f"\nSystem Memory: {mem.available / (1024**3):.1f} GB available / {mem.total / (1024**3):.1f} GB total")
    
    # GPUメモリ
    device = cp.cuda.Device()
    mem_info = device.mem_info
    print(f"GPU Memory: {mem_info[0] / (1024**3):.1f} GB free / {mem_info[1] / (1024**3):.1f} GB total")
    
    # RMMプール
    try:
        mempool = cp.get_default_memory_pool()
        print(f"RMM Pool: {mempool.used_bytes() / (1024**3):.1f} GB used / {mempool.total_bytes() / (1024**3):.1f} GB allocated")
    except:
        pass

def generate_synthetic_can_data(n_messages):
    """合成CANデータの生成"""
    print(f"\nGenerating {n_messages:,} CAN messages...")
    
    # タイムスタンプ
    timestamps = np.linspace(46408.0, 46468.0, n_messages, dtype=np.float64)
    
    # アドレス分布（実際のCANデータに基づく）
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
    
    # 残りはランダムなアドレス
    remaining = n_messages - len(addresses)
    other_addresses = np.random.choice([452, 466, 467, 705, 321, 562], remaining)
    addresses.extend(other_addresses)
    
    # シャッフル
    np.random.shuffle(addresses)
    addresses = np.array(addresses[:n_messages], dtype=np.int64)
    
    # データバイト（リアルなCANデータをシミュレート）
    data_bytes = np.zeros((n_messages, 8), dtype=np.uint8)
    
    for i in range(n_messages):
        if addresses[i] == 170:  # 4輪速度
            for j in range(4):
                speed_kmh = np.random.uniform(55, 65)  # 55-65 km/h
                raw_value = int((speed_kmh + 67.67) / 0.01)
                data_bytes[i, j*2] = (raw_value >> 8) & 0xFF
                data_bytes[i, j*2 + 1] = raw_value & 0xFF
        elif addresses[i] == 37:  # ステアリング
            data_bytes[i] = [0x00, 0x00, 0x10, 0x00, 0xC0, 0x00, 0x00, 0xFD]
        else:
            data_bytes[i] = np.random.randint(0, 256, 8, dtype=np.uint8)
    
    data_size_mb = (timestamps.nbytes + addresses.nbytes + data_bytes.nbytes) / (1024**2)
    print(f"✓ Generated {data_size_mb:.1f} MB of data")
    
    return timestamps, addresses, data_bytes

def process_can_data(n_messages, decoder):
    """CANデータを処理"""
    print(f"\n{'='*60}")
    print(f"Processing {n_messages:,} messages ({n_messages/1e6:.0f}M)")
    print(f"{'='*60}")
    
    print_memory_status()
    
    # データ生成
    gen_start = time.time()
    timestamps, addresses, data_bytes = generate_synthetic_can_data(n_messages)
    gen_time = time.time() - gen_start
    
    # GPU処理
    print("\nProcessing on GPU...")
    gpu_start = time.time()
    
    try:
        # バッチ処理
        if n_messages <= 100_000_000:  # 100M以下は一括処理
            results = decoder.decode_batch(timestamps, addresses, data_bytes)
        else:  # 100M超はチャンク処理
            print("Using chunked processing for large dataset...")
            results = process_in_chunks(decoder, timestamps, addresses, data_bytes, 
                                      chunk_size=50_000_000)
        
        cp.cuda.Stream.null.synchronize()
        gpu_time = time.time() - gpu_start
        
        # 結果統計
        n_decoded = sum(len(df) for df in results.values() if df is not None)
        
        # パフォーマンス指標
        data_size_gb = (timestamps.nbytes + addresses.nbytes + data_bytes.nbytes) / (1024**3)
        throughput_msg = n_messages / gpu_time / 1e6  # Mmsg/s
        throughput_gb = data_size_gb / gpu_time  # GB/s
        
        print(f"\n✓ Processing complete!")
        print(f"  Time: {gpu_time:.3f} seconds")
        print(f"  Throughput: {throughput_msg:.1f} Mmsg/s ({throughput_gb:.2f} GB/s)")
        print(f"  Decoded messages: {n_decoded:,}")
        
        print_memory_status()
        
        return {
            'n_messages': n_messages,
            'gpu_time': gpu_time,
            'throughput_mmsg': throughput_msg,
            'throughput_gb': throughput_gb,
            'n_decoded': n_decoded,
            'success': True
        }
        
    except Exception as e:
        print(f"\n✗ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'n_messages': n_messages,
            'success': False,
            'error': str(e)
        }
    finally:
        # クリーンアップ
        del timestamps, addresses, data_bytes
        if 'results' in locals():
            del results
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()

def process_in_chunks(decoder, timestamps, addresses, data_bytes, chunk_size=50_000_000):
    """大規模データのチャンク処理"""
    n_messages = len(timestamps)
    n_chunks = (n_messages + chunk_size - 1) // chunk_size
    
    all_results = {signal: [] for signal in decoder.signal_configs.keys()}
    
    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, n_messages)
        
        print(f"  Processing chunk {chunk_idx + 1}/{n_chunks} ({end_idx - start_idx:,} messages)...")
        
        # チャンクデータ
        chunk_timestamps = timestamps[start_idx:end_idx]
        chunk_addresses = addresses[start_idx:end_idx]
        chunk_data = data_bytes[start_idx:end_idx]
        
        # 処理
        chunk_results = decoder.decode_batch(chunk_timestamps, chunk_addresses, chunk_data)
        
        # 結果を統合
        for signal, df in chunk_results.items():
            if df is not None and len(df) > 0:
                all_results[signal].append(df)
        
        # メモリクリーンアップ
        del chunk_timestamps, chunk_addresses, chunk_data, chunk_results
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
    
    # 全チャンクの結果を結合
    final_results = {}
    for signal, dfs in all_results.items():
        if dfs:
            final_results[signal] = cudf.concat(dfs, ignore_index=True)
            final_results[signal] = final_results[signal].sort_values('timestamp').reset_index(drop=True)
        else:
            final_results[signal] = None
    
    return final_results

def main():
    """メインベンチマーク実行"""
    print("CAN GPU Processing Benchmark with Optimized RMM Settings")
    print("="*60)
    
    # デコーダー初期化
    decoder = GPUCANDecoder(batch_size=10_000_000)
    
    # テストサイズ（段階的に増加）
    test_sizes = [
        10_000_000,     # 10M (240MB)
        50_000_000,     # 50M (1.2GB)
        100_000_000,    # 100M (2.4GB)
        200_000_000,    # 200M (4.8GB)
        500_000_000,    # 500M (12GB)
        1_000_000_000,  # 1B (24GB)
    ]
    
    results = []
    
    for n_messages in test_sizes:
        # メモリチェック
        required_gb = (n_messages * 24) / (1024**3)
        device = cp.cuda.Device()
        free_gb = device.mem_info[0] / (1024**3)
        
        print(f"\nNext test: {n_messages:,} messages (requires ~{required_gb:.1f} GB)")
        
        if required_gb > free_gb * 1.2:  # 20%の安全マージン
            print(f"⚠️  Skipping - insufficient GPU memory (free: {free_gb:.1f} GB)")
            continue
        
        result = process_can_data(n_messages, decoder)
        results.append(result)
        
        # 大規模テスト後は追加のクリーンアップ
        if n_messages >= 500_000_000:
            print("\nPerforming deep memory cleanup...")
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
            time.sleep(2)  # GPUメモリの解放を待つ
    
    # 結果サマリー
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    
    successful = [r for r in results if r.get('success', False)]
    if successful:
        print(f"\nSuccessful tests: {len(successful)}/{len(results)}")
        print("\nPerformance Summary:")
        print(f"{'Messages':>15} {'Time (s)':>10} {'Throughput':>15} {'GB/s':>10}")
        print("-" * 60)
        
        for r in successful:
            print(f"{r['n_messages']:>15,} {r['gpu_time']:>10.2f} "
                  f"{r['throughput_mmsg']:>10.1f} Mmsg/s {r['throughput_gb']:>10.2f}")
        
        # 最大処理量
        max_messages = max(r['n_messages'] for r in successful)
        max_result = next(r for r in successful if r['n_messages'] == max_messages)
        print(f"\nMaximum processed: {max_messages:,} messages")
        print(f"Peak throughput: {max_result['throughput_mmsg']:.1f} Mmsg/s")
    
    # 推奨事項
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS")
    print(f"{'='*60}")
    
    print("\n1. RMM Configuration (Optimal for 24GB GPU):")
    print("   rmm.reinitialize(")
    print("       managed_memory=True,")
    print("       pool_allocator=True,")
    print("       initial_pool_size=8<<30,   # 8GB")
    print("       maximum_pool_size=22<<30   # 22GB")
    print("   )")
    
    print("\n2. Batch Sizes:")
    print("   - Small datasets (<100M): Process in single batch")
    print("   - Medium datasets (100M-500M): Use 50M chunks")
    print("   - Large datasets (>500M): Use 100M chunks with cleanup")
    
    print("\n3. Memory Management:")
    print("   - Always call gc.collect() between large batches")
    print("   - Use cp.get_default_memory_pool().free_all_blocks()")
    print("   - Monitor GPU memory usage during processing")
    
    print("\n4. For 1B messages specifically:")
    print("   - Use 10 chunks of 100M messages each")
    print("   - Expected processing time: ~10-15 seconds total")
    print("   - Ensure no other GPU processes are running")

if __name__ == "__main__":
    main()