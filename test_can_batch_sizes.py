#!/usr/bin/env python3
"""
CANデータバッチサイズ最適化テスト
OOMを回避しながら最適なバッチサイズを見つけます
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

def setup_rmm_optimized():
    """24GB GPU用の最適化されたRMM設定"""
    rmm.reinitialize(
        managed_memory=True,
        pool_allocator=True,
        initial_pool_size=8<<30,   # 8GB initial
        maximum_pool_size=22<<30   # 22GB max (24GBの約92%)
    )
    print("RMM initialized with optimized settings:")
    print("  - Managed memory: Enabled")
    print("  - Initial pool: 8GB")
    print("  - Maximum pool: 22GB")

def generate_test_can_data(n_messages):
    """テスト用CANデータを生成"""
    # リアルなCANデータ分布
    timestamps = np.linspace(46408.0, 46468.0, n_messages)
    
    # アドレス分布（実際のCANデータに基づく）
    address_weights = {
        170: 0.037,  # 4輪速度
        37: 0.037,   # ステアリング
        36: 0.037,
        740: 0.044,
        180: 0.018,
    }
    
    addresses = []
    for addr, weight in address_weights.items():
        count = int(n_messages * weight)
        addresses.extend([addr] * count)
    
    # 残りはランダム
    remaining = n_messages - len(addresses)
    other_addrs = np.random.choice([452, 466, 467, 705], remaining)
    addresses.extend(other_addrs)
    
    np.random.shuffle(addresses)
    addresses = np.array(addresses[:n_messages], dtype=np.int64)
    
    # データバイト
    data_bytes = np.random.randint(0, 256, (n_messages, 8), dtype=np.uint8)
    
    return timestamps, addresses, data_bytes

def test_batch_processing(batch_sizes_millions):
    """異なるバッチサイズでのメモリ使用量とパフォーマンスをテスト"""
    
    print("\n=== Batch Size Testing for CAN Data ===")
    
    # GPUデコーダーを初期化
    decoder = GPUCANDecoder(batch_size=10_000_000)
    
    results = []
    
    for batch_size_m in batch_sizes_millions:
        batch_size = batch_size_m * 1_000_000
        print(f"\n--- Testing {batch_size_m}M messages ({batch_size:,}) ---")
        
        # メモリ状況を表示
        device = cp.cuda.Device()
        mem_info = device.mem_info
        free_gb_before = mem_info[0] / (1024**3)
        print(f"GPU free memory before: {free_gb_before:.2f} GB")
        
        try:
            # データ生成
            print("Generating test data...")
            gen_start = time.time()
            timestamps, addresses, data_bytes = generate_test_can_data(batch_size)
            gen_time = time.time() - gen_start
            
            data_size_mb = (timestamps.nbytes + addresses.nbytes + data_bytes.nbytes) / (1024**2)
            print(f"Data size: {data_size_mb:.1f} MB")
            print(f"Generation time: {gen_time:.2f} seconds")
            
            # GPUへの転送とデコード
            print("Processing on GPU...")
            proc_start = time.time()
            
            # CuDFデータフレームに変換
            timestamps_gpu = cp.asarray(timestamps)
            addresses_gpu = cp.asarray(addresses)
            data_gpu = cp.asarray(data_bytes)
            
            # デコード実行
            results_gpu = decoder.decode_batch(timestamps, addresses, data_bytes)
            
            # GPU同期
            cp.cuda.Stream.null.synchronize()
            proc_time = time.time() - proc_start
            
            # 結果の統計
            n_decoded = sum(len(df) for df in results_gpu.values() if df is not None)
            
            # メモリ使用量を確認
            mem_info_after = device.mem_info
            free_gb_after = mem_info_after[0] / (1024**3)
            memory_used_gb = free_gb_before - free_gb_after
            
            # パフォーマンス指標
            throughput_msg = batch_size / proc_time / 1e6  # Mmsg/s
            throughput_gb = (data_size_mb / 1024) / proc_time  # GB/s
            
            print(f"✓ Success!")
            print(f"  Processing time: {proc_time:.3f} seconds")
            print(f"  Throughput: {throughput_msg:.1f} Mmsg/s ({throughput_gb:.2f} GB/s)")
            print(f"  Memory used: {memory_used_gb:.2f} GB")
            print(f"  Decoded messages: {n_decoded:,}")
            
            results.append({
                'batch_size_m': batch_size_m,
                'success': True,
                'time': proc_time,
                'throughput_mmsg': throughput_msg,
                'throughput_gb': throughput_gb,
                'memory_gb': memory_used_gb,
                'n_decoded': n_decoded
            })
            
            # クリーンアップ
            del timestamps, addresses, data_bytes
            del timestamps_gpu, addresses_gpu, data_gpu, results_gpu
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
            
        except Exception as e:
            print(f"✗ Failed: {type(e).__name__}: {e}")
            results.append({
                'batch_size_m': batch_size_m,
                'success': False,
                'error': str(e)
            })
            
            # エラー後のクリーンアップ
            gc.collect()
            try:
                cp.get_default_memory_pool().free_all_blocks()
            except:
                pass
    
    return results

def find_optimal_batch_size():
    """最適なバッチサイズを見つける"""
    
    # 初期テストサイズ（百万メッセージ単位）
    test_sizes = [1, 5, 10, 25, 50, 75, 100, 150, 200, 250, 300]
    
    print("\n=== Finding Optimal Batch Size ===")
    results = test_batch_processing(test_sizes)
    
    # 成功した最大バッチサイズを見つける
    successful = [r for r in results if r.get('success', False)]
    if successful:
        max_batch = max(r['batch_size_m'] for r in successful)
        max_result = next(r for r in successful if r['batch_size_m'] == max_batch)
        
        print(f"\n=== Optimal Configuration Found ===")
        print(f"Maximum successful batch size: {max_batch}M messages")
        print(f"  Throughput: {max_result['throughput_mmsg']:.1f} Mmsg/s")
        print(f"  Memory used: {max_result['memory_gb']:.2f} GB")
        
        # 安全マージンを考慮した推奨サイズ
        recommended = int(max_batch * 0.8)
        print(f"\nRecommended batch size (80% of max): {recommended}M messages")
        
        return recommended
    else:
        print("\nNo successful batch sizes found!")
        return None

def test_chunked_processing(total_messages_m, chunk_size_m):
    """チャンク処理のテスト"""
    print(f"\n=== Testing Chunked Processing ===")
    print(f"Total messages: {total_messages_m}M")
    print(f"Chunk size: {chunk_size_m}M")
    
    n_chunks = (total_messages_m + chunk_size_m - 1) // chunk_size_m
    print(f"Number of chunks: {n_chunks}")
    
    decoder = GPUCANDecoder(batch_size=10_000_000)
    
    total_time = 0
    total_decoded = 0
    
    for chunk_idx in range(n_chunks):
        chunk_start = chunk_idx * chunk_size_m
        chunk_end = min(chunk_start + chunk_size_m, total_messages_m)
        chunk_messages = (chunk_end - chunk_start) * 1_000_000
        
        print(f"\n  Chunk {chunk_idx + 1}/{n_chunks}: {chunk_messages:,} messages")
        
        try:
            # データ生成
            timestamps, addresses, data_bytes = generate_test_can_data(chunk_messages)
            
            # 処理
            start_time = time.time()
            results = decoder.decode_batch(timestamps, addresses, data_bytes)
            cp.cuda.Stream.null.synchronize()
            chunk_time = time.time() - start_time
            
            n_decoded = sum(len(df) for df in results.values() if df is not None)
            
            total_time += chunk_time
            total_decoded += n_decoded
            
            print(f"    Time: {chunk_time:.2f}s, Decoded: {n_decoded:,}")
            
            # クリーンアップ
            del timestamps, addresses, data_bytes, results
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
            
        except Exception as e:
            print(f"    Failed: {e}")
            break
    
    if total_time > 0:
        avg_throughput = (total_messages_m * 1_000_000) / total_time / 1e6
        print(f"\nTotal processing time: {total_time:.2f} seconds")
        print(f"Average throughput: {avg_throughput:.1f} Mmsg/s")
        print(f"Total decoded messages: {total_decoded:,}")

def main():
    """メインテスト実行"""
    print("CAN Data Batch Size Optimization Test")
    print("=====================================")
    
    # システム情報を表示
    print("\nSystem Information:")
    mem = psutil.virtual_memory()
    print(f"  RAM: {mem.total / (1024**3):.1f} GB")
    
    device = cp.cuda.Device()
    mem_info = device.mem_info
    print(f"  GPU: {mem_info[1] / (1024**3):.1f} GB")
    
    # RMM設定
    print("\nInitializing RMM...")
    setup_rmm_optimized()
    
    # 最適なバッチサイズを見つける
    optimal_batch_size_m = find_optimal_batch_size()
    
    if optimal_batch_size_m:
        # 大規模データのチャンク処理をテスト
        print("\n" + "="*50)
        print("Testing large-scale processing with optimal chunk size")
        print("="*50)
        
        # 1億メッセージをチャンク処理
        test_chunked_processing(100, optimal_batch_size_m)
        
        # 10億メッセージの推定
        print("\n=== Estimated Performance for 1B Messages ===")
        chunks_needed = (1000 + optimal_batch_size_m - 1) // optimal_batch_size_m
        estimated_time = chunks_needed * (100 / optimal_batch_size_m) * 60  # 分単位
        print(f"Chunks needed: {chunks_needed}")
        print(f"Estimated time: {estimated_time:.1f} minutes")
    
    # 最終推奨事項
    print("\n" + "="*50)
    print("RECOMMENDATIONS")
    print("="*50)
    print("\n1. RMM Configuration:")
    print("   rmm.reinitialize(")
    print("       managed_memory=True,")
    print("       pool_allocator=True,")
    print("       initial_pool_size=8<<30,   # 8GB")
    print("       maximum_pool_size=22<<30   # 22GB")
    print("   )")
    
    if optimal_batch_size_m:
        print(f"\n2. Batch Size:")
        print(f"   - Optimal: {optimal_batch_size_m}M messages per batch")
        print(f"   - Safe: {int(optimal_batch_size_m * 0.8)}M messages per batch")
    
    print("\n3. Processing Strategy:")
    print("   - Use chunked processing for datasets > 100M messages")
    print("   - Free memory between chunks with gc.collect()")
    print("   - Monitor GPU memory usage during processing")

if __name__ == "__main__":
    main()