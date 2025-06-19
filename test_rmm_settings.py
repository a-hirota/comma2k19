#!/usr/bin/env python3
"""
RMMメモリ設定テストスクリプト
24GB GPU用の最適なRMM設定を検証します
"""

import os
import sys
import gc
import time
import numpy as np
import cupy as cp
import rmm
import psutil
from datetime import datetime

def print_gpu_memory_info():
    """GPU メモリ情報を表示"""
    device = cp.cuda.Device()
    mem_info = device.mem_info
    free_gb = mem_info[0] / (1024**3)
    total_gb = mem_info[1] / (1024**3)
    used_gb = total_gb - free_gb
    
    print(f"\nGPU Memory Status:")
    print(f"  Total: {total_gb:.2f} GB")
    print(f"  Used:  {used_gb:.2f} GB")
    print(f"  Free:  {free_gb:.2f} GB")
    
    # RMM メモリプール情報
    try:
        mempool = cp.get_default_memory_pool()
        print(f"\nRMM Memory Pool:")
        print(f"  Used:  {mempool.used_bytes() / (1024**3):.2f} GB")
        print(f"  Total: {mempool.total_bytes() / (1024**3):.2f} GB")
    except:
        print("  (No active memory pool)")
    
    return free_gb, total_gb

def print_system_memory_info():
    """システムメモリ情報を表示"""
    mem = psutil.virtual_memory()
    print(f"\nSystem Memory:")
    print(f"  Total:     {mem.total / (1024**3):.2f} GB")
    print(f"  Available: {mem.available / (1024**3):.2f} GB")
    print(f"  Used:      {mem.used / (1024**3):.2f} GB ({mem.percent:.1f}%)")

def test_rmm_configuration(config_name, init_func, test_sizes_mb):
    """特定のRMM設定でメモリアロケーションをテスト"""
    print(f"\n{'='*60}")
    print(f"Testing: {config_name}")
    print(f"{'='*60}")
    
    # RMMを初期化
    try:
        init_func()
        print("RMM initialized successfully")
    except Exception as e:
        print(f"Failed to initialize RMM: {e}")
        return []
    
    results = []
    
    for size_mb in test_sizes_mb:
        print(f"\n--- Allocating {size_mb} MB ---")
        print_gpu_memory_info()
        
        size_bytes = size_mb * 1024 * 1024
        n_elements = size_bytes // 8  # float64要素数
        
        try:
            # 割り当て時間を測定
            start_time = time.time()
            
            # CuPy配列を割り当て
            arr = cp.zeros(n_elements, dtype=cp.float64)
            
            # 実際に書き込んでメモリを確保
            arr[:] = 1.0
            cp.cuda.Stream.null.synchronize()
            
            alloc_time = time.time() - start_time
            
            print(f"✓ Successfully allocated {size_mb} MB in {alloc_time:.3f} seconds")
            print(f"  Throughput: {size_mb / alloc_time:.1f} MB/s")
            
            # メモリ情報を再表示
            print_gpu_memory_info()
            
            results.append({
                'size_mb': size_mb,
                'status': 'success',
                'time': alloc_time,
                'config': config_name
            })
            
            # クリーンアップ
            del arr
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
            
        except Exception as e:
            print(f"✗ Failed to allocate {size_mb} MB: {type(e).__name__}: {e}")
            results.append({
                'size_mb': size_mb,
                'status': 'failed',
                'time': 0,
                'config': config_name,
                'error': str(e)
            })
            
            # エラー後のクリーンアップ
            gc.collect()
            try:
                cp.get_default_memory_pool().free_all_blocks()
            except:
                pass
    
    return results

def main():
    """メインテスト実行"""
    print("RMM Memory Configuration Test")
    print(f"Time: {datetime.now()}")
    
    # 初期状態を表示
    print("\n=== Initial System State ===")
    print_system_memory_info()
    free_gb, total_gb = print_gpu_memory_info()
    
    # テストサイズ（MB単位）
    # 24GBのGPUに対して段階的にテスト
    test_sizes_mb = [
        100,      # 100MB
        500,      # 500MB
        1024,     # 1GB
        2048,     # 2GB
        5120,     # 5GB
        10240,    # 10GB
        15360,    # 15GB
        20480,    # 20GB
        22528     # 22GB (24GBの約92%)
    ]
    
    # 各RMM設定をテスト
    all_results = []
    
    # 設定1: デフォルト（プールなし）
    def init_default():
        rmm.reinitialize()
    
    results = test_rmm_configuration(
        "Default (No Pool)",
        init_default,
        test_sizes_mb[:5]  # 小さいサイズのみ
    )
    all_results.extend(results)
    
    # 設定2: 固定プール（10GB）
    def init_fixed_pool_10gb():
        rmm.reinitialize(
            pool_allocator=True,
            initial_pool_size=10<<30,  # 10GB
        )
    
    results = test_rmm_configuration(
        "Fixed Pool (10GB)",
        init_fixed_pool_10gb,
        test_sizes_mb[:6]  # 10GBまで
    )
    all_results.extend(results)
    
    # 設定3: 可変プール（2GB初期、20GB最大）
    def init_variable_pool():
        rmm.reinitialize(
            pool_allocator=True,
            initial_pool_size=2<<30,   # 2GB
            maximum_pool_size=20<<30   # 20GB
        )
    
    results = test_rmm_configuration(
        "Variable Pool (2-20GB)",
        init_variable_pool,
        test_sizes_mb[:8]  # 20GBまで
    )
    all_results.extend(results)
    
    # 設定4: Managed Memory + 可変プール
    def init_managed_variable():
        rmm.reinitialize(
            managed_memory=True,
            pool_allocator=True,
            initial_pool_size=2<<30,   # 2GB
            maximum_pool_size=22<<30   # 22GB (24GBの約92%)
        )
    
    results = test_rmm_configuration(
        "Managed Memory + Variable Pool (2-22GB)",
        init_managed_variable,
        test_sizes_mb  # 全サイズ
    )
    all_results.extend(results)
    
    # 設定5: 最適化設定（推奨）
    def init_optimized():
        # 24GB GPUに最適化
        # 初期プールを大きめに、最大値をGPUメモリの90%程度に設定
        rmm.reinitialize(
            managed_memory=True,
            pool_allocator=True,
            initial_pool_size=8<<30,   # 8GB (素早い初期割り当て)
            maximum_pool_size=22<<30   # 22GB (安全マージンを確保)
        )
    
    results = test_rmm_configuration(
        "Optimized (8-22GB, Managed)",
        init_optimized,
        test_sizes_mb  # 全サイズ
    )
    all_results.extend(results)
    
    # 結果サマリー
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    for config in ["Default (No Pool)", "Fixed Pool (10GB)", "Variable Pool (2-20GB)", 
                   "Managed Memory + Variable Pool (2-22GB)", "Optimized (8-22GB, Managed)"]:
        config_results = [r for r in all_results if r['config'] == config]
        if config_results:
            success_results = [r for r in config_results if r['status'] == 'success']
            if success_results:
                max_size = max(r['size_mb'] for r in success_results)
                avg_speed = np.mean([r['size_mb'] / r['time'] for r in success_results if r['time'] > 0])
                print(f"\n{config}:")
                print(f"  Max successful allocation: {max_size} MB ({max_size/1024:.1f} GB)")
                print(f"  Average throughput: {avg_speed:.1f} MB/s")
                print(f"  Success rate: {len(success_results)}/{len(config_results)}")
            else:
                print(f"\n{config}: All allocations failed")
    
    # 推奨設定
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS FOR 24GB GPU")
    print(f"{'='*60}")
    print("\n1. For maximum performance (no host memory fallback):")
    print("   rmm.reinitialize(")
    print("       pool_allocator=True,")
    print("       initial_pool_size=8<<30,   # 8GB")
    print("       maximum_pool_size=22<<30   # 22GB")
    print("   )")
    
    print("\n2. For handling larger datasets (with host memory fallback):")
    print("   rmm.reinitialize(")
    print("       managed_memory=True,")
    print("       pool_allocator=True,")
    print("       initial_pool_size=8<<30,   # 8GB")
    print("       maximum_pool_size=22<<30   # 22GB")
    print("   )")
    
    print("\n3. For your CAN data processing (100M messages = 2.24GB):")
    print("   - Use configuration #2 with managed memory")
    print("   - Process in chunks of 50-100M messages")
    print("   - This leaves headroom for intermediate calculations")

if __name__ == "__main__":
    main()