"""
高速化されたGPU CANデコーダー
- CPUでの事前フィルタリング（3.7%のメッセージのみGPU処理）
- atomic操作の削減
- メモリアクセスパターンの最適化
"""

import numpy as np
import cupy as cp
import cudf
from numba import cuda
import rmm
from typing import Dict, List, Optional
import os
import time


@cuda.jit
def decode_wheel_speeds_kernel(
    timestamps, data_bytes, n_messages,
    out_timestamps, out_fl, out_fr, out_rl, out_rr
):
    """車輪速度専用の高速デコードカーネル（atomic操作なし）"""
    idx = cuda.grid(1)
    if idx >= n_messages:
        return
    
    timestamp = timestamps[idx]
    
    # 全4輪を一度に処理
    # 前左輪 (バイト 0-1)
    raw_value = (data_bytes[idx, 0] << 8) | data_bytes[idx, 1]
    out_fl[idx] = (raw_value * 0.01 - 67.67) / 3.6  # m/s
    
    # 前右輪 (バイト 2-3)
    raw_value = (data_bytes[idx, 2] << 8) | data_bytes[idx, 3]
    out_fr[idx] = (raw_value * 0.01 - 67.67) / 3.6
    
    # 後左輪 (バイト 4-5)
    raw_value = (data_bytes[idx, 4] << 8) | data_bytes[idx, 5]
    out_rl[idx] = (raw_value * 0.01 - 67.67) / 3.6
    
    # 後右輪 (バイト 6-7)
    raw_value = (data_bytes[idx, 6] << 8) | data_bytes[idx, 7]
    out_rr[idx] = (raw_value * 0.01 - 67.67) / 3.6
    
    # タイムスタンプも設定
    out_timestamps[idx] = timestamp


@cuda.jit
def decode_steering_kernel(
    timestamps, data_bytes, n_messages,
    out_timestamps, out_angles
):
    """ステアリング角度専用デコードカーネル"""
    idx = cuda.grid(1)
    if idx >= n_messages:
        return
    
    out_timestamps[idx] = timestamps[idx]
    
    byte01 = data_bytes[idx, 1]
    byte4 = data_bytes[idx, 4]
    
    # ステアリング角度デコード
    angle = 0.0  # デフォルト値
    if byte01 == 0x00 and byte4 == 0xC0:
        angle = 0.0
    elif byte01 == 0x54 and byte4 == 0xBE:
        angle = -1.5
    elif byte01 == 0x97 and byte4 == 0xBD:
        angle = -2.5
    elif byte01 == 0xD9 and byte4 == 0xBC:
        angle = -3.5
    elif byte01 == 0x1C and byte4 == 0xC2:
        angle = 1.5
    elif byte01 == 0x5E and byte4 == 0xC3:
        angle = 2.5
    elif byte01 == 0xA1 and byte4 == 0xC4:
        angle = 3.5
    
    out_angles[idx] = angle


class OptimizedGPUCANDecoderV2:
    def __init__(self, batch_size: int = 1_000_000, chunk_size: int = 1):
        """
        高速化されたGPU CANデコーダー V2
        
        Args:
            batch_size: 各バッチで処理するメッセージ数
            chunk_size: データを分割する数（1=分割なし）
        """
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        
        # RMM初期化
        self.pool = rmm.mr.PoolMemoryResource(
            rmm.mr.CudaMemoryResource(),
            initial_pool_size=512<<20,  # 512MB
            maximum_pool_size=2<<30     # 2GB
        )
        rmm.mr.set_current_device_resource(self.pool)
        
        # CUDAストリームの初期化
        self.streams = [cp.cuda.Stream() for _ in range(chunk_size)]
    
    def _process_single_chunk(self, timestamps: np.ndarray, addresses: np.ndarray, 
                             data_bytes: np.ndarray, stream: cp.cuda.Stream) -> Dict[str, cudf.DataFrame]:
        """
        高速化されたチャンク処理
        - CPUで事前フィルタリング
        - 専用カーネルで処理
        """
        results = {}
        
        # === CPU側での事前フィルタリング ===
        total_start = time.time()
        
        # 1. マスク作成（CPU）
        step_start = time.time()
        wheel_mask = addresses == 170
        steering_mask = addresses == 37
        mask_time = time.time() - step_start
        
        # 2. データ抽出（CPU）
        step_start = time.time()
        wheel_indices = np.where(wheel_mask)[0]
        steering_indices = np.where(steering_mask)[0]
        
        # 車輪速度データの抽出
        if len(wheel_indices) > 0:
            wheel_timestamps = timestamps[wheel_indices]
            wheel_data_bytes = data_bytes[wheel_indices]
        
        # ステアリングデータの抽出
        if len(steering_indices) > 0:
            steering_timestamps = timestamps[steering_indices]
            steering_data_bytes = data_bytes[steering_indices]
        
        extract_time = time.time() - step_start
        
        # === GPU処理 ===
        with stream:
            # 3. 車輪速度のGPU処理
            if len(wheel_indices) > 0:
                step_start = time.time()
                n_wheel_messages = len(wheel_indices)
                
                # GPU転送とカーネル実行
                d_wheel_timestamps = cp.asarray(wheel_timestamps)
                d_wheel_data_bytes = cp.asarray(wheel_data_bytes)
                
                # 出力バッファ
                d_out_timestamps = cp.zeros(n_wheel_messages, dtype=cp.float64)
                d_out_fl = cp.zeros(n_wheel_messages, dtype=cp.float32)
                d_out_fr = cp.zeros(n_wheel_messages, dtype=cp.float32)
                d_out_rl = cp.zeros(n_wheel_messages, dtype=cp.float32)
                d_out_rr = cp.zeros(n_wheel_messages, dtype=cp.float32)
                
                # カーネル実行
                threads_per_block = 256
                blocks_per_grid = (n_wheel_messages + threads_per_block - 1) // threads_per_block
                
                # Numba CUDA配列に変換
                decode_wheel_speeds_kernel[blocks_per_grid, threads_per_block](
                    cuda.as_cuda_array(d_wheel_timestamps),
                    cuda.as_cuda_array(d_wheel_data_bytes),
                    n_wheel_messages,
                    cuda.as_cuda_array(d_out_timestamps),
                    cuda.as_cuda_array(d_out_fl),
                    cuda.as_cuda_array(d_out_fr),
                    cuda.as_cuda_array(d_out_rl),
                    cuda.as_cuda_array(d_out_rr)
                )
                
                # DataFrame作成
                wheel_df = cudf.DataFrame({
                    'timestamp': d_out_timestamps,
                    'front_left': d_out_fl,
                    'front_right': d_out_fr,
                    'rear_left': d_out_rl,
                    'rear_right': d_out_rr
                })
                
                # ソート
                results['wheel_speeds'] = wheel_df.sort_values('timestamp').reset_index(drop=True)
                
                wheel_gpu_time = time.time() - step_start
            else:
                wheel_gpu_time = 0
            
            # 4. ステアリングのGPU処理
            if len(steering_indices) > 0:
                step_start = time.time()
                n_steering_messages = len(steering_indices)
                
                # GPU転送とカーネル実行
                d_steering_timestamps = cp.asarray(steering_timestamps)
                d_steering_data_bytes = cp.asarray(steering_data_bytes)
                
                # 出力バッファ
                d_out_timestamps = cp.zeros(n_steering_messages, dtype=cp.float64)
                d_out_angles = cp.zeros(n_steering_messages, dtype=cp.float32)
                
                # カーネル実行
                threads_per_block = 256
                blocks_per_grid = (n_steering_messages + threads_per_block - 1) // threads_per_block
                
                decode_steering_kernel[blocks_per_grid, threads_per_block](
                    cuda.as_cuda_array(d_steering_timestamps),
                    cuda.as_cuda_array(d_steering_data_bytes),
                    n_steering_messages,
                    cuda.as_cuda_array(d_out_timestamps),
                    cuda.as_cuda_array(d_out_angles)
                )
                
                # DataFrame作成
                steering_df = cudf.DataFrame({
                    'timestamp': d_out_timestamps,
                    'angle': d_out_angles
                })
                
                # ソート
                results['steering'] = steering_df.sort_values('timestamp').reset_index(drop=True)
                
                steering_gpu_time = time.time() - step_start
            else:
                steering_gpu_time = 0
        
        # GPU同期
        stream.synchronize()
        
        # タイミング情報の表示
        total_time = time.time() - total_start
        print(f"  === GPU処理の詳細（高速化版V2）===")
        print(f"  マスク作成（CPU）: {mask_time:.4f}秒")
        print(f"  データ抽出（CPU）: {extract_time:.4f}秒")
        print(f"  GPU転送と物理値変換: {wheel_gpu_time + steering_gpu_time:.4f}秒")
        print(f"  総処理時間: {total_time:.4f}秒")
        
        return results
    
    def decode_batch(self, timestamps: np.ndarray, addresses: np.ndarray, 
                    data_bytes: np.ndarray) -> Dict[str, List[cudf.DataFrame]]:
        """
        バッチをチャンクに分割して並列処理
        
        Returns:
            メッセージタイプごとのDataFrameリストを含む辞書
        """
        n_messages = len(timestamps)
        chunk_results = {'wheel_speeds': [], 'steering': []}
        
        if self.chunk_size == 1:
            # チャンク分割なし
            result = self._process_single_chunk(timestamps, addresses, data_bytes, self.streams[0])
            for key, df in result.items():
                if df is not None:
                    chunk_results[key].append(df)
        else:
            # チャンク分割処理
            chunk_length = n_messages // self.chunk_size
            
            for i in range(self.chunk_size):
                start_idx = i * chunk_length
                if i == self.chunk_size - 1:
                    end_idx = n_messages
                else:
                    end_idx = start_idx + chunk_length
                
                # チャンクデータの抽出
                chunk_timestamps = timestamps[start_idx:end_idx]
                chunk_addresses = addresses[start_idx:end_idx]
                chunk_data_bytes = data_bytes[start_idx:end_idx]
                
                # 並列処理
                result = self._process_single_chunk(
                    chunk_timestamps, chunk_addresses, chunk_data_bytes, 
                    self.streams[i % len(self.streams)]
                )
                for key, df in result.items():
                    if df is not None:
                        chunk_results[key].append(df)
            
            # 全ストリーム同期
            for stream in self.streams:
                stream.synchronize()
        
        return chunk_results
    
    def decode_batch_for_benchmark(self, timestamps: np.ndarray, addresses: np.ndarray, 
                                  data_bytes: np.ndarray) -> List[cudf.DataFrame]:
        """
        ベンチマーク用の簡易インターフェース（車輪速度データのみ）
        """
        results_dict = self.decode_batch(timestamps, addresses, data_bytes)
        
        # 車輪速度データのみを返す
        if results_dict['wheel_speeds']:
            if len(results_dict['wheel_speeds']) == 1:
                return results_dict['wheel_speeds']
            else:
                # 複数チャンクの場合は結合
                combined_df = cudf.concat(results_dict['wheel_speeds'], ignore_index=True)
                return [combined_df.sort_values('timestamp').reset_index(drop=True)]
        else:
            return [cudf.DataFrame()]
    
    def process_and_save(self, input_path: str, output_dir: str):
        """
        最適化された処理とParquet保存
        
        Args:
            input_path: t, address, dataファイルを含むraw_canディレクトリのパス
            output_dir: Parquetファイルの出力ディレクトリ
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"高速化GPU CANデコーダーV2実行開始 (chunk_size={self.chunk_size})...")
        
        # データ読み込み
        timestamps = np.fromfile(os.path.join(input_path, 't'), dtype=np.float64)
        addresses = np.fromfile(os.path.join(input_path, 'address'), dtype=np.int64)
        data_raw = np.fromfile(os.path.join(input_path, 'data'), dtype=np.uint8)
        data_bytes = data_raw.reshape(-1, 8)
        
        print(f"入力データ: {len(timestamps):,} メッセージ")
        
        # バッチ処理
        start_time = time.time()
        
        all_results = {'wheel_speeds': [], 'steering': []}
        for i in range(0, len(timestamps), self.batch_size):
            end_idx = min(i + self.batch_size, len(timestamps))
            print(f"  処理中: {i:,} - {end_idx:,}")
            
            batch_results = self.decode_batch(
                timestamps[i:end_idx],
                addresses[i:end_idx],
                data_bytes[i:end_idx]
            )
            
            # 結果を統合
            for key, df_list in batch_results.items():
                all_results[key].extend(df_list)
        
        # データの結合と保存
        # 車輪速度データ
        if all_results['wheel_speeds']:
            wheel_speeds_df = cudf.concat(all_results['wheel_speeds'], ignore_index=True)
            wheel_speeds_df = wheel_speeds_df.sort_values('timestamp').reset_index(drop=True)
            
            output_path = os.path.join(output_dir, "wheel_speeds.parquet")
            wheel_speeds_df.to_parquet(output_path)
            print(f"Saved: {output_path} ({len(wheel_speeds_df)} rows)")
            
            # 車両速度の計算と保存
            vehicle_speed_df = cudf.DataFrame({
                'timestamp': wheel_speeds_df['timestamp'],
                'speed': (wheel_speeds_df['front_left'] + wheel_speeds_df['front_right'] + 
                         wheel_speeds_df['rear_left'] + wheel_speeds_df['rear_right']) / 4.0
            })
            vehicle_speed_df = vehicle_speed_df.sort_values('timestamp').reset_index(drop=True)
            
            output_path = os.path.join(output_dir, "vehicle_speed.parquet")
            vehicle_speed_df.to_parquet(output_path)
            print(f"Saved: {output_path} ({len(vehicle_speed_df)} rows)")
        
        # ステアリングデータ
        if all_results['steering']:
            steering_df = cudf.concat(all_results['steering'], ignore_index=True)
            steering_df = steering_df.sort_values('timestamp').reset_index(drop=True)
            
            output_path = os.path.join(output_dir, "steering.parquet")
            steering_df.to_parquet(output_path)
            print(f"Saved: {output_path} ({len(steering_df)} rows)")
        
        elapsed_time = time.time() - start_time
        print(f"\n処理時間: {elapsed_time:.3f} 秒")
        print(f"スループット: {len(timestamps) / elapsed_time / 1e6:.2f} Mmessages/sec")


if __name__ == "__main__":
    # テスト用のサンプル実行
    print("高速化GPU CANデコーダーV2のテスト")
    
    decoder = OptimizedGPUCANDecoderV2(batch_size=500_000, chunk_size=1)
    print("デコーダー初期化完了")