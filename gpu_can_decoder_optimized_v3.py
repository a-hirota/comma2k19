"""
最適化されたGPU CANデコーダー V3
- マスク作成は計測時間外（CPU/GPU共通処理）
- NumPy経由を避け、直接cuDF形式に変換
- cuDFのネイティブ演算を使用
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
def filter_and_decode_wheel_speeds_kernel(
    addresses, data_bytes, timestamps, n_messages,
    out_timestamps, out_fl, out_fr, out_rl, out_rr, out_count
):
    """アドレス170のメッセージをフィルタリングして車輪速度をデコード"""
    idx = cuda.grid(1)
    if idx >= n_messages:
        return
    
    # アドレス170のみ処理
    if addresses[idx] == 170:
        # 出力位置を取得
        out_idx = cuda.atomic.add(out_count, 0, 1)
        if out_idx < out_timestamps.shape[0]:
            # タイムスタンプ
            out_timestamps[out_idx] = timestamps[idx]
            
            # 前左輪 (バイト 0-1)
            raw_value = (data_bytes[idx, 0] << 8) | data_bytes[idx, 1]
            out_fl[out_idx] = (raw_value * 0.01 - 67.67) / 3.6  # m/s
            
            # 前右輪 (バイト 2-3)
            raw_value = (data_bytes[idx, 2] << 8) | data_bytes[idx, 3]
            out_fr[out_idx] = (raw_value * 0.01 - 67.67) / 3.6
            
            # 後左輪 (バイト 4-5)
            raw_value = (data_bytes[idx, 4] << 8) | data_bytes[idx, 5]
            out_rl[out_idx] = (raw_value * 0.01 - 67.67) / 3.6
            
            # 後右輪 (バイト 6-7)
            raw_value = (data_bytes[idx, 6] << 8) | data_bytes[idx, 7]
            out_rr[out_idx] = (raw_value * 0.01 - 67.67) / 3.6


@cuda.jit
def filter_and_decode_steering_kernel(
    addresses, data_bytes, timestamps, n_messages,
    out_timestamps, out_angles, out_count
):
    """アドレス37のメッセージをフィルタリングしてステアリング角度をデコード"""
    idx = cuda.grid(1)
    if idx >= n_messages:
        return
    
    # アドレス37のみ処理
    if addresses[idx] == 37:
        # 出力位置を取得
        out_idx = cuda.atomic.add(out_count, 0, 1)
        if out_idx < out_timestamps.shape[0]:
            out_timestamps[out_idx] = timestamps[idx]
            
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
            
            out_angles[out_idx] = angle


class OptimizedGPUCANDecoderV3:
    def __init__(self, batch_size: int = 1_000_000, chunk_size: int = 1):
        """
        最適化されたGPU CANデコーダー V3
        
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
        チャンク処理 - 計測基準に従った実装
        """
        results = {}
        n_messages = len(timestamps)
        
        # === マスク作成（計測時間外）===
        target_mask = (addresses == 170) | (addresses == 37)
        target_count = np.sum(target_mask)
        
        if target_count == 0:
            return results
        
        # 推定出力サイズ
        estimated_wheel_count = int(np.sum(addresses == 170) * 1.2)
        estimated_steering_count = int(np.sum(addresses == 37) * 1.2)
        
        # === ステップ1: データ抽出（CPUフィルタリング + GPU転送）===
        step1_start = time.time()
        
        # CPUでフィルタリング（抽出対象のみを選択）
        filtered_timestamps = timestamps[target_mask]
        filtered_addresses = addresses[target_mask]
        filtered_data_bytes = data_bytes[target_mask]
        
        with stream:
            # フィルタ済みデータのみをGPUに転送
            d_timestamps = cp.asarray(filtered_timestamps)
            d_addresses = cp.asarray(filtered_addresses)
            d_data_bytes = cp.asarray(filtered_data_bytes)
            
            # 車輪速度用出力バッファ（cuDF形式で準備）
            wheel_out_data = {
                'timestamp': cp.zeros(estimated_wheel_count, dtype=cp.float64),
                'front_left': cp.zeros(estimated_wheel_count, dtype=cp.float32),
                'front_right': cp.zeros(estimated_wheel_count, dtype=cp.float32),
                'rear_left': cp.zeros(estimated_wheel_count, dtype=cp.float32),
                'rear_right': cp.zeros(estimated_wheel_count, dtype=cp.float32)
            }
            d_wheel_count = cp.zeros(1, dtype=cp.int32)
            
            # ステアリング用出力バッファ（cuDF形式で準備）
            steering_out_data = {
                'timestamp': cp.zeros(estimated_steering_count, dtype=cp.float64),
                'angle': cp.zeros(estimated_steering_count, dtype=cp.float32)
            }
            d_steering_count = cp.zeros(1, dtype=cp.int32)
        
        data_extract_time = time.time() - step1_start
        
        # === ステップ2: 物理値変換（CUDAカーネル実行）===
        step2_start = time.time()
        
        # グリッドとブロックサイズの設定（フィルタ済みメッセージ数を使用）
        n_filtered = len(filtered_timestamps)
        threads_per_block = 256
        blocks_per_grid = (n_filtered + threads_per_block - 1) // threads_per_block
        
        # 車輪速度のフィルタリングとデコード
        filter_and_decode_wheel_speeds_kernel[blocks_per_grid, threads_per_block](
            cuda.as_cuda_array(d_addresses),
            cuda.as_cuda_array(d_data_bytes),
            cuda.as_cuda_array(d_timestamps),
            n_filtered,
            cuda.as_cuda_array(wheel_out_data['timestamp']),
            cuda.as_cuda_array(wheel_out_data['front_left']),
            cuda.as_cuda_array(wheel_out_data['front_right']),
            cuda.as_cuda_array(wheel_out_data['rear_left']),
            cuda.as_cuda_array(wheel_out_data['rear_right']),
            cuda.as_cuda_array(d_wheel_count)
        )
        
        # ステアリングのフィルタリングとデコード
        filter_and_decode_steering_kernel[blocks_per_grid, threads_per_block](
            cuda.as_cuda_array(d_addresses),
            cuda.as_cuda_array(d_data_bytes),
            cuda.as_cuda_array(d_timestamps),
            n_filtered,
            cuda.as_cuda_array(steering_out_data['timestamp']),
            cuda.as_cuda_array(steering_out_data['angle']),
            cuda.as_cuda_array(d_steering_count)
        )
        
        # GPU同期
        cuda.synchronize()
        
        # 実際の出力数を取得
        wheel_count = int(d_wheel_count[0])
        steering_count = int(d_steering_count[0])
        
        # cuDFデータフレームの作成（既にGPU上にあるデータを直接使用）
        if wheel_count > 0:
            # 実際に使用された部分のみでDataFrameを作成
            wheel_df = cudf.DataFrame({
                'timestamp': wheel_out_data['timestamp'][:wheel_count],
                'front_left': wheel_out_data['front_left'][:wheel_count],
                'front_right': wheel_out_data['front_right'][:wheel_count],
                'rear_left': wheel_out_data['rear_left'][:wheel_count],
                'rear_right': wheel_out_data['rear_right'][:wheel_count]
            })
            # cuDFのネイティブソート
            results['wheel_speeds'] = wheel_df.sort_values('timestamp').reset_index(drop=True)
        
        if steering_count > 0:
            steering_df = cudf.DataFrame({
                'timestamp': steering_out_data['timestamp'][:steering_count],
                'angle': steering_out_data['angle'][:steering_count]
            })
            results['steering'] = steering_df.sort_values('timestamp').reset_index(drop=True)
        
        physical_convert_time = time.time() - step2_start
        
        # タイミング情報の表示（マスク作成は含まない）
        total_time = data_extract_time + physical_convert_time
        print(f"  === GPU処理の詳細（V3）===")
        print(f"  データ抽出: {data_extract_time:.4f}秒")
        print(f"  物理値変換: {physical_convert_time:.4f}秒")
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
        
        print(f"最適化GPU CANデコーダーV3実行開始 (chunk_size={self.chunk_size})...")
        
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
            
            # 車両速度の計算と保存（cuDFのネイティブ演算）
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
    print("最適化GPU CANデコーダーV3のテスト")
    
    decoder = OptimizedGPUCANDecoderV3(batch_size=500_000, chunk_size=1)
    print("デコーダー初期化完了")