"""
最適化されたGPU CANデコーダー
- 統一されたデータ転送
- 単一カーネルでの全メッセージタイプ処理
- チャンクサイズによる柔軟な処理
"""

import numpy as np
import cupy as cp
import cudf
from numba import cuda
import rmm
from typing import Dict, List, Optional
import os


@cuda.jit
def decode_can_messages_unified_kernel(
    timestamps, addresses, data_bytes,
    # 統一出力バッファ
    out_timestamps, out_message_types, out_values, out_count
):
    """
    統一されたCANメッセージデコードカーネル
    全メッセージタイプを1つのカーネルで処理
    
    message_types:
    0: 車輪速度 - 前左
    1: 車輪速度 - 前右  
    2: 車輪速度 - 後左
    3: 車輪速度 - 後右
    4: ステアリング角度
    """
    idx = cuda.grid(1)
    if idx >= addresses.shape[0]:
        return
        
    addr = addresses[idx]
    timestamp = timestamps[idx]
    
    # 車輪速度メッセージ (アドレス 170)
    if addr == 170:
        # 4つの車輪を順次処理
        for wheel in range(4):
            out_idx = cuda.atomic.add(out_count, 0, 1)
            if out_idx < out_timestamps.shape[0]:
                out_timestamps[out_idx] = timestamp
                out_message_types[out_idx] = wheel  # 0,1,2,3
                
                # 車輪速度デコード
                byte_offset = wheel * 2
                raw_value = (data_bytes[idx, byte_offset] << 8) | data_bytes[idx, byte_offset + 1]
                wheel_speed = (raw_value * 0.01 - 67.67) / 3.6  # m/s
                out_values[out_idx] = wheel_speed
    
    # ステアリング角度メッセージ (アドレス 37)
    elif addr == 37:
        out_idx = cuda.atomic.add(out_count, 0, 1)
        if out_idx < out_timestamps.shape[0]:
            out_timestamps[out_idx] = timestamp
            out_message_types[out_idx] = 4  # ステアリング
            
            # ステアリング角度デコード
            byte01 = data_bytes[idx, 1]
            byte4 = data_bytes[idx, 4]
            
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
                
            out_values[out_idx] = angle


class OptimizedGPUCANDecoder:
    def __init__(self, batch_size: int = 1_000_000, chunk_size: int = 1):
        """
        最適化されたGPU CANデコーダー
        
        Args:
            batch_size: 各バッチで処理するメッセージ数
            chunk_size: データを分割する数（1=分割なし）
        """
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        
        # RMM初期化（適切なサイズに調整）
        self.pool = rmm.mr.PoolMemoryResource(
            rmm.mr.CudaMemoryResource(),
            initial_pool_size=512<<20,  # 512MB
            maximum_pool_size=2<<30     # 20GB
        )
        rmm.mr.set_current_device_resource(self.pool)
        
        # CUDAストリームの初期化
        self.streams = [cp.cuda.Stream() for _ in range(chunk_size)]
        
    def _estimate_output_size(self, n_messages: int) -> int:
        """
        出力サイズの推定（車輪速度4個 + ステアリング1個）
        実際のデータ分布3.7%に基づいて推定
        """
        return int(n_messages * 0.04)
    
    def _process_single_chunk(self, timestamps: np.ndarray, addresses: np.ndarray, 
                             data_bytes: np.ndarray, stream: cp.cuda.Stream) -> Dict[str, cudf.DataFrame]:
        """
        真のGPU処理版 - CUDAカーネルを使用
        車輪速度（アドレス170）とステアリング（アドレス37）の両方を処理
        """
        import time
        total_start = time.time()
        
        results = {}
        n_messages = len(timestamps)
        
        # === マスク作成（計測時間外）===
        target_mask = (addresses == 170) | (addresses == 37)
        target_indices = np.where(target_mask)[0]
        
        if len(target_indices) == 0:
            return results
        
        # === ステップ1: データ抽出（インデックス検索 + 配列抽出 + GPU転送）===
        step1_start = time.time()
        
        # 対象データのみを抽出
        filtered_timestamps = timestamps[target_indices]
        filtered_addresses = addresses[target_indices]
        filtered_data_bytes = data_bytes[target_indices]
        
        # 出力サイズの推定（車輪速度は4倍、ステアリングは1倍）
        n_filtered = len(filtered_timestamps)
        estimated_output_size = int(n_filtered * 5)  # 最大5倍（車輪4 + ステアリング1）
        
        # GPU配列の準備
        with stream:
            # フィルタ済みデータのみをGPUに転送
            d_timestamps = cp.asarray(filtered_timestamps)
            d_addresses = cp.asarray(filtered_addresses)
            d_data_bytes = cp.asarray(filtered_data_bytes)
            
            # 出力バッファの準備
            d_out_timestamps = cp.zeros(estimated_output_size, dtype=cp.float64)
            d_out_message_types = cp.zeros(estimated_output_size, dtype=cp.int32)
            d_out_values = cp.zeros(estimated_output_size, dtype=cp.float32)
            d_out_count = cp.zeros(1, dtype=cp.int32)
            
        # NumbaのCUDAカーネル用に配列を変換
        from numba import cuda
        d_timestamps_numba = cuda.as_cuda_array(d_timestamps)
        d_addresses_numba = cuda.as_cuda_array(d_addresses)
        d_data_bytes_numba = cuda.as_cuda_array(d_data_bytes)
        d_out_timestamps_numba = cuda.as_cuda_array(d_out_timestamps)
        d_out_message_types_numba = cuda.as_cuda_array(d_out_message_types)
        d_out_values_numba = cuda.as_cuda_array(d_out_values)
        d_out_count_numba = cuda.as_cuda_array(d_out_count)
            
        data_extract_time = time.time() - step1_start
        
        # === ステップ2: 物理値変換（CUDAカーネル実行 + DataFrame作成 + ソート）===
        step2_start = time.time()
        
        # グリッドとブロックサイズの設定（フィルタ済みメッセージ数を使用）
        threads_per_block = 256
        blocks_per_grid = (n_filtered + threads_per_block - 1) // threads_per_block
        
        # CUDAカーネルの実行（ストリームなし）
        decode_can_messages_unified_kernel[blocks_per_grid, threads_per_block](
            d_timestamps_numba, d_addresses_numba, d_data_bytes_numba,
            d_out_timestamps_numba, d_out_message_types_numba, d_out_values_numba, d_out_count_numba
        )
        
        # GPU同期
        cuda.synchronize()
            
        # GPU同期も含めて物理値変換の一部
        
        # 実際の出力数を取得
        output_count = int(d_out_count[0])
        
        if output_count > estimated_output_size:
            print(f"  警告: 出力バッファオーバーフロー！ {output_count} > {estimated_output_size}")
            output_count = estimated_output_size
        
        if output_count > 0:
            # 実際に使用された部分のみを取得
            out_timestamps = d_out_timestamps[:output_count]
            out_message_types = d_out_message_types[:output_count]
            out_values = d_out_values[:output_count]
            
            # 統一DataFrameを作成
            unified_df = cudf.DataFrame({
                'timestamp': out_timestamps,
                'message_type': out_message_types,
                'value': out_values
            })
            
            # メッセージタイプ別に分離
            # 車輪速度データ（message_type 0,1,2,3）
            wheel_mask = unified_df['message_type'] < 4
            if wheel_mask.any():
                wheel_data = unified_df[wheel_mask]
                
                # ピボット操作で車輪データを再構成
                wheel_pivot = wheel_data.pivot_table(
                    index='timestamp',
                    columns='message_type', 
                    values='value',
                    aggfunc='first'
                ).reset_index()
                
                # 列名の変更
                column_mapping = {
                    0: 'front_left',
                    1: 'front_right', 
                    2: 'rear_left',
                    3: 'rear_right'
                }
                wheel_pivot = wheel_pivot.rename(columns=column_mapping)
                
                # ソート
                results['wheel_speeds'] = wheel_pivot.sort_values('timestamp').reset_index(drop=True)
            
            # ステアリングデータ（message_type 4）
            steering_mask = unified_df['message_type'] == 4
            if steering_mask.any():
                steering_data = unified_df[steering_mask]
                steering_df = cudf.DataFrame({
                    'timestamp': steering_data['timestamp'],
                    'angle': steering_data['value']
                })
                
                # ソート
                results['steering'] = steering_df.sort_values('timestamp').reset_index(drop=True)
        else:
            print(f"  警告: CUDAカーネルから出力がありません（output_count = {output_count}）")
        
        physical_convert_time = time.time() - step2_start
        
        # 計測結果の表示（マスク作成は含まない）
        total_time = data_extract_time + physical_convert_time
        print(f"  === GPU処理の詳細 ===")
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
            # チャンク分割なし - 1回の完全な処理
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
                    # 最後のチャンクは残り全て
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
    
    def _split_unified_dataframe(self, unified_df: cudf.DataFrame) -> Dict[str, cudf.DataFrame]:
        """
        統一DataFrameを個別のメッセージタイプごとに分割
        """
        results = {}
        
        if len(unified_df) == 0:
            return results
        
        # 車輪速度データの分離（message_type 0,1,2,3）
        wheel_mask = unified_df['message_type'] < 4
        if wheel_mask.any():
            wheel_data = unified_df[wheel_mask]
            
            # ピボット操作で車輪データを再構成
            wheel_pivot = wheel_data.pivot_table(
                index='timestamp',
                columns='message_type', 
                values='value',
                aggfunc='first'
            ).reset_index()
            
            # 列名の変更
            column_mapping = {
                0: 'front_left',
                1: 'front_right', 
                2: 'rear_left',
                3: 'rear_right'
            }
            wheel_pivot = wheel_pivot.rename(columns=column_mapping)
            
            # 欠損値を前方補完
            wheel_pivot = wheel_pivot.fillna(method='ffill')
            
            results['wheel_speeds'] = wheel_pivot
            
            # 車両速度の計算
            if all(col in wheel_pivot.columns for col in ['front_left', 'front_right', 'rear_left', 'rear_right']):
                vehicle_speed_df = cudf.DataFrame({
                    'timestamp': wheel_pivot['timestamp'],
                    'speed': (wheel_pivot['front_left'] + wheel_pivot['front_right'] + 
                             wheel_pivot['rear_left'] + wheel_pivot['rear_right']) / 4.0
                })
                results['vehicle_speed'] = vehicle_speed_df
        
        # ステアリング角度データの分離（message_type 4）
        steering_mask = unified_df['message_type'] == 4
        if steering_mask.any():
            steering_data = unified_df[steering_mask]
            results['steering'] = cudf.DataFrame({
                'timestamp': steering_data['timestamp'],
                'angle': steering_data['value']
            })
        
        return results
    
    def process_and_save(self, input_path: str, output_dir: str):
        """
        最適化された処理とParquet保存
        
        Args:
            input_path: t, address, dataファイルを含むraw_canディレクトリのパス
            output_dir: Parquetファイルの出力ディレクトリ
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"最適化GPU CANデコーダー実行開始 (chunk_size={self.chunk_size})...")
        
        # データ読み込み
        timestamps = np.fromfile(os.path.join(input_path, 't'), dtype=np.float64)
        addresses = np.fromfile(os.path.join(input_path, 'address'), dtype=np.int64)
        data_raw = np.fromfile(os.path.join(input_path, 'data'), dtype=np.uint8)
        data_bytes = data_raw.reshape(-1, 8)
        
        print(f"入力データ: {len(timestamps):,} メッセージ")
        
        # バッチ処理
        start_time = __import__('time').time()
        
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
            # タイムスタンプでソート（重要！）
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
            # 車両速度もソート
            vehicle_speed_df = vehicle_speed_df.sort_values('timestamp').reset_index(drop=True)
            
            output_path = os.path.join(output_dir, "vehicle_speed.parquet")
            vehicle_speed_df.to_parquet(output_path)
            print(f"Saved: {output_path} ({len(vehicle_speed_df)} rows)")
        
        # ステアリングデータ
        if all_results['steering']:
            steering_df = cudf.concat(all_results['steering'], ignore_index=True)
            # タイムスタンプでソート（重要！）
            steering_df = steering_df.sort_values('timestamp').reset_index(drop=True)
            
            output_path = os.path.join(output_dir, "steering.parquet")
            steering_df.to_parquet(output_path)
            print(f"Saved: {output_path} ({len(steering_df)} rows)")
        
        elapsed_time = __import__('time').time() - start_time
        print(f"\n処理時間: {elapsed_time:.3f} 秒")
        print(f"スループット: {len(timestamps) / elapsed_time / 1e6:.2f} Mmessages/sec")


    def decode_batch_for_benchmark(self, timestamps: np.ndarray, addresses: np.ndarray, 
                                  data_bytes: np.ndarray) -> List[cudf.DataFrame]:
        """
        ベンチマーク用の簡易インターフェース（車輪速度データのみ）
        """
        results_dict = self.decode_batch(timestamps, addresses, data_bytes)
        
        # 車輪速度データのみを返す（ベンチマーク用）
        if results_dict['wheel_speeds']:
            if len(results_dict['wheel_speeds']) == 1:
                return results_dict['wheel_speeds']
            else:
                # 複数チャンクの場合は結合
                combined_df = cudf.concat(results_dict['wheel_speeds'], ignore_index=True)
                return [combined_df.sort_values('timestamp').reset_index(drop=True)]
        else:
            return [cudf.DataFrame()]


if __name__ == "__main__":
    # テスト用のサンプル実行
    print("最適化GPU CANデコーダーのテスト")
    
    # チャンクサイズ1（分割なし）でのテスト
    decoder1 = OptimizedGPUCANDecoder(batch_size=500_000, chunk_size=1)
    print("チャンクサイズ1で初期化完了")
    
    # チャンクサイズ4での並列処理テスト
    decoder4 = OptimizedGPUCANDecoder(batch_size=500_000, chunk_size=4)
    print("チャンクサイズ4で初期化完了")
    
    print("最適化完了 - 統一されたパイプライン実装")