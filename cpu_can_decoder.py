"""
CPU-based CAN data decoder for comparison with GPU implementation
Uses the same OpenPilot DBC definitions for Toyota RAV4
"""

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import time
from typing import Dict


class CPUCANDecoder:
    def __init__(self, batch_size: int = 100_000):
        """
        Initialize CPU CAN decoder
        
        Args:
            batch_size: Number of messages to process in each batch
        """
        self.batch_size = batch_size
        
    def decode_wheel_speeds(self, timestamps, addresses, data_bytes):
        """
        Decode wheel speed messages (address 170)
        OpenPilot DBC: (0.01,-67.67) "kph" for Toyota RAV4
        """
        # Find wheel speed messages
        wheel_mask = addresses == 170
        wheel_indices = np.where(wheel_mask)[0]
        
        if len(wheel_indices) == 0:
            return None
            
        # Pre-allocate output arrays
        n_messages = len(wheel_indices)
        out_timestamps = timestamps[wheel_indices]
        out_front_left = np.zeros(n_messages, dtype=np.float32)
        out_front_right = np.zeros(n_messages, dtype=np.float32)
        out_rear_left = np.zeros(n_messages, dtype=np.float32)
        out_rear_right = np.zeros(n_messages, dtype=np.float32)
        
        # Decode each message
        for i, idx in enumerate(wheel_indices):
            # Front left (bytes 0-1)
            raw_value = (int(data_bytes[idx, 0]) << 8) | int(data_bytes[idx, 1])
            out_front_left[i] = (raw_value * 0.01 - 67.67) / 3.6  # Convert to m/s
            
            # Front right (bytes 2-3)
            raw_value = (int(data_bytes[idx, 2]) << 8) | int(data_bytes[idx, 3])
            out_front_right[i] = (raw_value * 0.01 - 67.67) / 3.6
            
            # Rear left (bytes 4-5)
            raw_value = (int(data_bytes[idx, 4]) << 8) | int(data_bytes[idx, 5])
            out_rear_left[i] = (raw_value * 0.01 - 67.67) / 3.6
            
            # Rear right (bytes 6-7)
            raw_value = (int(data_bytes[idx, 6]) << 8) | int(data_bytes[idx, 7])
            out_rear_right[i] = (raw_value * 0.01 - 67.67) / 3.6
            
        return {
            'timestamp': out_timestamps,
            'front_left': out_front_left,
            'front_right': out_front_right,
            'rear_left': out_rear_left,
            'rear_right': out_rear_right
        }
    
    def decode_steering_angle(self, timestamps, addresses, data_bytes):
        """
        Decode steering angle messages (address 37)
        """
        # Find steering messages
        steering_mask = addresses == 37
        steering_indices = np.where(steering_mask)[0]
        
        if len(steering_indices) == 0:
            return None
            
        # Pre-allocate output arrays
        n_messages = len(steering_indices)
        out_timestamps = timestamps[steering_indices]
        out_angle = np.zeros(n_messages, dtype=np.float32)
        
        # Decode each message
        for i, idx in enumerate(steering_indices):
            byte01 = data_bytes[idx, 1]
            byte4 = data_bytes[idx, 4]
            
            # Apply observed decoding logic
            if byte01 == 0x00 and byte4 == 0xC0:
                out_angle[i] = 0.0
            elif byte01 == 0x54 and byte4 == 0xBE:
                out_angle[i] = -1.5
            elif byte01 == 0x97 and byte4 == 0xBD:
                out_angle[i] = -2.5
            elif byte01 == 0xD9 and byte4 == 0xBC:
                out_angle[i] = -3.5
            elif byte01 == 0x1C and byte4 == 0xC2:
                out_angle[i] = 1.5
            elif byte01 == 0x5E and byte4 == 0xC3:
                out_angle[i] = 2.5
            elif byte01 == 0xA1 and byte4 == 0xC4:
                out_angle[i] = 3.5
            else:
                out_angle[i] = 0.0
                
        return {
            'timestamp': out_timestamps,
            'angle': out_angle
        }
    
    def decode_batch(self, timestamps: np.ndarray, addresses: np.ndarray, 
                    data_bytes: np.ndarray) -> Dict[str, pd.DataFrame]:
        """
        Decode a batch of CAN messages on CPU
        
        Returns:
            Dictionary with decoded DataFrames for each message type
        """
        results = {}
        
        # Decode wheel speeds
        wheel_data = self.decode_wheel_speeds(timestamps, addresses, data_bytes)
        if wheel_data is not None:
            wheel_df = pd.DataFrame(wheel_data)
            # Sort by timestamp
            wheel_df = wheel_df.sort_values('timestamp').reset_index(drop=True)
            results['wheel_speeds'] = wheel_df
            
            # Calculate vehicle speed (average of 4 wheels)
            vehicle_speed_df = pd.DataFrame({
                'timestamp': wheel_df['timestamp'],
                'speed': (wheel_df['front_left'] + wheel_df['front_right'] + 
                         wheel_df['rear_left'] + wheel_df['rear_right']) / 4.0
            })
            results['vehicle_speed'] = vehicle_speed_df
        
        # Decode steering angle
        steering_data = self.decode_steering_angle(timestamps, addresses, data_bytes)
        if steering_data is not None:
            steering_df = pd.DataFrame(steering_data)
            # Sort by timestamp
            steering_df = steering_df.sort_values('timestamp').reset_index(drop=True)
            results['steering'] = steering_df
            
        return results
    
    def process_and_save(self, input_path: str, output_dir: str):
        """
        Process CAN data from raw_can directory and save as Parquet
        
        Args:
            input_path: Path to raw_can directory containing t, address, data files
            output_dir: Output directory for Parquet files
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("CPU CANデコーダー実行開始...")
        
        # Load data from individual files (comma2k19 format)
        timestamps = np.fromfile(os.path.join(input_path, 't'), dtype=np.float64)
        addresses = np.fromfile(os.path.join(input_path, 'address'), dtype=np.int64)
        data_raw = np.fromfile(os.path.join(input_path, 'data'), dtype=np.uint8)
        data_bytes = data_raw.reshape(-1, 8)  # Each CAN message has 8 bytes
        
        print(f"入力データ: {len(timestamps):,} メッセージ")
        
        # Process in batches
        all_results = {
            'wheel_speeds': [],
            'vehicle_speed': [],
            'steering': []
        }
        
        start_time = time.time()
        
        for i in range(0, len(timestamps), self.batch_size):
            end_idx = min(i + self.batch_size, len(timestamps))
            print(f"  処理中: {i:,} - {end_idx:,}")
            
            batch_results = self.decode_batch(
                timestamps[i:end_idx],
                addresses[i:end_idx],
                data_bytes[i:end_idx]
            )
            
            for name, df in batch_results.items():
                if df is not None and len(df) > 0:
                    all_results[name].append(df)
        
        # Combine all batches and save
        for name, df_list in all_results.items():
            if df_list:
                combined_df = pd.concat(df_list, ignore_index=True)
                # Final sort by timestamp
                combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
                
                output_path = os.path.join(output_dir, f"{name}_cpu.parquet")
                combined_df.to_parquet(output_path)
                print(f"Saved: {output_path} ({len(combined_df)} rows)")
        
        elapsed_time = time.time() - start_time
        print(f"\n処理時間: {elapsed_time:.3f} 秒")
        print(f"スループット: {len(timestamps) / elapsed_time / 1e6:.2f} Mmessages/sec")