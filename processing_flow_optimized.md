# GPU処理フロー比較: 従来版 vs 最適化版

## 従来版GPU処理フロー

```
入力データ: timestamps[], addresses[], data_bytes[][]
    ↓
1. データ転送（3回）
   ├─ d_timestamps = cp.asarray(timestamps)
   ├─ d_addresses = cp.asarray(addresses)  
   └─ d_data_bytes = cp.asarray(data_bytes)
    ↓
2. 車輪速度フィルタリング
   ├─ wheel_mask = d_addresses == 170
   ├─ wheel_indices = cp.where(wheel_mask)[0]
   ├─ wheel_timestamps = d_timestamps[wheel_indices]
   └─ wheel_data = d_data_bytes[wheel_indices]
    ↓
3. メモリ確保
   └─ d_speeds = cp.zeros((n_wheels, 4), dtype=cp.float32)
    ↓
4. カーネル実行
   └─ decode_wheel_speeds[blocks, threads](wheel_data, d_speeds)
    ↓
5. DataFrame作成
   └─ cudf.DataFrame({timestamp, front_left, front_right, rear_left, rear_right})
    ↓
6. ステアリング処理（別処理）
   ├─ steering_mask = d_addresses == 37
   ├─ steering_indices = cp.where(steering_mask)[0]
   ├─ steering_data = d_data_bytes[steering_indices]
   └─ decode_steering[blocks, threads](steering_data, d_steering)
    ↓
7. 車両速度計算処理（別処理）
   └─ vehicle_speed = (front_left + front_right + rear_left + rear_right) / 4
    ↓
出力: 3つの独立したDataFrame
```

## 最適化版GPU処理フロー

```
入力データ: timestamps[], addresses[], data_bytes[][]
    ↓
チャンクサイズ = 1（デフォルト）の場合:
    ↓
1. データ転送（3回 - 従来版と同じ）
   ├─ d_timestamps = cp.asarray(timestamps)
   ├─ d_addresses = cp.asarray(addresses)  
   └─ d_data_bytes = cp.asarray(data_bytes)
    ↓
2. 車輪速度フィルタリング（従来版と同じ）
   ├─ wheel_mask = d_addresses == 170
   ├─ wheel_indices = cp.where(wheel_mask)[0]
   ├─ wheel_timestamps = d_timestamps[wheel_indices]
   └─ wheel_data = d_data_bytes[wheel_indices]
    ↓
3. メモリ確保（従来版と同じ）
   └─ d_speeds = cp.zeros((n_wheels, 4), dtype=cp.float32)
    ↓
4. カーネル実行（従来版と同じ）
   └─ decode_wheel_speeds[blocks, threads](wheel_data, d_speeds)
    ↓
5. DataFrame作成（従来版と同じ）
   └─ cudf.DataFrame({timestamp, front_left, front_right, rear_left, rear_right})
    ↓
出力: 1つのDataFrame（車輪速度のみ）
```

## チャンクサイズ > 1の場合の最適化版フロー

```
入力データ: timestamps[], addresses[], data_bytes[][]
    ↓
1. データ分割
   ├─ chunk_length = n_messages // chunk_size
   └─ 各チャンクに分割: [start_idx:end_idx]
    ↓
2. 並列処理（各チャンクで独立実行）
   ├─ Stream 0: chunk_0データ処理
   ├─ Stream 1: chunk_1データ処理
   ├─ Stream 2: chunk_2データ処理
   └─ Stream N: chunk_Nデータ処理
    ↓ (各ストリームで上記の1-5を実行)
3. 同期
   └─ 全ストリーム完了待機
    ↓
出力: chunk_size個のDataFrameリスト
```

## 性能比較結果

| 処理量 | 従来GPU | 最適化GPU | 最適化効果 |
|--------|---------|-----------|------------|
| 10K    | 0.392秒 | 0.146秒  | 2.7x高速化 |
| 50K    | 0.009秒 | 0.009秒  | ほぼ同等   |
| 100K   | 0.011秒 | 0.012秒  | わずか劣化 |
| 500K   | 0.013秒 | 0.060秒  | 4.6x劣化   |
| 1M     | 0.017秒 | 0.114秒  | 6.7x劣化   |

## 問題点の分析

**最適化版が遅い理由:**
1. **統一カーネルのオーバーヘッド**: 複雑なカーネルロジック
2. **不要なメモリ確保**: estimated_output_size の過大見積もり
3. **追加の型変換**: unified processing での変換コスト
4. **Stream処理のオーバーヘッド**: 小さなデータでの並列化ペナルティ

**従来版が速い理由:**
1. **特化されたフィルタリング**: 車輪速度のみの効率的処理
2. **最適化されたメモリパターン**: GPUメモリアクセス最適化
3. **シンプルなカーネル**: decode_wheel_speeds の単純性
4. **直接的なDataFrame作成**: 中間変換なし

## 最適化の方向性

**正しい最適化アプローチ:**
1. 従来版のロジックを維持
2. chunk_size=1の場合は従来版と同等の処理
3. chunk_size>1の場合のみ並列ストリーム処理
4. 統一カーネルではなく、特化されたカーネルを維持