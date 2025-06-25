# GPU vs CPU処理フロー詳細分析

## 概要

CANデータのGPUスループットが期待通りに向上しない原因を分析するため、GPU処理とCPU処理の詳細フローを比較します。

## GPU処理フロー（GPUCANDecoder）

### 1. 初期化フェーズ
```python
def __init__(self, batch_size: int = 1_000_000):
    self.batch_size = batch_size
    self.pool = rmm.mr.PoolMemoryResource(
        rmm.mr.CudaMemoryResource(),
        initial_pool_size=2**30,  # 1GB
        maximum_pool_size=2**31   # 2GB
    )
    rmm.mr.set_current_device_resource(self.pool)
```

**潜在的な問題点:**
- RMM メモリプールのサイズが制限されている（1-2GB）
- メモリプールの初期化オーバーヘッドが存在

### 2. メインデコード処理（decode_batch）

#### 2.1 CPU→GPU データ転送
```python
d_timestamps = cp.asarray(timestamps)      # CPU→GPU転送 1
d_addresses = cp.asarray(addresses)        # CPU→GPU転送 2  
d_data_bytes = cp.asarray(data_bytes)      # CPU→GPU転送 3
```

**問題点:**
- **3回の独立したCPU→GPU転送** - PCIeバンド幅の非効率利用
- 各転送が同期的に実行される可能性
- メモリコピーオーバーヘッドが大きい

#### 2.2 GPU メモリ事前割り当て
```python
max_wheel_msgs = int(n_messages * 0.05)     # 5%と仮定
max_steering_msgs = int(n_messages * 0.05)

# 車輪速度出力バッファ（5個の配列）
d_wheel_timestamps = cp.zeros(max_wheel_msgs, dtype=cp.float64)
d_front_left = cp.zeros(max_wheel_msgs, dtype=cp.float32)
d_front_right = cp.zeros(max_wheel_msgs, dtype=cp.float32)
d_rear_left = cp.zeros(max_wheel_msgs, dtype=cp.float32)
d_rear_right = cp.zeros(max_wheel_msgs, dtype=cp.float32)
d_wheel_count = cp.zeros(1, dtype=cp.int32)

# ステアリング出力バッファ（3個の配列）
d_steering_timestamps = cp.zeros(max_steering_msgs, dtype=cp.float64)
d_steering_angle = cp.zeros(max_steering_msgs, dtype=cp.float32)
d_steering_count = cp.zeros(1, dtype=cp.int32)
```

**問題点:**
- **8個の独立したメモリ割り当て** - GPU メモリフラグメンテーション
- 事前割り当てサイズが実際のデータサイズと大きく異なる可能性

#### 2.3 CUDAカーネル実行
```python
threads_per_block = 256
blocks_per_grid = (n_messages + threads_per_block - 1) // threads_per_block

# カーネル1: 車輪速度デコード
decode_wheel_speeds_kernel[blocks_per_grid, threads_per_block](
    d_timestamps, d_addresses, d_data_bytes,
    d_wheel_timestamps, d_front_left, d_front_right,
    d_rear_left, d_rear_right, d_wheel_count
)

# カーネル2: ステアリング角度デコード  
decode_steering_angle_kernel[blocks_per_grid, threads_per_block](
    d_timestamps, d_addresses, d_data_bytes,
    d_steering_timestamps, d_steering_angle, d_steering_count
)
```

**問題点:**
- **2個の独立したカーネル起動** - カーネル起動オーバーヘッド
- 同じ入力データを2回読み込み（メモリ帯域幅の無駄）
- アトミック操作（`cuda.atomic.add`）によるメモリ競合

#### 2.4 カーネル内処理分析

**車輪速度カーネル（decode_wheel_speeds_kernel）:**
```python
@cuda.jit
def decode_wheel_speeds_kernel(timestamps, addresses, data_bytes, ...):
    idx = cuda.grid(1)
    if idx < addresses.shape[0]:
        if addresses[idx] == 170:  # 条件分岐 - ワープ発散
            out_idx = cuda.atomic.add(out_count, 0, 1)  # アトミック操作
            # 4つの車輪を順次処理
            for wheel in range(4):
                raw_value = (data_bytes[idx, wheel*2] << 8) | data_bytes[idx, wheel*2+1]
                wheel_speed = (raw_value * 0.01 - 67.67) / 3.6
```

**重大な問題点:**
1. **ワープ発散**: アドレス170のメッセージは約3.7%しかないため、ワープ内の大部分のスレッドが処理を行わない
2. **アトミック競合**: 複数スレッドが同時に`out_count`にアクセス
3. **メモリアクセスパターン**: 非連続メモリアクセス

#### 2.5 GPU→CPU データ転送とDataFrame作成
```python
# 同期待機
cp.cuda.Stream.null.synchronize()

# カウント取得（GPU→CPU転送）
wheel_count = int(d_wheel_count[0])
steering_count = int(d_steering_count[0])

# cuDF DataFrame作成（GPU上）
wheel_df = cudf.DataFrame({
    'timestamp': d_wheel_timestamps[:wheel_count],
    'front_left': d_front_left[:wheel_count],
    # ...
})
wheel_df = wheel_df.sort_values('timestamp').reset_index(drop=True)  # ソート処理
```

**問題点:**
- 明示的な同期待機によるパイプライン停止
- GPU上でのソート処理は効率的だが、小さなデータサイズでは恩恵が少ない

## CPU処理フロー（CPUCANDecoder）

### 1. メインデコード処理（decode_batch）

#### 1.1 車輪速度デコード
```python
def decode_wheel_speeds(self, timestamps, addresses, data_bytes):
    # 効率的なマスク操作
    wheel_mask = addresses == 170           # ベクトル化された比較
    wheel_indices = np.where(wheel_mask)[0] # 一度でインデックス取得
    
    if len(wheel_indices) == 0:
        return None
        
    # 事前割り当て（正確なサイズ）
    n_messages = len(wheel_indices)
    out_timestamps = timestamps[wheel_indices]  # 効率的なインデクシング
    
    # ループで各メッセージを処理
    for i, idx in enumerate(wheel_indices):
        # 4つの車輪を順次処理（GPUと同様）
        for wheel in range(4):
            raw_value = (int(data_bytes[idx, wheel*2]) << 8) | int(data_bytes[idx, wheel*2+1])
            out_wheel[wheel][i] = (raw_value * 0.01 - 67.67) / 3.6
```

**効率的な点:**
1. **マスク操作**: NumPyのベクトル化された演算を利用
2. **正確な事前割り当て**: 無駄なメモリ使用なし
3. **効率的なインデクシング**: 必要なデータのみアクセス

#### 1.2 DataFrame作成とソート
```python
wheel_df = pd.DataFrame(wheel_data)  # 効率的なDataFrame作成
wheel_df = wheel_df.sort_values('timestamp').reset_index(drop=True)  # 最適化されたソート
```

## 処理フロー比較図

```
GPU処理フロー:
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌──────────────────┐
│   CPU Memory    │    │   GPU Memory     │    │  CUDA Kernels   │    │   cuDF/GPU       │
│                 │    │                  │    │                 │    │                  │
│ timestamps  ────┼──→ │ d_timestamps     │    │ decode_wheel_   │    │ DataFrame作成     │
│ addresses   ────┼──→ │ d_addresses      │ ──→│ speeds_kernel   │ ──→│ ソート処理       │
│ data_bytes  ────┼──→ │ d_data_bytes     │    │                 │    │ 同期待機         │
│                 │    │                  │    │ decode_steering_│    │                  │
│                 │    │ 8個の出力バッファ  │    │ angle_kernel    │    │                  │
└─────────────────┘    └──────────────────┘    └─────────────────┘    └──────────────────┘
     3回転送                8回メモリ確保           2回カーネル起動           DataFrame処理
   (同期実行)              (フラグメンテーション)    (起動オーバーヘッド)      (小データで非効率)

CPU処理フロー:
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Input Data    │    │   NumPy/pandas   │    │   Output        │
│                 │    │                  │    │                 │
│ timestamps  ────┼──→ │ マスク操作        │ ──→│ DataFrame作成    │
│ addresses   ────┼──→ │ インデクシング     │    │ 効率的ソート     │
│ data_bytes  ────┼──→ │ ループ処理        │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
    メモリ内処理            最適化された演算         高速処理
   (コピーなし)           (SIMD/ベクトル化)      (小データ特化)
```

## パフォーマンス問題の根本原因

### 1. データサイズとGPU効率のミスマッチ
- **車輪速度メッセージ**: 全体の約3.7%（1M件中37K件）
- **ステアリングメッセージ**: 全体の約3.7%（1M件中37K件）
- GPUの並列性を活かすには小さすぎるデータサイズ

### 2. メモリ転送オーバーヘッド
- **CPU→GPU転送**: 3回の独立転送
- **PCIe帯域幅**: 転送サイズが小さいため効率が悪い
- **メモリコピー時間**: 実際の計算時間より大きい可能性

### 3. GPU アーキテクチャの非効率利用
- **ワープ発散**: 条件分岐により32スレッド中29スレッドが待機
- **アトミック競合**: 複数スレッドが同じメモリ位置にアクセス
- **カーネル起動コスト**: 少ない計算量に対して起動コストが大きい

### 4. 最適化されたCPU実装
- **NumPy最適化**: SIMD命令とベクトル化
- **効率的なメモリアクセス**: キャッシュ効率の良いアクセスパターン
- **条件分岐の最小化**: マスク操作による効率的なフィルタリング

## 改善提案

### 1. データ転送の最適化
```python
# 改善案: 単一転送での統合
combined_data = np.column_stack([timestamps, addresses, data_bytes.flatten()])
d_combined = cp.asarray(combined_data)
```

### 2. カーネル統合
```python
# 改善案: 単一カーネルで全メッセージタイプを処理
@cuda.jit
def decode_all_messages_kernel(combined_data, output_buffer):
    # 全メッセージタイプを1つのカーネルで処理
```

### 3. ワープ効率の改善
```python
# 改善案: 事前フィルタリング
wheel_indices = cp.where(d_addresses == 170)[0]
if len(wheel_indices) > 0:
    decode_wheel_speeds_optimized[blocks, threads](wheel_indices, ...)
```

### 4. ストリーム並列化
```python
# 改善案: 複数ストリームでの並列処理
stream1 = cp.cuda.Stream()
stream2 = cp.cuda.Stream()
# 非同期処理
```

## 結論

現在のGPU実装は小規模データ（37K件程度）に対してオーバーエンジニアリングされており、以下の理由でCPU実装より遅くなっています：

1. **GPU転送コスト > GPU計算利益**
2. **ワープ発散による並列性の損失**
3. **アトミック操作による競合**
4. **高度に最適化されたCPU実装（NumPy/pandas）**

GPUが有効になるのは、より大規模なデータセット（1M件以上の関連メッセージ）か、より複雑な計算処理の場合と考えられます。