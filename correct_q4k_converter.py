import numpy as np
import torch
import struct
from typing import Union, List

# Q4_K 상수들 (llama.cpp와 동일)
QK_K = 256
K_SCALE_SIZE = 12

def quantize_row_q4_k_ref(x: np.ndarray) -> np.ndarray:
    """
    llama.cpp의 quantize_row_q4_k_ref 함수를 Python으로 구현
    Reference: https://github.com/ggerganov/llama.cpp/blob/master/ggml-quants.c
    """
    assert len(x) == QK_K, f"Input size must be {QK_K}, got {len(x)}"
    
    # 8개 서브블록으로 나누기 (각 32개 요소)
    L = np.zeros(QK_K, dtype=np.uint8)
    scales = np.zeros(8, dtype=np.float32)
    mins = np.zeros(8, dtype=np.float32)
    
    # 각 서브블록별 스케일과 최소값 계산
    for i in range(8):
        start_idx = 32 * i
        end_idx = 32 * (i + 1)
        sub_block = x[start_idx:end_idx]
        
        # 서브블록의 min, max
        min_val = sub_block.min()
        max_val = sub_block.max()
        
        # 스케일 계산
        if max_val > min_val:
            scale = (max_val - min_val) / 15.0
            scales[i] = scale
            mins[i] = min_val
            
            # 4비트로 양자화
            for j in range(32):
                val = int(round((sub_block[j] - min_val) / scale))
                L[start_idx + j] = max(0, min(15, val))
        else:
            scales[i] = 1.0
            mins[i] = min_val
            L[start_idx:end_idx] = 0
    
    # 슈퍼블록 스케일 계산
    max_scale = max(scales) if max(scales) > 0 else 1.0
    max_min = max(np.abs(mins)) if len(mins) > 0 else 0.0
    
    # 스케일과 최소값을 6비트로 양자화
    d = max_scale / 63.0 if max_scale > 0 else 1.0
    dmin = max_min / 63.0 if max_min > 0 else 1.0
    
    # 6비트 양자화된 스케일과 최소값
    scales_q = np.zeros(8, dtype=np.uint8)
    mins_q = np.zeros(8, dtype=np.uint8)
    
    for i in range(8):
        scales_q[i] = int(round(scales[i] / d)) if d > 0 else 0
        mins_q[i] = int(round(mins[i] / dmin)) if dmin > 0 else 0
        scales_q[i] = max(0, min(63, scales_q[i]))
        mins_q[i] = max(0, min(63, mins_q[i]))
    
    return d, dmin, scales_q, mins_q, L

def pack_q4k_block(d: float, dmin: float, scales_q: np.ndarray, mins_q: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Q4_K 블록 데이터를 llama.cpp 형식에 맞게 패킹"""
    
    # 스케일과 최소값을 패킹 (12바이트)
    scales_and_mins = np.zeros(K_SCALE_SIZE, dtype=np.uint8)
    
    # 스케일과 최소값을 4비트씩 패킹
    for i in range(8):
        if i % 2 == 0:
            scales_and_mins[i // 2] = scales_q[i] & 0x3F
            scales_and_mins[i // 2 + 8] = mins_q[i] & 0x3F
        else:
            scales_and_mins[i // 2] |= (scales_q[i] & 0x3F) << 6
            scales_and_mins[i // 2 + 8] |= (mins_q[i] & 0x3F) << 6
    
    # 4비트 값들을 2개씩 1바이트로 패킹
    qs = np.zeros(QK_K // 2, dtype=np.uint8)
    for i in range(0, QK_K, 2):
        qs[i // 2] = (L[i] & 0xF) | ((L[i + 1] & 0xF) << 4)
    
    # 블록 구조체 구성: [d(2bytes) + dmin(2bytes) + scales_and_mins(12bytes) + qs(128bytes)]
    # 총 144바이트
    block_data = bytearray()
    
    # d와 dmin을 half(float16)로 저장
    block_data.extend(struct.pack('<e', d))      # 2 bytes
    block_data.extend(struct.pack('<e', dmin))   # 2 bytes
    block_data.extend(scales_and_mins.tobytes()) # 12 bytes
    block_data.extend(qs.tobytes())              # 128 bytes
    
    return np.frombuffer(block_data, dtype=np.uint8)

def convert_to_q4k(tensor: torch.Tensor) -> np.ndarray:
    """PyTorch tensor를 Q4_K 형식으로 변환"""
    
    # CPU로 이동하고 float32로 변환
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    if tensor.dtype != torch.float32:
        tensor = tensor.float()
    
    # 2D로 reshape
    original_shape = tensor.shape
    if len(original_shape) > 2:
        tensor = tensor.reshape(-1, original_shape[-1])
    elif len(original_shape) == 1:
        tensor = tensor.unsqueeze(0)
    
    arr = tensor.numpy()
    n_rows, n_cols = arr.shape
    
    # QK_K로 패딩
    if n_cols % QK_K != 0:
        pad_size = QK_K - (n_cols % QK_K)
        arr = np.pad(arr, ((0, 0), (0, pad_size)), mode='constant')
        n_cols = arr.shape[1]
    
    n_blocks_per_row = n_cols // QK_K
    total_blocks = n_rows * n_blocks_per_row
    
    # 각 블록을 변환
    quantized_blocks = []
    
    for row in range(n_rows):
        for block_idx in range(n_blocks_per_row):
            start_col = block_idx * QK_K
            end_col = start_col + QK_K
            block_data = arr[row, start_col:end_col]
            
            # Q4_K 양자화
            d, dmin, scales_q, mins_q, L = quantize_row_q4_k_ref(block_data)
            
            # 블록 패킹
            packed_block = pack_q4k_block(d, dmin, scales_q, mins_q, L)
            quantized_blocks.append(packed_block)
    
    result = np.array(quantized_blocks)
    return result, original_shape, (n_rows, n_blocks_per_row)

def save_q4k_weights(weights: Union[torch.nn.Module, List[torch.Tensor]], output_path: str):
    """PyTorch 모델의 가중치를 Q4_K 형식으로 저장"""
    
    if isinstance(weights, torch.nn.Module):
        weight_dict = weights.state_dict()
        weight_list = [(name, tensor) for name, tensor in weight_dict.items()]
    else:
        weight_list = [(f"tensor_{i}", tensor) for i, tensor in enumerate(weights)]
    
    quantized_data = []
    
    print(f"Converting {len(weight_list)} tensors to Q4_K format...")
    
    for i, (name, tensor) in enumerate(weight_list):
        if tensor.dtype in [torch.float32, torch.float16]:
            print(f"  [{i+1}/{len(weight_list)}] Converting {name}: {tensor.shape}")
            
            q4k_blocks, original_shape, block_info = convert_to_q4k(tensor)
            
            quantized_data.append({
                'name': name,
                'original_shape': original_shape,
                'block_info': block_info,  # (n_rows, n_blocks_per_row)
                'data': q4k_blocks
            })
            
            # 압축률 계산
            original_size = tensor.numel() * 4  # float32 기준
            compressed_size = len(q4k_blocks) * len(q4k_blocks[0])
            compression_ratio = original_size / compressed_size
            print(f"    Original: {original_size:,} bytes, Compressed: {compressed_size:,} bytes")
            print(f"    Compression ratio: {compression_ratio:.2f}x")
        else:
            print(f"  Skipping {name}: unsupported dtype {tensor.dtype}")
    
    # 바이너리 파일로 저장
    with open(output_path, 'wb') as f:
        # 헤더: 매직 넘버 + 버전 + 텐서 개수
        f.write(b'Q4KF')  # 매직 넘버
        f.write(struct.pack('<I', 1))  # 버전
        f.write(struct.pack('<I', len(quantized_data)))  # 텐서 개수
        
        for layer in quantized_data:
            # 텐서 이름
            name_bytes = layer['name'].encode('utf-8')
            f.write(struct.pack('<I', len(name_bytes)))
            f.write(name_bytes)
            
            # 원본 shape
            f.write(struct.pack('<I', len(layer['original_shape'])))
            for dim in layer['original_shape']:
                f.write(struct.pack('<I', dim))
            
            # 블록 정보
            f.write(struct.pack('<II', *layer['block_info']))
            
            # 양자화된 데이터
            data = layer['data']
            f.write(struct.pack('<I', len(data)))  # 블록 개수
            f.write(struct.pack('<I', len(data[0]))) # 블록 크기
            f.write(data.tobytes())
    
    print(f"✅ Q4_K weights saved to: {output_path}")

def load_q4k_weights(file_path: str) -> List[dict]:
    """Q4_K 형식으로 저장된 가중치를 로드"""
    
    with open(file_path, 'rb') as f:
        # 헤더 읽기
        magic = f.read(4)
        if magic != b'Q4KF':
            raise ValueError(f"Invalid magic number: {magic}")
        
        version = struct.unpack('<I', f.read(4))[0]
        if version != 1:
            raise ValueError(f"Unsupported version: {version}")
        
        num_tensors = struct.unpack('<I', f.read(4))[0]
        
        loaded_weights = []
        
        for _ in range(num_tensors):
            # 텐서 이름
            name_len = struct.unpack('<I', f.read(4))[0]
            name = f.read(name_len).decode('utf-8')
            
            # 원본 shape
            shape_len = struct.unpack('<I', f.read(4))[0]
            shape = []
            for _ in range(shape_len):
                shape.append(struct.unpack('<I', f.read(4))[0])
            
            # 블록 정보
            n_rows, n_blocks_per_row = struct.unpack('<II', f.read(8))
            
            # 양자화된 데이터
            num_blocks = struct.unpack('<I', f.read(4))[0]
            block_size = struct.unpack('<I', f.read(4))[0]
            
            data_bytes = f.read(num_blocks * block_size)
            data = np.frombuffer(data_bytes, dtype=np.uint8).reshape(num_blocks, block_size)
            
            loaded_weights.append({
                'name': name,
                'original_shape': tuple(shape),
                'block_info': (n_rows, n_blocks_per_row),
                'data': data
            })
    
    return loaded_weights

# 사용 예제
if __name__ == "__main__":
    # 예제 텐서 생성
    test_tensor = torch.randn(1024, 2048)  # 예: 2MB 가중치
    
    print("=== Q4_K Conversion Test ===")
    print(f"Original tensor shape: {test_tensor.shape}")
    print(f"Original tensor size: {test_tensor.numel() * 4:,} bytes")
    
    # Q4_K로 변환
    q4k_blocks, original_shape, block_info = convert_to_q4k(test_tensor)
    
    print(f"Quantized blocks shape: {q4k_blocks.shape}")
    print(f"Block size: {q4k_blocks.shape[1]} bytes")
    print(f"Compression ratio: {(test_tensor.numel() * 4) / (q4k_blocks.shape[0] * q4k_blocks.shape[1]):.2f}x")
    
    # 파일로 저장
    save_q4k_weights([test_tensor], "test_q4k_weights.bin")
    
    # 파일에서 로드
    loaded = load_q4k_weights("test_q4k_weights.bin")
    print(f"\n✅ Successfully loaded {len(loaded)} tensors")
    print(f"First tensor: {loaded[0]['name']}, shape: {loaded[0]['original_shape']}")