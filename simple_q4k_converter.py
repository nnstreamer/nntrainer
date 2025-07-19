#!/usr/bin/env python3
"""
Simple Q4_K Converter: FP32 -> Q4_K Binary
==========================================

PyTorch 모델의 FP32 가중치를 Q4_K 형식으로 변환하여 바이너리 파일로 저장

Usage:
    python simple_q4k_converter.py model.pth output.bin
"""

import numpy as np
import torch
import struct
import sys
from pathlib import Path

# Q4_K 상수
QK_K = 256  # 블록 크기 (8개 서브블록 × 32개 요소)

def quantize_q4k_block(x: np.ndarray) -> bytes:
    """
    256개 요소를 Q4_K 형식으로 양자화
    
    구조:
    - 8개 서브블록 (각 32개 요소)
    - 각 서브블록: min, scale 계산 후 4비트 양자화
    - 출력: [d_scale(2B), d_min(2B), scales(8B), mins(8B), quants(128B)] = 148 bytes
    """
    assert len(x) == QK_K, f"블록 크기는 {QK_K}이어야 함"
    
    # 8개 서브블록으로 분할 (각 32개)
    sub_blocks = x.reshape(8, 32)
    
    # 각 서브블록의 min/max 계산
    scales = np.zeros(8, dtype=np.float32)
    mins = np.zeros(8, dtype=np.float32)
    quants = np.zeros(QK_K // 2, dtype=np.uint8)  # 4비트 패킹
    
    for i in range(8):
        block = sub_blocks[i]
        min_val = float(np.min(block))
        max_val = float(np.max(block))
        
        if max_val == min_val:
            scale = 1.0
        else:
            scale = (max_val - min_val) / 15.0  # 4비트 범위 0-15
        
        scales[i] = scale
        mins[i] = min_val
        
        # 4비트 양자화
        for j in range(32):
            if scale > 0:
                q = int(np.round((block[j] - min_val) / scale))
                q = np.clip(q, 0, 15)
            else:
                q = 0
            
            # 2개씩 패킹 (4비트 + 4비트 = 1바이트)
            idx = i * 32 + j
            if j % 2 == 0:
                quants[idx // 2] = q
            else:
                quants[idx // 2] |= (q << 4)
    
    # 전역 스케일 계산
    max_scale = np.max(scales) if np.max(scales) > 0 else 1.0
    d_scale = max_scale
    
    min_min = np.min(mins)
    max_min = np.max(mins)
    d_min = min_min
    
    # 스케일 정규화 (0-255 범위로)
    scales_norm = np.zeros(8, dtype=np.uint8)
    mins_norm = np.zeros(8, dtype=np.uint8)
    
    for i in range(8):
        if d_scale > 0:
            scales_norm[i] = int(np.clip(scales[i] / d_scale * 255, 0, 255))
        
        if max_min != min_min:
            mins_norm[i] = int(np.clip((mins[i] - min_min) / (max_min - min_min) * 255, 0, 255))
    
    # 바이너리 패킹
    block_data = bytearray()
    block_data.extend(struct.pack('<f', d_scale))      # 4 bytes
    block_data.extend(struct.pack('<f', d_min))        # 4 bytes  
    block_data.extend(scales_norm.tobytes())           # 8 bytes
    block_data.extend(mins_norm.tobytes())             # 8 bytes
    block_data.extend(quants.tobytes())                # 128 bytes
    
    return bytes(block_data)  # 총 152 bytes per block

def convert_tensor_q4k(tensor: torch.Tensor) -> bytes:
    """텐서를 Q4_K 바이너리로 변환"""
    # CPU로 이동 및 FP32 변환
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    arr = tensor.numpy().astype(np.float32).flatten()
    
    # QK_K(256) 배수로 패딩
    n_elements = len(arr)
    n_blocks = (n_elements + QK_K - 1) // QK_K
    padded_size = n_blocks * QK_K
    
    if padded_size > n_elements:
        padded_arr = np.zeros(padded_size, dtype=np.float32)
        padded_arr[:n_elements] = arr
    else:
        padded_arr = arr
    
    # 블록별로 양자화
    result = bytearray()
    blocks = padded_arr.reshape(n_blocks, QK_K)
    
    for block in blocks:
        q_block = quantize_q4k_block(block)
        result.extend(q_block)
    
    return bytes(result)

def convert_model_q4k(model_path: str, output_path: str):
    """모델을 Q4_K 바이너리로 변환"""
    print(f"Loading model: {model_path}")
    
    # 모델 로딩
    if model_path.endswith('.safetensors'):
        try:
            from safetensors.torch import load_file
            tensors = load_file(model_path)
        except ImportError:
            print("Error: safetensors not installed. Run: pip install safetensors")
            return
    else:
        checkpoint = torch.load(model_path, map_location='cpu')
        if isinstance(checkpoint, dict):
            tensors = checkpoint.get('state_dict', checkpoint)
        else:
            tensors = checkpoint.state_dict() if hasattr(checkpoint, 'state_dict') else checkpoint
    
    # Float 텐서만 필터링
    float_tensors = {k: v for k, v in tensors.items() 
                    if v.dtype in [torch.float32, torch.float16, torch.bfloat16]}
    
    print(f"Found {len(float_tensors)} float tensors")
    
    # 텐서별로 변환 및 저장
    with open(output_path, 'wb') as f:
        # 헤더: 텐서 개수
        f.write(struct.pack('<I', len(float_tensors)))
        
        for name, tensor in float_tensors.items():
            print(f"Converting: {name} {list(tensor.shape)}")
            
            # 텐서 정보 저장 (이름 길이, 이름, 원본 shape, 원소 개수)
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('<I', len(name_bytes)))  # 이름 길이
            f.write(name_bytes)                          # 이름
            f.write(struct.pack('<I', len(tensor.shape)))  # 차원 수
            for dim in tensor.shape:
                f.write(struct.pack('<I', dim))          # 각 차원 크기
            
            # Q4_K 변환 데이터
            q4k_data = convert_tensor_q4k(tensor)
            f.write(struct.pack('<I', len(q4k_data)))    # 데이터 크기
            f.write(q4k_data)                            # Q4_K 데이터
    
    # 결과 출력
    original_size = sum(t.numel() * 4 for t in float_tensors.values())  # FP32 기준
    compressed_size = Path(output_path).stat().st_size
    ratio = original_size / compressed_size
    
    print(f"\n✅ Conversion completed!")
    print(f"Original size: {original_size / 1024**2:.1f} MB")
    print(f"Q4_K size: {compressed_size / 1024**2:.1f} MB") 
    print(f"Compression ratio: {ratio:.2f}x")
    print(f"Output: {output_path}")

def load_q4k_model(file_path: str) -> dict:
    """Q4_K 바이너리 파일 로딩 (참고용)"""
    tensors = {}
    
    with open(file_path, 'rb') as f:
        # 텐서 개수 읽기
        n_tensors = struct.unpack('<I', f.read(4))[0]
        
        for _ in range(n_tensors):
            # 텐서 정보 읽기
            name_len = struct.unpack('<I', f.read(4))[0]
            name = f.read(name_len).decode('utf-8')
            
            n_dims = struct.unpack('<I', f.read(4))[0]
            shape = []
            for _ in range(n_dims):
                shape.append(struct.unpack('<I', f.read(4))[0])
            
            # Q4_K 데이터 읽기
            data_len = struct.unpack('<I', f.read(4))[0]
            q4k_data = f.read(data_len)
            
            tensors[name] = {
                'shape': tuple(shape),
                'q4k_data': q4k_data
            }
    
    return tensors

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python simple_q4k_converter.py <model.pth> <output.bin>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    output_path = sys.argv[2]
    
    if not Path(model_path).exists():
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)
    
    convert_model_q4k(model_path, output_path)