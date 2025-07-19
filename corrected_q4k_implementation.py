import numpy as np
import torch
from typing import Union, List

def convert_to_q4k_corrected(tensor: torch.Tensor) -> np.ndarray:
    """PyTorch tensor를 올바른 Q4_K 형식으로 변환합니다."""
    # CPU로 이동하고 numpy array로 변환
    if tensor.is_cuda:
        tensor = tensor.cpu()
    arr = tensor.numpy().astype(np.float32)
    
    # Q4_K는 256개 요소의 슈퍼블록을 사용
    QK_K = 256
    
    # 형태 확인 및 재구성
    if len(arr.shape) != 2:
        arr = arr.reshape(-1, arr.shape[-1])
    
    n_rows, n_cols = arr.shape
    
    # QK_K(256)로 패딩
    if n_cols % QK_K != 0:
        pad_size = QK_K - (n_cols % QK_K)
        arr = np.pad(arr, ((0, 0), (0, pad_size)))
        n_cols = arr.shape[1]
    
    # 슈퍼블록으로 재구성 (각 슈퍼블록은 256개 요소)
    n_blocks = n_cols // QK_K
    blocks = arr.reshape(-1, QK_K)
    
    quantized_blocks = []
    for block in blocks:
        # Q4_K: 8개 서브블록, 각각 32개 요소
        sub_blocks = block.reshape(8, 32)
        
        # 각 서브블록에 대한 스케일과 최소값 계산
        scales = []
        mins = []
        quantized_values = []
        
        for sub_block in sub_blocks:
            # 서브블록의 최소값과 최대값
            min_val = sub_block.min()
            max_val = sub_block.max()
            
            # 스케일 계산 (4비트이므로 15로 나눔)
            scale = (max_val - min_val) / 15.0 if max_val != min_val else 1.0
            
            scales.append(scale)
            mins.append(min_val)
            
            # 값들을 0-15 범위로 양자화
            if scale > 0:
                q_vals = np.clip(np.round((sub_block - min_val) / scale), 0, 15).astype(np.uint8)
            else:
                q_vals = np.zeros(32, dtype=np.uint8)
            
            quantized_values.extend(q_vals)
        
        # 스케일들을 6비트로 양자화 (0-63 범위)
        scales = np.array(scales)
        mins = np.array(mins)
        
        # 슈퍼블록 스케일
        d_scale = scales.max() if scales.max() > 0 else 1.0
        d_min = mins.min()
        
        # 스케일과 최소값을 6비트로 양자화
        scales_q = np.clip(np.round(scales / d_scale * 63), 0, 63).astype(np.uint8)
        mins_q = np.clip(np.round((mins - d_min) / (d_scale / 63) if d_scale > 0 else 0), 0, 63).astype(np.uint8)
        
        # 4비트 값들을 패킹 (2개씩 1바이트로)
        packed_values = []
        for i in range(0, len(quantized_values), 2):
            val1 = quantized_values[i] & 0xF
            val2 = quantized_values[i+1] & 0xF if i+1 < len(quantized_values) else 0
            packed_values.append(val1 | (val2 << 4))
        
        # 스케일과 최소값을 6비트씩 패킹 (4개씩 3바이트로)
        packed_scales = []
        for i in range(0, 8, 4):
            # 6비트 값 4개를 3바이트로 패킹
            s0 = scales_q[i] if i < 8 else 0
            s1 = scales_q[i+1] if i+1 < 8 else 0  
            s2 = scales_q[i+2] if i+2 < 8 else 0
            s3 = scales_q[i+3] if i+3 < 8 else 0
            
            # 24비트에 6비트씩 4개 패킹
            packed = s0 | (s1 << 6) | (s2 << 12) | (s3 << 18)
            packed_scales.extend([
                packed & 0xFF,
                (packed >> 8) & 0xFF, 
                (packed >> 16) & 0xFF
            ])
        
        packed_mins = []
        for i in range(0, 8, 4):
            m0 = mins_q[i] if i < 8 else 0
            m1 = mins_q[i+1] if i+1 < 8 else 0
            m2 = mins_q[i+2] if i+2 < 8 else 0  
            m3 = mins_q[i+3] if i+3 < 8 else 0
            
            packed = m0 | (m1 << 6) | (m2 << 12) | (m3 << 18)
            packed_mins.extend([
                packed & 0xFF,
                (packed >> 8) & 0xFF,
                (packed >> 16) & 0xFF  
            ])
        
        # 블록 데이터 결합: [d_scale(2) + d_min(2) + scales(6) + mins(6) + values(128)]
        block_data = np.concatenate([
            np.array([d_scale], dtype=np.float16).view(np.uint8),      # 2 bytes
            np.array([d_min], dtype=np.float16).view(np.uint8),        # 2 bytes  
            np.array(packed_scales, dtype=np.uint8),                   # 6 bytes
            np.array(packed_mins, dtype=np.uint8),                     # 6 bytes
            np.array(packed_values, dtype=np.uint8)                    # 128 bytes
        ])
        
        quantized_blocks.append(block_data)
    
    return np.array(quantized_blocks)

def save_q4k_weights_corrected(weights: Union[torch.nn.Module, List[torch.Tensor]], 
                              output_path: str):
    """PyTorch 모델이나 tensor 리스트를 올바른 Q4_K 형식으로 저장합니다."""
    # 모델인 경우 state_dict로 변환
    if isinstance(weights, torch.nn.Module):
        weights = list(weights.state_dict().values())
    
    quantized_layers = []
    for tensor in weights:
        if tensor.dtype in [torch.float32, torch.float16]:
            q4k_data = convert_to_q4k_corrected(tensor)
            quantized_layers.append({
                'shape': tensor.shape,
                'data': q4k_data
            })
    
    # 바이너리 형식으로 저장
    with open(output_path, 'wb') as f:
        f.write(np.uint32(len(quantized_layers)).tobytes())
        for layer in quantized_layers:
            # Shape 정보 저장
            f.write(np.uint32(len(layer['shape'])).tobytes())
            for dim in layer['shape']:
                f.write(np.uint32(dim).tobytes())
            # 양자화된 데이터 저장
            f.write(np.uint32(len(layer['data'])).tobytes())
            f.write(layer['data'].tobytes())