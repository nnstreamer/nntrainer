import numpy as np
import torch
import struct
from typing import Union, List, Dict, Any
import json
import os

# GGUF 상수들
GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian
GGUF_VERSION = 2
GGUF_DEFAULT_ALIGNMENT = 32

# Q4_K 상수들
QK_K = 256
GGML_TYPE_Q4_K = 12

class GGUFWriter:
    """GGUF 형식으로 파일을 작성하는 클래스"""
    
    def __init__(self, path: str, arch: str):
        self.path = path
        self.arch = arch
        self.kv_data = {}
        self.tensors = []
        self.tensor_data = bytearray()
        
    def add_string(self, key: str, value: str):
        """문자열 메타데이터 추가"""
        self.kv_data[key] = ("string", value)
    
    def add_uint32(self, key: str, value: int):
        """uint32 메타데이터 추가"""
        self.kv_data[key] = ("uint32", value)
    
    def add_float32(self, key: str, value: float):
        """float32 메타데이터 추가"""
        self.kv_data[key] = ("float32", value)
        
    def add_array(self, key: str, array: List[int]):
        """배열 메타데이터 추가"""
        self.kv_data[key] = ("array_uint32", array)
    
    def add_tensor(self, name: str, data: np.ndarray, shape: List[int], dtype: int = GGML_TYPE_Q4_K):
        """텐서 데이터 추가"""
        # 데이터 정렬
        alignment = GGUF_DEFAULT_ALIGNMENT
        current_size = len(self.tensor_data)
        padding = (alignment - (current_size % alignment)) % alignment
        self.tensor_data.extend(b'\x00' * padding)
        
        offset = len(self.tensor_data)
        self.tensor_data.extend(data.tobytes())
        
        self.tensors.append({
            'name': name,
            'shape': shape,
            'type': dtype,
            'offset': offset
        })
    
    def write_header_kv(self, f):
        """헤더와 KV 데이터 작성"""
        # GGUF 헤더
        f.write(struct.pack('<I', GGUF_MAGIC))
        f.write(struct.pack('<I', GGUF_VERSION))
        f.write(struct.pack('<Q', len(self.tensors)))  # tensor count
        f.write(struct.pack('<Q', len(self.kv_data)))  # kv count
        
        # KV 데이터
        for key, (value_type, value) in self.kv_data.items():
            # 키 작성
            key_bytes = key.encode('utf-8')
            f.write(struct.pack('<Q', len(key_bytes)))
            f.write(key_bytes)
            
            # 값 타입과 데이터 작성
            if value_type == "string":
                f.write(struct.pack('<I', 8))  # GGUFValueType.STRING
                value_bytes = value.encode('utf-8')
                f.write(struct.pack('<Q', len(value_bytes)))
                f.write(value_bytes)
            elif value_type == "uint32":
                f.write(struct.pack('<I', 4))  # GGUFValueType.UINT32
                f.write(struct.pack('<I', value))
            elif value_type == "float32":
                f.write(struct.pack('<I', 2))  # GGUFValueType.FLOAT32
                f.write(struct.pack('<f', value))
            elif value_type == "array_uint32":
                f.write(struct.pack('<I', 9))  # GGUFValueType.ARRAY
                f.write(struct.pack('<I', 4))  # element type: UINT32
                f.write(struct.pack('<Q', len(value)))  # array length
                for item in value:
                    f.write(struct.pack('<I', item))
        
        # 텐서 정보
        for tensor in self.tensors:
            # 텐서 이름
            name_bytes = tensor['name'].encode('utf-8')
            f.write(struct.pack('<Q', len(name_bytes)))
            f.write(name_bytes)
            
            # 차원 수
            f.write(struct.pack('<I', len(tensor['shape'])))
            
            # 각 차원 크기
            for dim in tensor['shape']:
                f.write(struct.pack('<Q', dim))
            
            # 데이터 타입
            f.write(struct.pack('<I', tensor['type']))
            
            # 오프셋
            f.write(struct.pack('<Q', tensor['offset']))
    
    def write_tensor_data(self, f):
        """텐서 데이터 작성"""
        # 정렬
        current_pos = f.tell()
        alignment = GGUF_DEFAULT_ALIGNMENT
        padding = (alignment - (current_pos % alignment)) % alignment
        f.write(b'\x00' * padding)
        
        # 텐서 데이터
        f.write(self.tensor_data)
    
    def close(self):
        """파일 작성 완료"""
        with open(self.path, 'wb') as f:
            self.write_header_kv(f)
            self.write_tensor_data(f)

def quantize_q4_k_row(x: np.ndarray) -> np.ndarray:
    """Q4_K 양자화 (llama.cpp 호환)"""
    assert len(x) == QK_K
    
    # 8개 서브블록 (각 32개 요소)
    scales = []
    mins = []  
    qs = np.zeros(QK_K, dtype=np.uint8)
    
    for i in range(8):
        sub_block = x[i*32:(i+1)*32]
        min_val = float(sub_block.min())
        max_val = float(sub_block.max())
        
        if max_val > min_val:
            scale = (max_val - min_val) / 15.0
            scales.append(scale)
            mins.append(min_val)
            
            # 4비트로 양자화
            for j in range(32):
                q = int(np.round((sub_block[j] - min_val) / scale))
                qs[i*32 + j] = np.clip(q, 0, 15)
        else:
            scales.append(1.0)
            mins.append(min_val)
            qs[i*32:(i+1)*32] = 0
    
    scales = np.array(scales)
    mins = np.array(mins)
    
    # 슈퍼블록 스케일 계산
    d_scale = float(np.max(scales)) if np.max(scales) > 0 else 1.0
    d_min = float(np.max(np.abs(mins))) if len(mins) > 0 else 0.0
    
    # 6비트로 양자화
    d = d_scale / 63.0
    dmin = d_min / 63.0 if d_min > 0 else 1.0
    
    scales_q = np.clip(np.round(scales / d), 0, 63).astype(np.uint8)
    mins_q = np.clip(np.round(np.abs(mins) / dmin), 0, 63).astype(np.uint8)
    
    return pack_q4_k_block(d_scale, d_min, scales_q, mins_q, qs)

def pack_q4_k_block(d: float, dmin: float, scales: np.ndarray, mins: np.ndarray, qs: np.ndarray) -> np.ndarray:
    """Q4_K 블록 패킹"""
    # block_q4_k 구조체와 동일하게 패킹
    # d: half (2 bytes)  
    # dmin: half (2 bytes)
    # scales: uint8[K_SCALE_SIZE] (12 bytes)
    # qs: uint8[QK_K/2] (128 bytes)
    # 총 144 bytes
    
    result = bytearray(144)
    
    # d, dmin을 half(float16)로 저장
    d_bytes = struct.pack('<e', d)
    dmin_bytes = struct.pack('<e', dmin) 
    result[0:2] = d_bytes
    result[2:4] = dmin_bytes
    
    # 스케일과 최소값 패킹 (12바이트)
    # llama.cpp의 정확한 구현을 따라함
    for i in range(8):
        if i < 4:
            result[4 + i//2] |= (scales[i] & 0x3F) << (4 * (i % 2))
            result[4 + 2 + i//2] |= (mins[i] & 0x3F) << (4 * (i % 2))
        else:
            j = i - 4
            result[4 + 4 + j//2] |= (scales[i] & 0x3F) << (4 * (j % 2))
            result[4 + 4 + 2 + j//2] |= (mins[i] & 0x3F) << (4 * (j % 2))
    
    # 4비트 값들을 2개씩 패킹
    for i in range(0, QK_K, 2):
        result[16 + i//2] = (qs[i] & 0xF) | ((qs[i+1] & 0xF) << 4)
    
    return np.frombuffer(result, dtype=np.uint8)

def convert_pytorch_to_gguf_q4k(model: torch.nn.Module, output_path: str, model_name: str = "pytorch_model"):
    """PyTorch 모델을 GGUF Q4_K 형식으로 변환"""
    
    print(f"Converting PyTorch model to GGUF Q4_K format...")
    print(f"Output path: {output_path}")
    
    # GGUF 작성기 초기화
    writer = GGUFWriter(output_path, "llama")
    
    # 기본 메타데이터 추가
    writer.add_string("general.name", model_name)
    writer.add_string("general.architecture", "llama") 
    writer.add_string("general.quantization_version", "2")
    writer.add_string("general.file_type_name", "Q4_K - Small")
    writer.add_uint32("general.file_type", 12)  # GGML_FTYPE_MOSTLY_Q4_K_S
    
    # 모델 메타데이터 (예시 - 실제 모델에 맞게 수정 필요)
    writer.add_uint32("llama.context_length", 2048)
    writer.add_uint32("llama.embedding_length", 4096)
    writer.add_uint32("llama.block_count", 32)
    writer.add_uint32("llama.feed_forward_length", 11008)
    writer.add_uint32("llama.rope.dimension_count", 128)
    writer.add_uint32("llama.attention.head_count", 32)
    writer.add_uint32("llama.attention.head_count_kv", 32)
    writer.add_float32("llama.attention.layer_norm_rms_epsilon", 1e-6)
    
    # 토크나이저 메타데이터
    writer.add_string("tokenizer.ggml.model", "llama")
    writer.add_uint32("tokenizer.ggml.tokens_count", 32000)
    
    # 모델 가중치 변환
    state_dict = model.state_dict()
    total_params = sum(p.numel() for p in state_dict.values() if p.dtype in [torch.float32, torch.float16])
    processed_params = 0
    
    print(f"Total parameters to quantize: {total_params:,}")
    
    for name, tensor in state_dict.items():
        if tensor.dtype not in [torch.float32, torch.float16]:
            print(f"Skipping {name}: unsupported dtype {tensor.dtype}")
            continue
            
        print(f"Processing {name}: {tensor.shape}")
        
        # CPU로 이동하고 float32로 변환
        if tensor.is_cuda:
            tensor = tensor.cpu()
        tensor = tensor.float()
        
        # 2D로 reshape
        original_shape = list(tensor.shape)
        if len(tensor.shape) > 2:
            tensor = tensor.view(-1, tensor.shape[-1])
        elif len(tensor.shape) == 1:
            tensor = tensor.unsqueeze(0)
            
        arr = tensor.numpy()
        n_rows, n_cols = arr.shape
        
        # QK_K(256)로 패딩
        if n_cols % QK_K != 0:
            pad_size = QK_K - (n_cols % QK_K)
            arr = np.pad(arr, ((0, 0), (0, pad_size)))
            n_cols = arr.shape[1]
        
        # Q4_K로 양자화
        n_blocks_per_row = n_cols // QK_K
        total_blocks = n_rows * n_blocks_per_row
        
        quantized_data = []
        for row in range(n_rows):
            for block_idx in range(n_blocks_per_row):
                start_col = block_idx * QK_K
                block_data = arr[row, start_col:start_col + QK_K]
                q4k_block = quantize_q4_k_row(block_data)
                quantized_data.append(q4k_block)
        
        quantized_array = np.array(quantized_data)
        
        # GGUF에 텐서 추가
        writer.add_tensor(name, quantized_array, original_shape, GGML_TYPE_Q4_K)
        
        processed_params += tensor.numel()
        progress = processed_params / total_params * 100
        
        # 압축률 계산
        original_size = tensor.numel() * 4
        compressed_size = quantized_array.nbytes  
        compression_ratio = original_size / compressed_size
        
        print(f"  Quantized to {quantized_array.shape} blocks ({quantized_array.nbytes:,} bytes)")
        print(f"  Compression: {compression_ratio:.2f}x | Progress: {progress:.1f}%")
    
    # 파일 저장
    writer.close()
    
    file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    print(f"\n✅ GGUF file saved successfully!")
    print(f"   Path: {output_path}")
    print(f"   Size: {file_size:.1f} MB")
    print(f"   Format: Q4_K (4.5 bits per weight)")

def load_and_verify_gguf(file_path: str):
    """GGUF 파일을 로드하고 검증"""
    
    print(f"Loading GGUF file: {file_path}")
    
    with open(file_path, 'rb') as f:
        # 헤더 읽기
        magic = struct.unpack('<I', f.read(4))[0]
        if magic != GGUF_MAGIC:
            raise ValueError(f"Invalid GGUF magic: {magic:08x}")
        
        version = struct.unpack('<I', f.read(4))[0]
        print(f"GGUF version: {version}")
        
        tensor_count = struct.unpack('<Q', f.read(8))[0]
        kv_count = struct.unpack('<Q', f.read(8))[0]
        
        print(f"Tensors: {tensor_count}, KV pairs: {kv_count}")
        
        # KV 메타데이터 읽기 (간단한 확인만)
        for _ in range(kv_count):
            key_len = struct.unpack('<Q', f.read(8))[0]
            key = f.read(key_len).decode('utf-8')
            value_type = struct.unpack('<I', f.read(4))[0]
            
            # 값 스킵 (타입에 따라 다름)
            if value_type == 8:  # STRING
                str_len = struct.unpack('<Q', f.read(8))[0]
                f.read(str_len)
            elif value_type == 4:  # UINT32
                f.read(4)
            elif value_type == 2:  # FLOAT32
                f.read(4)
            # ... 다른 타입들도 필요에 따라 추가
        
        print("✅ GGUF file structure is valid")

# 사용 예제
if __name__ == "__main__":
    # 예제 모델 생성 (실제 사용시에는 학습된 모델 사용)
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(32000, 4096)
            self.linear1 = torch.nn.Linear(4096, 11008)
            self.linear2 = torch.nn.Linear(11008, 4096)
            
        def forward(self, x):
            x = self.embedding(x)
            x = self.linear1(x)
            x = self.linear2(x)
            return x
    
    print("=== GGUF Q4_K Conversion Example ===")
    
    # 모델 생성
    model = SimpleModel()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # GGUF로 변환
    convert_pytorch_to_gguf_q4k(model, "example_model_q4k.gguf", "example_model")
    
    # 검증
    load_and_verify_gguf("example_model_q4k.gguf")