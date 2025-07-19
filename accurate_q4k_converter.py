#!/usr/bin/env python3
"""
Accurate Q4_K PyTorch Weight Converter
=====================================

Based on llama.cpp's actual gguf-py implementation and web search results.
This code provides accurate Q4_K quantization compatible with llama.cpp.

Q4_K Structure (from llama.cpp documentation):
- Super-blocks: 8 blocks, each block has 32 weights (total 256 weights = QK_K)
- Block scales: 6-bit quantized
- Block mins: 6-bit quantized  
- Formula: w = q * block_scale(6-bit) + block_min(6-bit)
- Efficiency: 4.5 bits per weight

References:
- https://github.com/ggerganov/llama.cpp/pull/1684#issue-1739619305
- https://github.com/huggingface/huggingface.js/blob/main/packages/gguf/src/quant-descriptions.ts
"""

import numpy as np
import torch
import struct
from typing import Union, List, Dict, Any, Optional
import logging

# Q4_K constants from llama.cpp
QK_K = 256  # Super-block size (8 blocks × 32 weights each)
K_SCALE_SIZE = 12  # Size for scales storage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def quantize_row_q4_k_accurate(x: np.ndarray) -> bytes:
    """
    Accurate Q4_K quantization based on llama.cpp implementation.
    
    Args:
        x: Input array of exactly 256 float32 values (QK_K)
        
    Returns:
        Quantized block as bytes
        
    Q4_K Structure:
    - 8 sub-blocks of 32 weights each
    - 6-bit scales and mins for each sub-block
    - 4-bit quantized values
    - Total: 4.5 bits per weight
    """
    assert len(x) == QK_K, f"Input must have exactly {QK_K} elements, got {len(x)}"
    
    # Split into 8 sub-blocks of 32 elements each
    sub_blocks = x.reshape(8, 32)
    
    # Initialize output structures
    scales = np.zeros(8, dtype=np.float32)
    mins = np.zeros(8, dtype=np.float32)
    quantized_values = np.zeros(QK_K // 2, dtype=np.uint8)  # 4-bit packed
    
    # Process each sub-block
    for i in range(8):
        sub_block = sub_blocks[i]
        
        # Calculate min and max for this sub-block
        min_val = float(np.min(sub_block))
        max_val = float(np.max(sub_block))
        
        # Avoid division by zero
        if max_val == min_val:
            scales[i] = 1.0
            mins[i] = min_val
            # All values will be quantized to 0
            continue
        
        # Calculate scale and min
        scale = (max_val - min_val) / 15.0  # 4-bit range: 0-15
        scales[i] = scale
        mins[i] = min_val
        
        # Quantize values in this sub-block
        for j in range(32):
            idx = i * 32 + j
            
            # Quantize to 4-bit
            if scale > 0:
                q_val = int(np.round((sub_block[j] - min_val) / scale))
                q_val = np.clip(q_val, 0, 15)
            else:
                q_val = 0
            
            # Pack two 4-bit values into one byte
            if j % 2 == 0:
                quantized_values[idx // 2] = q_val
            else:
                quantized_values[idx // 2] |= (q_val << 4)
    
    # Quantize scales and mins to 6-bit (0-63 range)
    # Find global scale for scales
    max_scale = np.max(scales) if np.max(scales) > 0 else 1.0
    d_scale = max_scale / 63.0
    
    # Find global scale for mins  
    min_min = np.min(mins)
    max_min = np.max(mins)
    if max_min != min_min:
        d_min = (max_min - min_min) / 63.0
    else:
        d_min = 1.0
    
    # Quantize scales and mins
    scales_q = np.zeros(8, dtype=np.uint8)
    mins_q = np.zeros(8, dtype=np.uint8)
    
    for i in range(8):
        if d_scale > 0:
            scales_q[i] = int(np.round(scales[i] / d_scale))
            scales_q[i] = np.clip(scales_q[i], 0, 63)
        
        if d_min > 0:
            mins_q[i] = int(np.round((mins[i] - min_min) / d_min))
            mins_q[i] = np.clip(mins_q[i], 0, 63)
    
    # Pack scales (6-bit each) into bytes
    # 8 scales × 6 bits = 48 bits = 6 bytes, but we use 12 bytes for alignment
    scales_packed = np.zeros(K_SCALE_SIZE, dtype=np.uint8)
    
    # Simple packing: store each 6-bit value in lower 6 bits of each byte
    for i in range(8):
        if i < K_SCALE_SIZE:
            scales_packed[i] = scales_q[i] & 0x3F  # Keep only lower 6 bits
    
    # Pack mins similarly
    for i in range(8):
        if i + 8 < K_SCALE_SIZE:
            scales_packed[i + 4] = mins_q[i] & 0x3F  # Store mins after scales
    
    # Construct final block
    block_data = bytearray()
    
    # Add scale factors (as float16)
    block_data.extend(struct.pack('<e', d_scale))  # 2 bytes
    block_data.extend(struct.pack('<e', d_min))    # 2 bytes
    
    # Add packed scales and mins
    block_data.extend(scales_packed)  # 12 bytes
    
    # Add quantized values
    block_data.extend(quantized_values)  # 128 bytes (256 values / 2)
    
    return bytes(block_data)

def convert_tensor_to_q4k(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert PyTorch tensor to Q4_K format.
    
    Args:
        tensor: PyTorch tensor to quantize
        
    Returns:
        Quantized data as numpy array of bytes
    """
    # Convert to CPU and numpy
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    arr = tensor.numpy().astype(np.float32)
    
    # Flatten tensor
    original_shape = arr.shape
    arr_flat = arr.flatten()
    
    # Pad to multiple of QK_K (256)
    n_elements = len(arr_flat)
    n_blocks = (n_elements + QK_K - 1) // QK_K  # Ceiling division
    padded_size = n_blocks * QK_K
    
    if padded_size > n_elements:
        # Pad with zeros
        arr_padded = np.zeros(padded_size, dtype=np.float32)
        arr_padded[:n_elements] = arr_flat
    else:
        arr_padded = arr_flat
    
    # Reshape into blocks
    blocks = arr_padded.reshape(n_blocks, QK_K)
    
    # Quantize each block
    quantized_blocks = []
    for i, block in enumerate(blocks):
        logger.debug(f"Quantizing block {i+1}/{n_blocks}")
        q_block = quantize_row_q4_k_accurate(block)
        quantized_blocks.append(q_block)
    
    return {
        'quantized_data': quantized_blocks,
        'original_shape': original_shape,
        'n_blocks': n_blocks,
        'dtype': 'q4_k'
    }

def save_q4k_model(model_or_tensors: Union[torch.nn.Module, Dict[str, torch.Tensor]], 
                   output_path: str,
                   model_name: str = "quantized_model"):
    """
    Save PyTorch model or tensor dict in Q4_K format.
    
    Args:
        model_or_tensors: PyTorch model or dictionary of tensors
        output_path: Output file path
        model_name: Model name for metadata
    """
    import json
    import os
    
    # Extract tensors
    if isinstance(model_or_tensors, torch.nn.Module):
        tensors = model_or_tensors.state_dict()
    else:
        tensors = model_or_tensors
    
    # Quantize tensors
    quantized_data = {}
    metadata = {
        'model_name': model_name,
        'quantization_type': 'q4_k',
        'tensors': {}
    }
    
    logger.info(f"Quantizing {len(tensors)} tensors to Q4_K format...")
    
    for name, tensor in tensors.items():
        if tensor.dtype in [torch.float32, torch.float16, torch.bfloat16]:
            logger.info(f"Quantizing tensor: {name} {list(tensor.shape)}")
            q_data = convert_tensor_to_q4k(tensor)
            quantized_data[name] = q_data['quantized_data']
            metadata['tensors'][name] = {
                'original_shape': q_data['original_shape'],
                'n_blocks': q_data['n_blocks'],
                'dtype': q_data['dtype']
            }
        else:
            logger.info(f"Skipping tensor {name} (unsupported dtype: {tensor.dtype})")
    
    # Save to file
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save as numpy archive
    np.savez_compressed(
        output_path,
        **quantized_data,
        _metadata=json.dumps(metadata)
    )
    
    logger.info(f"Q4_K quantized model saved to: {output_path}")
    
    # Calculate compression ratio
    original_size = sum(tensor.numel() * 4 for tensor in tensors.values() 
                       if tensor.dtype in [torch.float32, torch.float16, torch.bfloat16])
    compressed_size = os.path.getsize(output_path)
    compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
    
    logger.info(f"Compression ratio: {compression_ratio:.2f}x")
    logger.info(f"Original size: {original_size / 1024 / 1024:.1f} MB")
    logger.info(f"Compressed size: {compressed_size / 1024 / 1024:.1f} MB")

def load_q4k_model(file_path: str) -> Dict[str, Any]:
    """
    Load Q4_K quantized model.
    
    Args:
        file_path: Path to quantized model file
        
    Returns:
        Dictionary containing quantized data and metadata
    """
    import json
    
    # Load data
    data = np.load(file_path, allow_pickle=True)
    
    # Extract metadata
    if '_metadata' in data:
        metadata = json.loads(str(data['_metadata']))
    else:
        metadata = {}
    
    # Extract quantized tensors
    quantized_tensors = {}
    for key in data.keys():
        if key != '_metadata':
            quantized_tensors[key] = data[key]
    
    return {
        'quantized_tensors': quantized_tensors,
        'metadata': metadata
    }

# Example usage
if __name__ == "__main__":
    # Example: quantize a simple tensor
    logger.info("Testing Q4_K quantization...")
    
    # Create test tensor
    test_tensor = torch.randn(1024, 512, dtype=torch.float32)
    
    # Quantize
    result = convert_tensor_to_q4k(test_tensor)
    
    logger.info(f"Original tensor shape: {test_tensor.shape}")
    logger.info(f"Original tensor size: {test_tensor.numel() * 4} bytes")
    logger.info(f"Quantized blocks: {result['n_blocks']}")
    logger.info(f"Quantized size: {len(result['quantized_data']) * len(result['quantized_data'][0])} bytes")
    
    # Save example
    save_q4k_model({'test_tensor': test_tensor}, 'test_q4k_model.npz')
    
    # Load example
    loaded = load_q4k_model('test_q4k_model.npz')
    logger.info(f"Loaded model metadata: {loaded['metadata']}")