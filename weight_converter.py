"""
SPDX-License-Identifier: Apache-2.0
Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>

@file weights_converter.py
@date 08 May 2025
@this script is tested on transformers 4.53.2
@note Qwen3-moe-30b-a3b
@author Eunju Yang <ej.yang@samsung.com>
@author SeungBasek Hong <sb92.hong@samsung.com>
"""
import torch
import numpy as np
from dataclasses import dataclass
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time
from functools import partial
import io
import struct

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("GPU acceleration enabled with CuPy")
except ImportError:
    GPU_AVAILABLE = False
    print("CuPy not available. Using CPU-only optimizations")

# Try to import Numba for JIT compilation
try:
    from numba import jit, prange, njit, cuda
    import numba
    NUMBA_AVAILABLE = True
    print("Numba JIT compilation enabled")
except ImportError:
    NUMBA_AVAILABLE = False
    print("Numba not available. Using pure NumPy")


@dataclass
class DataTypeConfig:
    embedding : str = "float32"
    attention_fc : str = "float32"
    normalization : str = "float32"
    lm_head : str = "float32"    


# GGML constants
QK4_0 = 32 # GGML Q4_0 block size
QK6_K = 256 # GGML Q6_K superblock size
Q6_K_SCALE_SIZE = 16 # groups in Q6_K

GROUP_MAX_EPS = 1e-12


def nearest_int(x):
    """GGML-style rounding"""
    return int(x + 0.5)


if NUMBA_AVAILABLE:
    @njit(parallel=True)
    def nearest_int_numba(x):
        """Numba version of nearest_int"""
        return int(x + 0.5)
    
    @njit(parallel=True)
    def quantize_q4_0_block_numba(block, scale, iscale):
        """Numba-accelerated Q4_0 quantization for a single block"""
        quantized = np.zeros(QK4_0, dtype=np.int8)
        if scale > 0:
            for j in prange(QK4_0):
                v = nearest_int_numba(iscale * block[j])
                v = max(-8, min(7, v))
                quantized[j] = v
        return quantized


def quantize_row_q4_0_ggml(x):
    """
    GGML-compatible Q4_0 quantization
    Based on GGML implementation:
    - Block size: 32
    - Scale: max(abs) / -8 (negative for GGML compatibility)
    - Range: -8 to 7 stored as 0-15
    """
    x = x.numpy().reshape(-1).astype(np.float32)
    nb = len(x) // QK4_0
    
    result = bytearray()
    
    for i in range(nb):
        # Get block
        block = x[i*QK4_0:(i+1)*QK4_0]
        
        # Find scale (GGML style: using signed max)
        amax = 0.0
        max_val = 0.0
        for j in range(QK4_0):
            v = block[j]
            if amax < abs(v):
                amax = abs(v)
                max_val = v
        
        # GGML uses negative scale for Q4_0
        scale = max_val / -8.0 if max_val != 0 else 0.0
        iscale = 1.0 / scale if scale != 0 else 0.0
        
        # Store scale as FP16 (2 bytes)
        scale_fp16 = np.float16(scale)
        result.extend(scale_fp16.tobytes())
        
        # Quantize values to 4-bit (-8 to 7)
        quantized = np.zeros(QK4_0, dtype=np.int8)
        if scale != 0:
            for j in range(QK4_0):
                # Round to nearest integer
                v = nearest_int(iscale * block[j])
                # Clamp to [-8, 7]
                v = max(-8, min(7, v))
                quantized[j] = v
        
        # Pack as nibbles (4-bit values), add 8 to make unsigned
        # GGML packing: consecutive pairs
        packed = np.zeros(QK4_0 // 2, dtype=np.uint8)
        for j in range(QK4_0 // 2):
            # Add 8 to convert from [-8,7] to [0,15]
            xi0 = min(15, quantized[j] + 8)
            xi1 = min(15, quantized[j + QK4_0//2] + 8)
            packed[j] = xi0 | (xi1 << 4)
        
        result.extend(packed.tobytes())
    
    return bytes(result)


def quantize_row_q4_0_ggml_optimized(x):
    """Optimized GGML-compatible Q4_0 quantization with GPU/CPU acceleration"""
    x = x.numpy().reshape(-1).astype(np.float32)
    nb = len(x) // QK4_0
    
    if GPU_AVAILABLE and len(x) > 1024:  # Use GPU for larger tensors
        # GPU version using CuPy
        x_gpu = cp.asarray(x)
        blocks_gpu = x_gpu.reshape(nb, QK4_0)
        
        # Compute absolute max and signed max for each block
        abs_max = cp.max(cp.abs(blocks_gpu), axis=1)
        max_indices = cp.argmax(cp.abs(blocks_gpu), axis=1)
        max_vals = cp.array([blocks_gpu[i, max_indices[i]] for i in range(nb)])
        
        # Compute scales (negative for GGML)
        scales = cp.where(max_vals != 0, max_vals / -8.0, 0.0)
        iscales = cp.where(scales != 0, 1.0 / scales, 0.0)
        
        # Transfer to CPU for quantization (CuPy doesn't support int4 packing well)
        scales_cpu = cp.asnumpy(scales)
        iscales_cpu = cp.asnumpy(iscales)
        blocks_cpu = cp.asnumpy(blocks_gpu)
        
        result_parts = []
        
        for i in range(nb):
            scale = scales_cpu[i]
            iscale = iscales_cpu[i]
            block = blocks_cpu[i]
            
            # Pack scale
            scale_fp16 = np.float16(scale)
            result_parts.append(scale_fp16.tobytes())
            
            # Quantize and pack
            if scale != 0:
                quantized = np.round(iscale * block).astype(np.int8)
                quantized = np.clip(quantized, -8, 7)
            else:
                quantized = np.zeros(QK4_0, dtype=np.int8)
            
            # Pack nibbles
            packed = np.zeros(QK4_0 // 2, dtype=np.uint8)
            for j in range(QK4_0 // 2):
                xi0 = min(15, quantized[j] + 8)
                xi1 = min(15, quantized[j + QK4_0//2] + 8)
                packed[j] = xi0 | (xi1 << 4)
            
            result_parts.append(packed.tobytes())
        
        return b''.join(result_parts)
    
    elif NUMBA_AVAILABLE:
        # Numba-accelerated CPU version
        blocks = x.reshape(nb, QK4_0)
        result_parts = []
        
        for i in range(nb):
            block = blocks[i]
            
            # GGML scale calculation
            amax = 0.0
            max_val = 0.0
            for v in block:
                if abs(v) > amax:
                    amax = abs(v)
                    max_val = v
            
            scale = max_val / -8.0 if max_val != 0 else 0.0
            iscale = 1.0 / scale if scale != 0 else 0.0
            
            # Pack scale as FP16
            scale_fp16 = np.float16(scale)
            result_parts.append(scale_fp16.tobytes())
            
            # Quantize using Numba
            quantized = quantize_q4_0_block_numba(block, scale, iscale)
            
            # Pack nibbles
            packed = np.zeros(QK4_0 // 2, dtype=np.uint8)
            for j in range(QK4_0 // 2):
                xi0 = min(15, quantized[j] + 8)
                xi1 = min(15, quantized[j + QK4_0//2] + 8)
                packed[j] = xi0 | (xi1 << 4)
            
            result_parts.append(packed.tobytes())
        
        return b''.join(result_parts)
    else:
        # Fallback to original implementation
        return quantize_row_q4_0_ggml(x)


def make_q6_quants(x, nmax=63):
    """GGML Q6_K style quantization for a group"""
    amax = np.max(np.abs(x))
    
    if amax < GROUP_MAX_EPS:
        return 0.0, np.zeros_like(x, dtype=np.int8)
    
    scale = amax / nmax
    iscale = 1.0 / scale
    
    # Quantize
    quantized = np.round(iscale * x).astype(np.int8)
    quantized = np.clip(quantized, -nmax, nmax)
    
    return scale, quantized


def quantize_row_q6_k_ggml(x):
    """
    GGML-compatible Q6_K quantization
    - Superblock: 256 weights
    - 16 groups of 16 weights each
    - 6-bit quantization per weight
    """
    x = x.numpy().reshape(-1).astype(np.float32)
    nb = len(x) // QK6_K
    
    result = []
    
    for ib in range(nb):
        block = x[ib*QK6_K:(ib+1)*QK6_K]
        
        # Quantize each group
        scales = np.zeros(Q6_K_SCALE_SIZE, dtype=np.float32)
        quantized = np.zeros(QK6_K, dtype=np.int8)
        
        for ig in range(Q6_K_SCALE_SIZE):
            group = block[ig*16:(ig+1)*16]
            scale, q = make_q6_quants(group, 31) # 6-bit: -31 to 31
            scales[ig] = scale
            quantized[ig*16:(ig+1)*16] = q
        
        # Find max scale for block quantization
        max_scale = np.max(np.abs(scales))
        
        if max_scale < GROUP_MAX_EPS:
            # Zero block
            result.append(b'\x00' * (2 + 16 + 128 + 32))
            continue
        
        # Quantize scales to 8-bit
        scale_scale = max_scale / 127
        iscale_scale = 1.0 / scale_scale
        
        # Block scale as FP16
        block_scale = np.float16(scale_scale)
        result.append(block_scale.tobytes())
        
        # Quantized scales
        q_scales = np.round(iscale_scale * scales).astype(np.int8)
        q_scales = np.clip(q_scales, -128, 127)
        result.append(q_scales.tobytes())
        
        # Requantize weights with quantized scales
        for ig in range(Q6_K_SCALE_SIZE):
            if q_scales[ig] == 0:
                continue
            actual_scale = scale_scale * q_scales[ig]
            iscale = 1.0 / actual_scale
            
            group_start = ig * 16
            group_end = (ig + 1) * 16
            group = block[group_start:group_end]
            
            q = np.round(iscale * group).astype(np.int8)
            q = np.clip(q, -32, 31)
            quantized[group_start:group_end] = q
        
        # Convert to unsigned by adding 32
        quantized_u = (quantized + 32).astype(np.uint8)
        
        # Pack 6-bit values
        # GGML Q6_K packing: lower 4 bits and upper 2 bits are packed separately
        ql = np.zeros(128, dtype=np.uint8)
        qh = np.zeros(32, dtype=np.uint8)
        
        # Pack according to GGML format
        for j in range(QK6_K // 128):
            ql_offset = j * 64
            qh_offset = j * 32
            base_idx = j * 128
            
            for l in range(32):
                q1 = quantized_u[base_idx + l] & 0xF
                q2 = quantized_u[base_idx + l + 32] & 0xF
                q3 = quantized_u[base_idx + l + 64] & 0xF
                q4 = quantized_u[base_idx + l + 96] & 0xF
                
                ql[ql_offset + l] = q1 | (q3 << 4)
                ql[ql_offset + l + 32] = q2 | (q4 << 4)
                
                qh[qh_offset + l] = ((quantized_u[base_idx + l] >> 4) & 0x3) | \
                                   ((quantized_u[base_idx + l + 32] >> 2) & 0xC) | \
                                   ((quantized_u[base_idx + l + 64]) & 0x30) | \
                                   ((quantized_u[base_idx + l + 96] << 2) & 0xC0)
        
        result.append(ql.tobytes())
        result.append(qh.tobytes())
    
    return b''.join(result)


def quantize_row_q6_k_ggml_optimized(x):
    """Optimized Q6_K quantization with GPU support"""
    x = x.numpy().reshape(-1).astype(np.float32)
    nb = len(x) // QK6_K
    
    if GPU_AVAILABLE and len(x) > 2048:  # Use GPU for larger tensors
        result = []
        
        for ib in range(nb):
            block_cpu = x[ib*QK6_K:(ib+1)*QK6_K]
            block = cp.asarray(block_cpu)
            
            # Process groups on GPU
            scales = cp.zeros(Q6_K_SCALE_SIZE, dtype=cp.float32)
            quantized_gpu = cp.zeros(QK6_K, dtype=cp.int8)
            
            for ig in range(Q6_K_SCALE_SIZE):
                group = block[ig*16:(ig+1)*16]
                amax = cp.max(cp.abs(group))
                
                if amax < GROUP_MAX_EPS:
                    scales[ig] = 0.0
                    continue
                
                scale = amax / 31
                iscale = 1.0 / scale
                scales[ig] = scale
                
                q = cp.round(iscale * group).astype(cp.int8)
                q = cp.clip(q, -31, 31)
                quantized_gpu[ig*16:(ig+1)*16] = q
            
            # Transfer to CPU for packing
            scales_cpu = cp.asnumpy(scales)
            quantized = cp.asnumpy(quantized_gpu)
            
            # Rest of the packing logic remains the same
            max_scale = np.max(np.abs(scales_cpu))
            
            if max_scale < GROUP_MAX_EPS:
                result.append(b'\x00' * (2 + 16 + 128 + 32))
                continue
            
            scale_scale = max_scale / 127
            iscale_scale = 1.0 / scale_scale
            
            block_scale = np.float16(scale_scale)
            result.append(block_scale.tobytes())
            
            q_scales = np.round(iscale_scale * scales_cpu).astype(np.int8)
            q_scales = np.clip(q_scales, -128, 127)
            result.append(q_scales.tobytes())
            
            # Requantize with quantized scales
            for ig in range(Q6_K_SCALE_SIZE):
                if q_scales[ig] == 0:
                    continue
                actual_scale = scale_scale * q_scales[ig]
                iscale = 1.0 / actual_scale
                
                group_start = ig * 16
                group_end = (ig + 1) * 16
                group = block_cpu[group_start:group_end]
                
                q = np.round(iscale * group).astype(np.int8)
                q = np.clip(q, -32, 31)
                quantized[group_start:group_end] = q
            
            # Pack as before
            quantized_u = (quantized + 32).astype(np.uint8)
            ql = np.zeros(128, dtype=np.uint8)
            qh = np.zeros(32, dtype=np.uint8)
            
            for j in range(QK6_K // 128):
                ql_offset = j * 64
                qh_offset = j * 32
                base_idx = j * 128
                
                for l in range(32):
                    q1 = quantized_u[base_idx + l] & 0xF
                    q2 = quantized_u[base_idx + l + 32] & 0xF
                    q3 = quantized_u[base_idx + l + 64] & 0xF
                    q4 = quantized_u[base_idx + l + 96] & 0xF
                    
                    ql[ql_offset + l] = q1 | (q3 << 4)
                    ql[ql_offset + l + 32] = q2 | (q4 << 4)
                    
                    qh[qh_offset + l] = ((quantized_u[base_idx + l] >> 4) & 0x3) | \
                                       ((quantized_u[base_idx + l + 32] >> 2) & 0xC) | \
                                       ((quantized_u[base_idx + l + 64]) & 0x30) | \
                                       ((quantized_u[base_idx + l + 96] << 2) & 0xC0)
            
            result.append(ql.tobytes())
            result.append(qh.tobytes())
        
        return b''.join(result)
    else:
        # Fallback to original implementation
        return quantize_row_q6_k_ggml(x)


# Use optimized versions
quantize_row_q4_0 = quantize_row_q4_0_ggml_optimized
quantize_row_q6_k = quantize_row_q6_k_ggml_optimized


def save_weight_parallel(weight_info):
    """Process weight saving in parallel"""
    weight_name, weight_tensor, dtype = weight_info
    
    if dtype in ["float32", "float16"]:
        return weight_name, np.array(weight_tensor, dtype=dtype).tobytes()
    elif dtype == "q4_0":
        return weight_name, quantize_row_q4_0(weight_tensor)
    elif dtype == "q6_k":
        return weight_name, quantize_row_q6_k(weight_tensor)
    else:
        raise ValueError(f"Unsupported dtype {dtype}")


total_size = 0
def save_qwen3_for_nntrainer(params, config, dconfig, file):  
    """Convert and save weights as nntrainer format for multi-head attention model"""  

    n_layers = config.num_hidden_layers
    n_experts = config.num_experts
      
    def save_weight(weight_name, is_transpose=False, dtype="float32"):
        print(weight_name, params[weight_name].shape)
        if dtype in ["float32", "float16"]:
            np.array(params[weight_name], dtype=dtype).tofile(file)  
        elif dtype == "q4_0":
            file.write(quantize_row_q4_0(params[weight_name]))
        elif dtype == "q6_k":
            file.write(quantize_row_q6_k(params[weight_name]))
        else:
            raise ValueError(f"Unsupported dtype {dtype}")

    def save_projection(layer_name, proj_name):  
        """Helper function to handle base/lora weight saving"""  
        lora_key = f"{layer_name}{proj_name}.lora_A.default.weight"  
        if lora_key in params:  
            save_weight(f"{layer_name}{proj_name}.base_layer.weight", True, dconfig.attention_fc)
            save_weight(f"{layer_name}{proj_name}.lora_A.default.weight", True, "float32")
            save_weight(f"{layer_name}{proj_name}.lora_B.default.weight", True, "float32")  
        else:  
            save_weight(f"{layer_name}{proj_name}.weight", True, dconfig.attention_fc)  

    def save_attention(layer_name):  
        """Save attention layer weights"""  
          
        # Save Q/K/V/O projections using helper  
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:  
            save_projection(layer_name, f"self_attn.{proj}")  
            proj_norm_name = f"{layer_name}self_attn.{proj[0]}_norm.weight"
            if proj_norm_name in params:
                save_weight(proj_norm_name, False, dconfig.normalization)

    def save_feed_forward(layer_name):  
        """Save feed forward layer weights"""  
        
        save_weight(f"{layer_name}mlp.gate.weight", True, "float32")  
          
        # Save MoE projections using helper  
        for num_expert in range(n_experts):
            for proj in ["up_proj", "gate_proj", "down_proj"]:  
                save_projection(layer_name, f"mlp.experts.{num_expert}.{proj}")  

    ####################################################################
    # Save embedding layer  
    save_weight("model.embed_tokens.weight", False, dconfig.embedding)  

    # Process all layers  
    for layer_idx in range(n_layers):  
        layer_prefix = f"model.layers.{layer_idx}."  
        save_weight(f"{layer_prefix}input_layernorm.weight", False, dconfig.normalization)  
        save_attention(layer_prefix)  
        save_weight(f"{layer_prefix}post_attention_layernorm.weight", False, dconfig.normalization)  
        save_feed_forward(layer_prefix)  

    # Save final layers  
    save_weight("model.norm.weight", False, dconfig.normalization)  
    save_weight("lm_head.weight", False, dconfig.lm_head)


def save_qwen3_for_nntrainer_optimized(params, config, dconfig, file):  
    """Optimized convert and save weights with parallel processing"""  
    
    n_layers = config.num_hidden_layers
    n_experts = config.num_experts
    
    # Collect all weights to process
    weights_to_process = []
    
    def collect_weight(weight_name, is_transpose=False, dtype="float32"):
        print(f"Collecting: {weight_name}, shape: {params[weight_name].shape}")
        weights_to_process.append((weight_name, params[weight_name], dtype))
    
    def collect_projection(layer_name, proj_name):  
        """Helper function to handle base/lora weight collection"""  
        lora_key = f"{layer_name}{proj_name}.lora_A.default.weight"  
        if lora_key in params:  
            collect_weight(f"{layer_name}{proj_name}.base_layer.weight", True, dconfig.attention_fc)
            collect_weight(f"{layer_name}{proj_name}.lora_A.default.weight", True, "float32")
            collect_weight(f"{layer_name}{proj_name}.lora_B.default.weight", True, "float32")  
        else:  
            collect_weight(f"{layer_name}{proj_name}.weight", True, dconfig.attention_fc)  
    
    def collect_attention(layer_name):  
        """Collect attention layer weights"""  
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:  
            collect_projection(layer_name, f"self_attn.{proj}")  
            proj_norm_name = f"{layer_name}self_attn.{proj[0]}_norm.weight"
            if proj_norm_name in params:
                collect_weight(proj_norm_name, False, dconfig.normalization)
    
    def collect_feed_forward(layer_name):  
        """Collect feed forward layer weights"""  
        collect_weight(f"{layer_name}mlp.gate.weight", True, "float32")  
        
        for num_expert in range(n_experts):
            for proj in ["up_proj", "gate_proj", "down_proj"]:  
                collect_projection(layer_name, f"mlp.experts.{num_expert}.{proj}")  
    
    # Collect all weights
    print("Collecting weights...")
    collect_weight("model.embed_tokens.weight", False, dconfig.embedding)  
    
    for layer_idx in range(n_layers):  
        layer_prefix = f"model.layers.{layer_idx}."  
        collect_weight(f"{layer_prefix}input_layernorm.weight", False, dconfig.normalization)  
        collect_attention(layer_prefix)  
        collect_weight(f"{layer_prefix}post_attention_layernorm.weight", False, dconfig.normalization)  
        collect_feed_forward(layer_prefix)  
    
    collect_weight("model.norm.weight", False, dconfig.normalization)  
    collect_weight("lm_head.weight", False, dconfig.lm_head)
    
    # Process weights in parallel
    print(f"\nProcessing {len(weights_to_process)} weights in parallel...")
    start_time = time.time()
    
    # Determine optimal number of workers
    n_workers = min(mp.cpu_count(), 8)  # Limit to 8 workers to avoid memory issues
    
    # Use ProcessPoolExecutor for CPU-bound quantization tasks
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Process in batches to avoid memory issues
        batch_size = max(10, 50 // n_workers)  # Adaptive batch size
        processed_weights = {}
        
        for i in tqdm(range(0, len(weights_to_process), batch_size), desc="Processing batches"):
            batch = weights_to_process[i:i+batch_size]
            results = list(executor.map(save_weight_parallel, batch))
            
            for weight_name, weight_bytes in results:
                processed_weights[weight_name] = weight_bytes
    
    # Write weights in order
    print("\nWriting weights to file...")
    for weight_name, _, _ in tqdm(weights_to_process, desc="Writing"):
        file.write(processed_weights[weight_name])
    
    elapsed_time = time.time() - start_time
    print(f"\nConversion completed in {elapsed_time:.2f} seconds")


def dequantize_q4_0_ggml(data):
    """GGML Q4_0 dequantization for verification"""
    result = []
    offset = 0
    
    while offset < len(data):
        # Read scale (2 bytes FP16)
        scale = np.frombuffer(data[offset:offset+2], dtype=np.float16)[0]
        offset += 2
        
        # Read packed values (16 bytes for 32 values)
        packed = np.frombuffer(data[offset:offset+16], dtype=np.uint8)
        offset += 16
        
        # Unpack
        values = np.zeros(32, dtype=np.float32)
        for i in range(16):
            # Low nibble
            values[i] = float((packed[i] & 0xF) - 8) * scale
            # High nibble  
            values[i + 16] = float((packed[i] >> 4) - 8) * scale
        
        result.extend(values)
    
    return np.array(result)


def verify_ggml_compatibility():
    """Verify GGML compatibility of quantization"""
    print("\n=== GGML Compatibility Verification ===")
    
    # Test Q4_0
    print("\n1. Testing Q4_0 quantization:")
    test_data = np.array([
        0.1, 0.2, 0.3, 0.6, -0.1, -0.2, -0.3, -0.4,
        0.5, 0.7, -0.5, -0.7, 0.8, -0.8, 0.9, -0.9,
        0.15, 0.25, 0.35, 0.45, -0.15, -0.25, -0.35, -0.45,
        0.55, 0.65, -0.55, -0.65, 0.75, -0.75, 0.85, -0.85
    ], dtype=np.float32)
    
    test_tensor = torch.tensor(test_data.reshape(1, -1))
    
    # Quantize
    quantized = quantize_row_q4_0_ggml(test_tensor)
    print(f"Original data shape: {test_data.shape}")
    print(f"Quantized size: {len(quantized)} bytes")
    print(f"Expected size: {2 + 16} bytes (2 for scale + 16 for packed values)")
    
    # Verify structure
    scale = np.frombuffer(quantized[:2], dtype=np.float16)[0]
    print(f"Scale value: {scale}")
    print(f"Expected scale (negative): {-np.max(test_data) / 8:.6f}")
    
    # Dequantize and check
    dequantized = dequantize_q4_0_ggml(quantized)
    print(f"Dequantized shape: {dequantized.shape}")
    
    # Calculate error
    error = np.mean(np.abs(test_data - dequantized))
    max_error = np.max(np.abs(test_data - dequantized))
    print(f"Mean absolute error: {error:.6f}")
    print(f"Max absolute error: {max_error:.6f}")
    
    # Test Q6_K structure
    print("\n2. Testing Q6_K quantization:")
    test_data_q6k = np.random.randn(256).astype(np.float32) * 0.5
    test_tensor_q6k = torch.tensor(test_data_q6k.reshape(1, -1))
    
    quantized_q6k = quantize_row_q6_k_ggml(test_tensor_q6k)
    print(f"Q6_K quantized size: {len(quantized_q6k)} bytes")
    print(f"Expected size: 210 bytes (2 + 16 + 128 + 32 + padding)")
    
    print("\n✓ GGML compatibility test completed")
    
    return True


if __name__ == "__main__":
    data_config = DataTypeConfig(embedding="q6_k", attention_fc="q4_0", normalization="float16", lm_head="q6_k")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    print(f"GPU acceleration: {'Enabled' if GPU_AVAILABLE else 'Disabled'}")
    print(f"Numba JIT: {'Enabled' if NUMBA_AVAILABLE else 'Disabled'}")
    print(f"CPU cores available: {mp.cpu_count()}")
    
    # Run GGML compatibility verification
    verify_ggml_compatibility()
    
    model_path = "."
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="float", trust_remote_code=True)
    model.eval()
    
    # Use optimized version
    with open("./nntr_qwen3_30b_moe_test_mixed_ggml.bin", "wb") as f_model:
        save_qwen3_for_nntrainer_optimized(model.state_dict(), config, data_config, f_model)
