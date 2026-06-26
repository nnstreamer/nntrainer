#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
@file   quantize_q4_0_conv.py
@brief  Offline ggml Q4_0 block quantization of 1x1 conv weights for YOLOv11m.

Reads yolov11m_fused.safetensors (FP32), quantizes every eligible 1x1 conv
filter to ggml Q4_0 + repacks into nntrainer's q4_0x4 (ARM) or q4_0x8 (x86)
interleaved layout, and writes yolov11m_fused_q40.safetensors.

Quantization scheme (ggml Q4_0, matches nntr_ggml_impl_quant.cpp):
  - Block size QK4_0 = 32 elements per block.
  - Per block of 32 values: amax = max(|x|), d = amax / -8.
  - id = 1/d (or 0 when d==0).
  - qi = clip(round(x * id) + 8, 0, 15)  (unsigned 4-bit, range [0, 15])
  - Nibble packing (low nibble = first 16 elements, high nibble = last 16):
      qs[j] = qi[j] | (qi[j + 16] << 4)   for j in [0, 16)
  - d stored as fp16 (2 bytes, little-endian), then qs[16] = 18 bytes per block.

Layout match:
  Runtime path (layer_devel.h, quantize_q4_0 / repack_q4_0):
    1. weight tensor dim for quantized 1x1 conv = [1, 1, K=in_ch, N=out_ch]
       (Conv2DLayer::finalize sets this when Q4_0 dtype and kh==kw==1)
    2. weight.transpose("0:2:1") -> [1, 1, N=out_ch, K=in_ch]
    3. quantize_q4_0(transposed, nrow=N, n_per_row=K)
       -> N rows of K/32 plain block_q4_0 blocks
    4. repack_q4_0(plain, N, K, target_isa) -> q4_0xINTERLEAVE in-place
    5. save -> [1, 1, K, N] bytes
  Load path (Q4_0_Tensor::read): reads getMemoryBytes() bytes directly,
  no repack on load. File must contain already-repacked data.

  This tool replicates steps 2-5 offline from the FP32 weight.
  Input filter shape: [out_ch, in_ch, 1, 1] in safetensors.
  Weight is squeezed to [out_ch, in_ch] (= the transpose step).
  Output header shape: [1, 1, in_ch, out_ch] (= [K, N]).

Exclusion rule: 1x1 conv where in_ch % 32 != 0 OR out_ch % 32 != 0 are
kept as FP32 (matches runtime constraint N%32==0 && K%32==0).

Usage:
  python quantize_q4_0_conv.py \\
      --input  ../res/yolov11m_fused.safetensors \\
      --output ../res/yolov11m_fused_q40.safetensors \\
      --target arm      # arm (q4_0x4) or x86 (q4_0x8)

  # Dry-run: show which tensors would be quantized without writing
  python quantize_q4_0_conv.py --input ../res/yolov11m_fused.safetensors --dry-run

References:
  nntrainer/tensor/cpu_backend/ggml_interface/nntr_ggml_impl/nntr_ggml_impl_quant.cpp
  nntrainer/tensor/q4_0_utils.h   (block_q4_0, block_q4_0x4, block_q4_0x8)
  nntrainer/layers/layer_devel.h  (Q4_0 save path)
  nntrainer/layers/conv2d_layer.cpp (quant_matmul_filter layout)
  Applications/CausalLM/res/qwen3/qwen3-0.6b/gguf_to_nntrainer.py (repack impl)
"""

import argparse
import json
import struct
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Q4_0 constants
# ---------------------------------------------------------------------------
QK4_0 = 32
Q4_0_BLOCK_BYTES = 18   # 2 bytes fp16 d + 16 bytes qs

# Safetensors dtype strings for Q4_0 (opaque blob stored as U8; nntr_dtype=Q4_0)
SAFETENSORS_DTYPE_Q4_0 = "U8"
NNTR_DTYPE_Q4_0 = "Q4_0"


# ---------------------------------------------------------------------------
# Safetensors I/O helpers (reused from quantize_int4_conv.py)
# ---------------------------------------------------------------------------

def parse_safetensors(path: str):
    """Parse safetensors file. Returns (header_dict, raw_data_bytes, header_size)."""
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header_json = f.read(header_size).decode("utf-8")
        data = f.read()
    header = json.loads(header_json)
    return header, data, header_size


def write_safetensors(tensors: dict, path: str):
    """Write tensors to a safetensors file.

    tensors: dict[name -> {"dtype": str, "shape": list[int], "data": bytes,
                           "nntr_dtype": str (opt), "nntr_shape": list (opt)}]
    """
    blob = bytearray()
    header = {"__metadata__": {"format": "nntrainer", "nntr_format": "nntr-safetensors-v1"}}
    offset = 0
    for name, t in tensors.items():
        data = t["data"]
        end = offset + len(data)
        entry = {
            "dtype": t["dtype"],
            "shape": t["shape"],
            "data_offsets": [offset, end],
        }
        if "nntr_dtype" in t:
            entry["nntr_dtype"] = t["nntr_dtype"]
        if "nntr_shape" in t:
            entry["nntr_shape"] = t["nntr_shape"]
        header[name] = entry
        blob += data
        offset = end

    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_json)))
        f.write(header_json)
        f.write(blob)


# ---------------------------------------------------------------------------
# Q4_0 quantization (matches quantize_row_q4_0_ref in nntr_ggml_impl_quant.cpp)
# ---------------------------------------------------------------------------

def quantize_q4_0(w2d: np.ndarray) -> bytes:
    """Quantise float32 array [N, K] to raw Q4_0 block stream (plain, pre-repack).

    Each row of K elements produces K/32 blocks.
    Block layout: [fp16 d (2 bytes)] [qs[16] uint8] = 18 bytes.

    Matches quantize_row_q4_0_ref:
      d = max(x) / -8  (max = the value with max abs, preserving sign)
      qi = clip(round(x / d) + 8, 0, 15)
      qs[j] = qi[j] | (qi[j+16] << 4)   for j in [0, 16)
    """
    assert w2d.ndim == 2
    N, K = w2d.shape
    assert K % QK4_0 == 0, f"K={K} must be divisible by QK4_0={QK4_0}"

    w2d = w2d.astype(np.float32)
    nblocks_per_row = K // QK4_0
    total_blocks = N * nblocks_per_row

    # Reshape to (total_blocks, 32)
    blocks = w2d.reshape(total_blocks, QK4_0)

    # Find the maximum-magnitude value in each block (preserving sign, like ggml)
    abs_vals = np.abs(blocks)
    max_idx = np.argmax(abs_vals, axis=1)          # (nb,)
    max_val = blocks[np.arange(total_blocks), max_idx]  # signed max

    d = (max_val / -8.0).astype(np.float32)        # (nb,)
    id_ = np.where(d != 0.0, 1.0 / d, np.float32(0.0))

    # Quantize: round(x * id) + 8, clip to [0, 15]
    q = np.clip(np.rint(blocks * id_[:, None]) + 8.0, 0, 15).astype(np.uint8)

    # Nibble packing: qs[j] = q[j] | (q[j+16] << 4) for j in [0,16)
    low  = q[:, :16]   # (nb, 16)
    high = q[:, 16:]   # (nb, 16)
    qs = (low | (high << 4)).astype(np.uint8)      # (nb, 16)

    # Encode d as fp16 (2 bytes, little-endian)
    d_fp16 = d.astype(np.float16).view(np.uint16)
    d_bytes = d_fp16.view(np.uint8).reshape(total_blocks, 2)

    # Interleave: [2 bytes d][16 bytes qs] per block
    raw = np.concatenate([d_bytes, qs], axis=1).astype(np.uint8)  # (nb, 18)
    return raw.tobytes()


# ---------------------------------------------------------------------------
# Q4_0 repack (matches nntr_make_block_q4_0x4 / nntr_make_block_q4_0x8
#              in q4_0_utils.cpp, ported from gguf_to_nntrainer.py)
# ---------------------------------------------------------------------------

def repack_q4_0(raw_q4_0: bytes, N: int, K: int, interleave: int) -> bytes:
    """Repack plain Q4_0 blocks (N rows x K cols) to nntrainer's q4_0x4/x8 layout.

    Source: block_q4_0 array [N, K/32, 18 bytes].
    Target (interleave=4): block_q4_0x4 groups of 4 rows.
      Per super-block: d[4]*2 bytes, qs[4*16] bytes (XOR'd 0x88).
      Layout of qs: 8 iterations over (nblocks*8) bytes, rearranging by
      row=(i%4) and off=(i//4)*8.
    Target (interleave=8): block_q4_0x8 groups of 8 rows.
      Per super-block: d[8]*2 bytes, qs[8*16] bytes (XOR'd 0x88).

    XOR mask 0x88: negates the offset-by-8 encoding (8 -> 0, so the GEMM
    kernel sees signed nibbles without re-adding the bias).

    Matches nntr_make_block_q4_0x4 / nntr_make_block_q4_0x8 in q4_0_utils.cpp.
    Ported from Applications/CausalLM/res/qwen3/qwen3-0.6b/gguf_to_nntrainer.py.
    """
    assert interleave in (4, 8), f"interleave must be 4 or 8, got {interleave}"
    assert N % interleave == 0, \
        f"N={N} must be divisible by interleave={interleave}"
    assert K % QK4_0 == 0
    nblocks = K // QK4_0

    src = np.frombuffer(raw_q4_0, dtype=np.uint8).reshape(N, nblocks, Q4_0_BLOCK_BYTES)
    d_all  = src[:, :, :2]   # (N, nblocks, 2)
    qs_all = src[:, :, 2:]   # (N, nblocks, 16)

    if interleave == 8:
        # block_q4_0x8: d[8] (16 bytes) + qs[128 bytes] per nblocks super-block
        # nntr_make_block_q4_0x8: 16 iterations, row=i%8, off=(i//8)*8
        out = np.empty((N // 8, nblocks, 16 + 128), dtype=np.uint8)
        for g in range(N // 8):
            rows = slice(g * 8, g * 8 + 8)
            # d: 8 rows * 2 bytes -> (nblocks, 16)
            out[g, :, :16] = d_all[rows].transpose(1, 0, 2).reshape(nblocks, 16)
            # qs: (8, nblocks, 16) -> rearrange by row=i%8, off=(i//8)*8
            qs_chunk = qs_all[rows]             # (8, nblocks, 16)
            dst = np.empty((nblocks, 16, 8), dtype=np.uint8)
            for i in range(16):
                row = i % 8
                off = (i // 8) * 8
                dst[:, i, :] = qs_chunk[row, :, off:off + 8]
            dst = dst.reshape(nblocks, 128)
            dst ^= np.uint8(0x88)
            out[g, :, 16:] = dst
        return out.tobytes()
    else:  # interleave == 4
        # block_q4_0x4: d[4] (8 bytes) + qs[64 bytes] per nblocks super-block
        # nntr_make_block_q4_0x4: 8 iterations, row=i%4, off=(i//4)*8
        out = np.empty((N // 4, nblocks, 8 + 64), dtype=np.uint8)
        for g in range(N // 4):
            rows = slice(g * 4, g * 4 + 4)
            # d: 4 rows * 2 bytes -> (nblocks, 8)
            out[g, :, :8] = d_all[rows].transpose(1, 0, 2).reshape(nblocks, 8)
            # qs: (4, nblocks, 16) -> rearrange by row=i%4, off=(i//4)*8
            qs_chunk = qs_all[rows]             # (4, nblocks, 16)
            dst = np.empty((nblocks, 8, 8), dtype=np.uint8)
            for i in range(8):
                row = i % 4
                off = (i // 4) * 8
                dst[:, i, :] = qs_chunk[row, :, off:off + 8]
            dst = dst.reshape(nblocks, 64)
            dst ^= np.uint8(0x88)
            out[g, :, 8:] = dst
        return out.tobytes()


# ---------------------------------------------------------------------------
# Filter selection
# ---------------------------------------------------------------------------

def is_eligible_1x1_conv_filter(name: str, shape: list) -> bool:
    """Return True if this tensor is an eligible 1x1 conv filter for Q4_0.

    Criteria:
    - name ends with ':filter'
    - shape is 4D: [out_ch, in_ch, kh, kw] with kh==kw==1
    - NOT a depthwise conv ('dw:filter' in name)
    - in_ch % 32 == 0 AND out_ch % 32 == 0 (Q4_0 block alignment required)
    """
    if not name.endswith(":filter"):
        return False
    if len(shape) != 4:
        return False
    out_ch, in_ch, kh, kw = shape
    if kh != 1 or kw != 1:
        return False
    if name.endswith("dw:filter"):
        return False
    # Both dimensions must be divisible by 32 (N%32==0 && K%32==0)
    if in_ch % QK4_0 != 0 or out_ch % QK4_0 != 0:
        return False
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True,
                    help="Input FP32 safetensors (yolov11m_fused.safetensors)")
    ap.add_argument("--output", default=None,
                    help="Output path (default: <input stem>_q40.safetensors)")
    ap.add_argument("--target", choices=["arm", "x86"], default="arm",
                    help="Repack interleave target: arm=q4_0x4 (4), x86=q4_0x8 (8). "
                         "Default: arm (for device inference on S23 etc.)")
    ap.add_argument("--dry-run", action="store_true",
                    help="List tensors that would be quantized without writing")
    args = ap.parse_args()

    interleave = 4 if args.target == "arm" else 8

    if args.output is None:
        stem = args.input
        if stem.endswith(".safetensors"):
            stem = stem[:-len(".safetensors")]
        args.output = stem + "_q40.safetensors"

    print(f"Input:      {args.input}")
    print(f"Output:     {args.output}")
    print(f"Target:     {args.target} (q4_0x{interleave})")

    header, data, _ = parse_safetensors(args.input)
    tensor_names = [n for n in header if n != "__metadata__"]

    meta = header.get("__metadata__", {})
    print(f"Metadata:   {meta}")

    out_tensors = {}
    n_quantized = 0
    n_skipped_alignment = 0
    n_kept = 0
    total_fp32_bytes = 0
    total_q40_bytes = 0
    skipped_list = []

    for name in tensor_names:
        entry = header[name]
        dtype = entry["dtype"]
        shape = entry["shape"]
        start, end = entry["data_offsets"]
        raw = data[start:end]

        if dtype != "F32" or not name.endswith(":filter") or len(shape) != 4:
            # Non-filter or already quantized: keep as-is
            if not args.dry_run:
                out_tensors[name] = {"dtype": dtype, "shape": shape, "data": raw}
            if dtype == "F32" and len(shape) == 4 and shape[2] == 1 and shape[3] == 1:
                out_ch, in_ch = shape[0], shape[1]
                reason = []
                if in_ch % QK4_0 != 0:
                    reason.append(f"in_ch={in_ch} not div32")
                if out_ch % QK4_0 != 0:
                    reason.append(f"out_ch={out_ch} not div32")
                if reason and name.endswith(":filter") and not name.endswith("dw:filter"):
                    n_skipped_alignment += 1
                    skipped_list.append((name, shape, ", ".join(reason)))
                    print(f"  [skip-align] {name}  {shape}  ({', '.join(reason)})")
                else:
                    n_kept += 1
                    print(f"  [keep]       {name}  {shape}  {dtype}")
            else:
                n_kept += 1
                print(f"  [keep]       {name}  {shape}  {dtype}")
            continue

        out_ch, in_ch, kh, kw = shape

        # Check depthwise
        if name.endswith("dw:filter"):
            if not args.dry_run:
                out_tensors[name] = {"dtype": dtype, "shape": shape, "data": raw}
            n_kept += 1
            print(f"  [keep-dw]    {name}  {shape}  {dtype}")
            continue

        # 1x1 conv: check alignment
        if kh != 1 or kw != 1:
            if not args.dry_run:
                out_tensors[name] = {"dtype": dtype, "shape": shape, "data": raw}
            n_kept += 1
            print(f"  [keep-ksize] {name}  {shape}  {dtype}")
            continue

        # Alignment check
        if in_ch % QK4_0 != 0 or out_ch % QK4_0 != 0:
            reason = []
            if in_ch % QK4_0 != 0:
                reason.append(f"in_ch={in_ch} not div32")
            if out_ch % QK4_0 != 0:
                reason.append(f"out_ch={out_ch} not div32")
            n_skipped_alignment += 1
            skipped_list.append((name, shape, ", ".join(reason)))
            if not args.dry_run:
                out_tensors[name] = {"dtype": dtype, "shape": shape, "data": raw}
            print(f"  [skip-align] {name}  {shape}  ({', '.join(reason)})")
            continue

        # Eligible: quantize
        fp32_bytes = len(raw)
        print(f"  [Q4_0]       {name}  {shape}  {fp32_bytes} bytes FP32", end="")

        if args.dry_run:
            print(" -> would quantize")
            n_quantized += 1
            continue

        # Load [out_ch, in_ch, 1, 1] FP32
        w = np.frombuffer(raw, dtype=np.float32).reshape(shape)

        # Squeeze to [out_ch, in_ch] (the runtime transpose step for a weight
        # originally stored as [1,1,K=in_ch,N=out_ch] is weight.transpose("0:2:1")
        # which swaps H<->W -> [1,1,N=out_ch,K=in_ch]. For our FP32 input in
        # standard [out_ch,in_ch,1,1] layout, squeezing gives [out_ch,in_ch]
        # which equals [N, K] — exactly what quantize_q4_0(nrow=N, n_per_row=K)
        # expects.)
        w2d = w[:, :, 0, 0]  # [out_ch=N, in_ch=K]
        N_rows = out_ch   # number of Q4_0 rows (= out_ch)
        K_cols = in_ch    # columns per row (= in_ch)

        # Step 1: quantize to plain Q4_0 blocks [N*K/32, 18 bytes]
        raw_q40 = quantize_q4_0(w2d)

        # Step 2: repack to q4_0x{interleave}
        # Requirement: N % interleave == 0
        if N_rows % interleave != 0:
            print(f" -> SKIP repack (out_ch={N_rows} not div {interleave}), kept FP32")
            out_tensors[name] = {"dtype": dtype, "shape": shape, "data": raw}
            n_skipped_alignment += 1
            skipped_list.append((name, shape, f"out_ch={N_rows} not div interleave={interleave}"))
            continue

        repacked = repack_q4_0(raw_q40, N_rows, K_cols, interleave)

        # Verify repacked size == Q4_0_Tensor::size() = N*K/QK4_0 * 18
        expected_bytes = (N_rows * K_cols // QK4_0) * Q4_0_BLOCK_BYTES
        assert len(repacked) == expected_bytes, \
            f"{name}: repacked size {len(repacked)} != expected {expected_bytes}"

        # Header shape: [1, 1, K=in_ch, N=out_ch] (Q4_0_Tensor constructor
        # requires batch=1, channel=1, width divisible by 32)
        nntr_shape = [1, 1, K_cols, N_rows]

        out_tensors[name] = {
            "dtype": SAFETENSORS_DTYPE_Q4_0,   # "U8" (opaque blob)
            "shape": [len(repacked)],            # flat byte count as shape[0]
            "data": repacked,
            "nntr_dtype": NNTR_DTYPE_Q4_0,      # "Q4_0"
            "nntr_shape": nntr_shape,            # [1, 1, in_ch, out_ch]
        }

        q40_bytes = len(repacked)
        ratio = 100.0 * q40_bytes / fp32_bytes
        print(f" -> {q40_bytes} bytes Q4_0 ({ratio:.1f}%)")

        n_quantized += 1
        total_fp32_bytes += fp32_bytes
        total_q40_bytes += q40_bytes

    print(f"\nSummary:")
    print(f"  Quantized (Q4_0): {n_quantized}")
    print(f"  Skipped (alignment/interleave): {n_skipped_alignment}")
    if skipped_list:
        for sname, sshape, sreason in skipped_list:
            print(f"    {sname}  {sshape}  ({sreason})")
    print(f"  Kept FP32/other:  {n_kept}")

    if not args.dry_run and n_quantized > 0:
        print(f"\n1x1 conv Q4_0 bytes: {total_fp32_bytes} FP32 -> "
              f"{total_q40_bytes} Q4_0 ({100.0 * total_q40_bytes / total_fp32_bytes:.1f}%)")
        write_safetensors(out_tensors, args.output)
        size = os.path.getsize(args.output)
        in_size = os.path.getsize(args.input)
        print(f"Written: {args.output}")
        print(f"  File size: {size / (1024*1024):.1f} MB  (input: {in_size / (1024*1024):.1f} MB)")

        # Blob layout summary
        print(f"\nBlob layout per Q4_0 weight tensor:")
        print(f"  [U8 blob, len = N*K/32 * 18 bytes]")
        print(f"  Repacked as q4_0x{interleave}: groups of {interleave} rows")
        print(f"  Per group-of-{interleave}: d[{interleave}]*2 bytes + "
              f"qs[{interleave}*16] bytes (XOR'd 0x88)")
        print(f"  Header nntr_shape: [1, 1, K=in_ch, N=out_ch]")
        print(f"  Header nntr_dtype: Q4_0")
    elif args.dry_run:
        print(f"\n(dry-run: no file written)")


if __name__ == "__main__":
    main()
