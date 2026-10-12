#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prepare column-sliced HTP FC weights without changing quantized values.

Usage::

    python3 tools/htp/prepare_fc_wh.py --model model.bin \
        --manifest fc_manifest.json --output prepared_fc
    python3 tools/htp/prepare_fc_wh.py --model model.bin \
        --manifest fc_manifest.json --output prepared_fc --verify-only

``nntr_quantize_stream --fc_dtype QS4CX_WH ...`` automatically emits
``htp_fc_manifest.json`` beside a successfully converted model containing
ConvBlock or dense FFN weights. Run this preparation tool on that model and
manifest, then copy the prepared directory to the device and set
``NNTR_HTP_PREPARED_DIR=/path/to/prepared_fc`` for inference. Preparation needs
no device or NPU. Existing WH models can use the equivalent exported manifest
without requantizing their weights.

The manifest contains ``schema_version: 1``, ``dtype: "QS4CX_WH"``, the
original ``model_size``, and ``weights`` with ``file_offset``, ``K``, ``N``
(and optional informational ``name`` / ``layer_type``). A sparse copy of the
model containing exactly these byte ranges is sufficient. Each source range
is K*N/2 WH bytes, N float32 scales, then N float32 column sums, all little
endian. Optional per-weight ``source_fnv1a64`` pins its expected content.

Output files are named OFFSET_K_N.nntrwh. Their portable version-1 format
is shared with htp_prepared_fc.h: a 64-byte header, 16-byte slice descriptors,
and payload-relative 512-byte-aligned slices of WH bytes + float32 scales +
int32 column sums. All source and payload bytes are checksummed with FNV-1a
64 (an integrity check, not a cryptographic signature). Whole WH tiles are
copied directly; weights are never dequantized, requantized, or unpacked.
"""

import argparse
import json
import math
import os
from pathlib import Path
import struct
import tempfile

HEADER = struct.Struct("<8sIIQQQIIQQ")
DESCRIPTOR = struct.Struct("<IIQ")
MAGIC = b"NNTRWH01"
MAX_PAYLOAD = 256 * 1024 * 1024
FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
UINT64_MASK = (1 << 64) - 1


def fnv1a64(data):
    """Checksum every byte, including padding and metadata."""
    value = FNV_OFFSET
    for byte in data:
        value = ((value ^ byte) * FNV_PRIME) & UINT64_MASK
    return value


def align512(size):
    return (size + 511) & ~511


def fc_slice_cols(k):
    """Match HtpComputeOps::fcSliceCols, including whole-head alignment."""
    columns = max(1, ((2 << 20) // 512) // (k // 32)) * 32
    return (columns // 512) * 512 if columns >= 512 else columns


def integer(value, field, maximum=UINT64_MASK):
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(f"{field} must be an integer in [0, {maximum}]")
    return value


def weight_bytes(k, n):
    return k * n // 2 + 8 * n


def weight_name(weight):
    return f"{weight['file_offset']}_{weight['K']}_{weight['N']}.nntrwh"


def load_manifest(path, model_size):
    with open(path, encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be an object")
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1 or manifest.get("dtype") != "QS4CX_WH":
        raise ValueError("expected schema_version=1 and dtype=QS4CX_WH")
    if integer(manifest.get("model_size"), "model_size") != model_size:
        raise ValueError("model_size does not match source file size")
    weights = manifest.get("weights")
    if not isinstance(weights, list) or not weights:
        raise ValueError("weights must be a nonempty list")
    ranges = []
    for weight in weights:
        if not isinstance(weight, dict):
            raise ValueError("each weight must be an object")
        off = integer(weight.get("file_offset"), "file_offset")
        k = integer(weight.get("K"), "K", (1 << 32) - 1)
        n = integer(weight.get("N"), "N", (1 << 32) - 1)
        if k == 0 or n == 0 or k % 32 or n % 32:
            raise ValueError("K and N must be positive multiples of 32")
        size = weight_bytes(k, n)
        if size > MAX_PAYLOAD or off + size > model_size:
            raise ValueError("source weight is too large or exceeds model bounds")
        ranges.append((off, off + size))
    previous_end = 0
    for off, end in sorted(ranges):
        if off < previous_end:
            raise ValueError("source weight ranges overlap or are duplicated")
        previous_end = end
    return weights


def read_source(stream, weight):
    k, n, off = weight["K"], weight["N"], weight["file_offset"]
    stream.seek(off)
    data = stream.read(weight_bytes(k, n))
    if len(data) != weight_bytes(k, n):
        raise ValueError("truncated source weight")
    source_hash = fnv1a64(data)
    pinned = weight.get("source_fnv1a64")
    if pinned is not None:
        if isinstance(pinned, str):
            pinned = int(pinned, 16)
        if integer(pinned, "source_fnv1a64") != source_hash:
            raise ValueError("source_fnv1a64 mismatch")
    return data, source_hash


def make_payload(source, k, n):
    wh_size = k * n // 2
    scales = struct.unpack_from(f"<{n}f", source, wh_size)
    columns = struct.unpack_from(f"<{n}f", source, wh_size + 4 * n)
    if any(not math.isfinite(v) or v < 0 for v in scales):
        raise ValueError("scales must be finite and nonnegative")
    if any(not math.isfinite(v) or not v.is_integer() or
           v < -8 * k or v > 7 * k or not -(1 << 31) <= v < (1 << 31)
           for v in columns):
        raise ValueError("column sums must be finite exact int32 values within int4 bounds")
    cap = fc_slice_cols(k)
    slices = [(start, min(cap, n - start)) for start in range(0, n, cap)]
    if len(slices) > 4096:
        raise ValueError("too many FC slices")
    payload = bytearray(align512(len(slices) * DESCRIPTOR.size))
    for index, (start, count) in enumerate(slices):
        payload.extend(b"\0" * (align512(len(payload)) - len(payload)))
        offset = len(payload)
        DESCRIPTOR.pack_into(payload, index * DESCRIPTOR.size, start, count, offset)
        # Each inner tile row has N/32 column tiles. A slice is a run of
        # complete 512-byte tiles, so there is no nibble permutation here.
        for row_tile in range(k // 32):
            begin = (row_tile * (n // 32) + start // 32) * 512
            payload.extend(source[begin:begin + (count // 32) * 512])
        payload.extend(source[wh_size + start * 4:wh_size + (start + count) * 4])
        payload.extend(struct.pack(f"<{count}i", *(int(v) for v in columns[start:start + count])))
        if len(payload) > MAX_PAYLOAD:
            raise ValueError("prepared payload exceeds 256 MiB")
    return payload, len(slices)


def encode_weight(source, source_hash, weight):
    k, n = weight["K"], weight["N"]
    payload, count = make_payload(source, k, n)
    header = HEADER.pack(MAGIC, 1, count, weight["file_offset"], len(source),
                         source_hash, k, n, len(payload), fnv1a64(payload))
    return header + payload


def verify_weight(path, source_hash, weight):
    """Validate layout/bounds and both identity checks before accepting bytes."""
    with open(path, "rb") as stream:
        header = stream.read(HEADER.size)
        if len(header) != HEADER.size:
            raise ValueError("truncated prepared header")
        magic, version, count, off, size, saved_hash, k, n, length, checksum = HEADER.unpack(header)
        if magic != MAGIC or version != 1:
            raise ValueError("unsupported prepared header")
        if (off, k, n, size, saved_hash) != (weight["file_offset"], weight["K"],
                                           weight["N"], weight_bytes(weight["K"], weight["N"]), source_hash):
            raise ValueError("prepared source identity or shape mismatch")
        cap = fc_slice_cols(k)
        if count != (n + cap - 1) // cap or count > 4096:
            raise ValueError("invalid prepared slice count")
        if length > MAX_PAYLOAD or path.stat().st_size != HEADER.size + length:
            raise ValueError("invalid prepared payload length")
        payload = stream.read(length)
        if len(payload) != length or fnv1a64(payload) != checksum:
            raise ValueError("prepared payload checksum mismatch")
    if len(payload) < align512(count * DESCRIPTOR.size):
        raise ValueError("truncated prepared descriptor table")
    cursor = align512(count * DESCRIPTOR.size)
    for index in range(count):
        start, columns, offset = DESCRIPTOR.unpack_from(payload, index * DESCRIPTOR.size)
        expected_start = index * cap
        expected_columns = min(cap, n - expected_start)
        if (start, columns, offset) != (expected_start, expected_columns, cursor):
            raise ValueError("invalid prepared slice descriptor")
        end = offset + weight_bytes(k, columns)
        if end > len(payload):
            raise ValueError("prepared slice exceeds payload bounds")
        cursor = align512(end) if index + 1 < count else end
    if cursor != len(payload):
        raise ValueError("unexpected prepared trailing bytes")
    return count


def prepare(model, manifest, output, verify_only=False):
    model, manifest, output = Path(model), Path(manifest), Path(output)
    weights = load_manifest(manifest, model.stat().st_size)
    if verify_only:
        with model.open("rb") as stream:
            for weight in weights:
                _, checksum = read_source(stream, weight)
                verify_weight(output / weight_name(weight), checksum, weight)
        return len(weights)
    if output.exists():
        raise ValueError("output already exists; choose a new directory or use --verify-only")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Publish the entire set only after every source entry was validated.
    with tempfile.TemporaryDirectory(prefix=".nntrwh-", dir=output.parent) as temporary:
        stage = Path(temporary) / "weights"
        stage.mkdir()
        with model.open("rb") as stream:
            for index, weight in enumerate(weights):
                source, checksum = read_source(stream, weight)
                data = encode_weight(source, checksum, weight)
                (stage / weight_name(weight)).write_bytes(data)
                print(f"[{index + 1}/{len(weights)}] {weight_name(weight)}: {len(data)} bytes", flush=True)
        os.rename(stage, output)
    return len(weights)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    try:
        count = prepare(args.model, args.manifest, args.output, args.verify_only)
    except (OSError, ValueError, KeyError, struct.error) as error:
        parser.exit(1, f"prepare_fc_wh: {error}\n")
    print(f"{'Verified' if args.verify_only else 'Prepared'} {count} weights in {args.output}")


if __name__ == "__main__":
    main()
