#!/usr/bin/env python3
# @brief Offline, lossless QS2CX_WH -> decode LUT rows (requires NumPy).
# SPDX-License-Identifier: Apache-2.0
"""Offline, lossless QS2CX_WH -> decode LUT rows (requires NumPy).

python3 tools/htp/prepare_qs2_lut.py --model model.bin \
    --manifest experts.json --output prepared_experts

Manifest: {"schema_version":1,"dtype":"QS2CX_WH","model_size":...,
"weights":[{"file_offset":...,"K":2048,"N":3072}, ...]}.
Alternatively repeat --weight OFFSET:K:N for individual weights.
For an existing model, run CausalLM with NNTR_HTP_QS2_EXPORT=1 and the
expert cache enabled, save stdout, and pass --manifest-log that_log.txt.
All virtual expert descriptors are exported during preload, not just hits.
Use --verify-only to check existing artifacts against the original model.

Deploy the directory, set NNTR_HTP_QS2_PREPARED_DIR to its absolute path,
NNTR_HTP_QS2_VTCM=1 and NNTR_HTP_QS2_GEMV=7. Mode 6 isolates dense LUTs
with runtime repacking. Prefill and fallbacks convert LUT tiles to WH in VTCM.
Version 2 stores one LUT code image in the expert arena. Prefill/fallback
converts tiles to WH in VTCM. The first use validates source and payload CRCs;
unchanged file identities reuse validation, and subsequent misses read only
the prepared image directly into the shared arena. After deployment, run
validate_qs2_lut model.bin prepared_experts on the target to validate all
artifacts once and write a .nntri2.validated receipt. Matching local file
identities reuse that verification across inference processes. Missing or
stale receipts fall back to full checks. Re-run validation after copying or
replacing files. Do not modify model/artifact files during inference.

Files OFFSET_K_N.nntri2 contain a 64-byte little-endian header followed by
K*N/4 bytes [K/32][N/128][8][128], then original palette/scales/sums.
See htp_prepared_qs2.h for header fields; --legacy emits the former v1 format.
No weight value, palette, scale, activation precision or zero point changes.
"""

import argparse
import json
import os
import re
from pathlib import Path
import struct
import tempfile
import zlib

import numpy as np

HEADER = struct.Struct("<8sIIQQQIIQQ")
MAX_BYTES = 256 << 20


def coordinates(k, n, t, h):
    """Independent WH coordinate mapping for one of four 2-bit fields."""
    row = 2 * t + h % 2 + (h // 2) * 16
    j = np.arange(n, dtype=np.uint64)
    col = (j // 128) * 128 + (j % 4) * 32 + (j % 128) // 4
    old = (row // 8) * 128 + (col % 32) * 4 + row % 4
    off = ((np.arange(k // 32, dtype=np.uint64)[:, None] * (n // 32)
            + col[None, :] // 32) * 256
           + (old // 256) * 128 + old % 128)
    shift = 2 * ((row // 4) % 2) + 4 * ((row // 8) % 2)
    return off, shift


def pack(source, k, n):
    wh = np.frombuffer(source, dtype=np.uint8, count=k*n//4)
    result = np.zeros((k//32, n//128, 8, 128), dtype=np.uint8)
    for t in range(8):
        for h in range(4):
            off, shift = coordinates(k, n, t, h)
            codes = ((wh[off] >> shift) & 3) << (2*h)
            result[:, :, t, :] |= codes.reshape(k//32, n//128, 128)
    return result.tobytes()


def unpack(payload, k, n):
    data = np.frombuffer(payload, dtype=np.uint8).reshape(k//32, n//128, 8, 128)
    wh = np.zeros(k*n//4, dtype=np.uint8)
    for t in range(8):
        for h in range(4):
            off, shift = coordinates(k, n, t, h)
            codes = (data[:, :, t, :].reshape(k//32, n) >> (2*h)) & 3
            wh[off] |= codes << shift
    return wh.tobytes()


def encode(source, offset, k, n, legacy=False):
    expected = k*n//4 + 4 + 8*n
    if len(source) != expected:
        raise ValueError("truncated weight")
    palette = np.frombuffer(source, dtype=np.int8, count=4, offset=k*n//4)
    tail = np.frombuffer(source, dtype="<f4", count=2*n, offset=k*n//4+4)
    if np.any(palette < -8) or np.any(palette > 7) or not np.all(np.isfinite(tail)):
        raise ValueError("invalid palette or metadata")
    payload = pack(source, k, n)
    if unpack(payload, k, n) != source[:k*n//4]:
        raise ValueError("layout roundtrip failed")
    if not legacy:
        payload += source[k*n//4:]
    version = 1 if legacy else 2
    header = HEADER.pack(b"NNTRI201" if legacy else b"NNTRI202", version, version,
                         offset, expected, zlib.crc32(source), k, n,
                         len(payload), zlib.crc32(payload))
    return header + payload


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--manifest-log", type=Path)
    parser.add_argument("--weight", action="append", default=[], metavar="OFFSET:K:N")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--legacy", action="store_true", help="emit previous dual-layout v1 artifacts for comparison only")
    args = parser.parse_args()
    size = args.model.stat().st_size
    weights = []
    if args.manifest:
        manifest = json.loads(args.manifest.read_text())
        if (manifest.get("schema_version"), manifest.get("dtype"), manifest.get("model_size")) != (1, "QS2CX_WH", size):
            raise ValueError("manifest version, dtype or model size mismatch")
        weights.extend(manifest["weights"])
    if args.manifest_log:
        entries = set(re.findall(r"^\[QS2-WEIGHT\] (\d+) (\d+) (\d+)$",
                                 args.manifest_log.read_text(), re.MULTILINE))
        weights.extend(dict(file_offset=int(off), K=int(k), N=int(n))
                       for off, k, n in sorted(entries, key=lambda row: int(row[0])))
    for spec in args.weight:
        offset, k, n = map(int, spec.split(":"))
        weights.append(dict(file_offset=offset, K=k, N=n))
    if not weights:
        parser.error("provide --manifest, --manifest-log or --weight")
    ranges = []
    for w in weights:
        off, k, n = (w[key] for key in ("file_offset", "K", "N"))
        if any(type(v) is not int for v in (off, k, n)) or off < 0 or k <= 0 or n <= 0 or k % 32 or n % 128 or max(k,n) >= 2**32:
            raise ValueError("invalid offset or shape; K%32=N%128=0 required")
        length = k*n//4 + 4 + 8*n
        if k*n//4 > MAX_BYTES or off + length > size:
            raise ValueError("weight exceeds size limits")
        ranges.append((off, off+length))
    end = 0
    for off, last in sorted(ranges):
        if off < end:
            raise ValueError("overlapping or duplicate weight ranges")
        end = last
    args.output.mkdir(parents=True, exist_ok=True)
    with args.model.open("rb") as model:
        for w in weights:
            off, k, n = (w[key] for key in ("file_offset", "K", "N"))
            model.seek(off)
            artifact = encode(model.read(k*n//4+4+8*n), off, k, n, args.legacy)
            path = args.output / f"{off}_{k}_{n}.nntri2"
            if args.verify_only:
                if path.read_bytes() != artifact:
                    raise ValueError(f"artifact mismatch: {path}")
            else:
                name = None
                try:
                    with tempfile.NamedTemporaryFile(dir=args.output, delete=False) as stream:
                        name = stream.name
                        stream.write(artifact)
                    os.replace(name, path)
                finally:
                    if name and os.path.exists(name):
                        os.unlink(name)
            print(f"verified {path} ({k*n//4} code bytes, {len(artifact)-64-k*n//4} metadata bytes)", flush=True)


if __name__ == "__main__":
    main()
