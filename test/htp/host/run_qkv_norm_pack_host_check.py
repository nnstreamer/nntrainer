#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify published FP32 norms and quantized AH packing with HVX emulation.

No device or Hexagon SDK is required. This checks aliases, tails, and rounding
boundaries against the separate kernels; silicon arithmetic needs device tests.
"""

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--sanitize", action="store_true")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    root = args.repo_root.resolve() if args.repo_root else here.parents[2]
    backend = root / "nntrainer/tensor/htp_backend"
    host = root / "test/htp/host"
    flags = ["-std=c11", "-O1" if args.sanitize else "-O2", "-Wall",
             "-Wextra", "-Werror", "-Wno-unused-parameter",
             "-ffp-contract=off", "-fno-strict-aliasing"]
    env = os.environ.copy()
    if args.sanitize:
        flags += ["-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-g1"]
        env["ASAN_OPTIONS"] = env.get("ASAN_OPTIONS", "") + ":halt_on_error=1"
        env["UBSAN_OPTIONS"] = env.get("UBSAN_OPTIONS", "") + ":halt_on_error=1"
    with tempfile.TemporaryDirectory(prefix="nntr-qkv-norm-") as path:
        binary = Path(path) / "qkv_norm_pack_host_check"
        command = shlex.split(os.environ.get("CC", "cc")) + flags
        command += ["-I" + str(p) for p in
                    (host / "hvx_emu", host / "stub", backend.parent,
                     backend / "hvx")]
        command += [str(here / "qkv_norm_pack_host_check.c"),
                    str(backend / "hvx/hvx_quant_u8.c"),
                    str(backend / "hvx/hvx_rmsnorm_rows_f32.c"),
                    "-lm", "-o", str(binary)]
        subprocess.run(command, check=True)
        subprocess.run([str(binary)], check=True, env=env)


if __name__ == "__main__":
    main()
