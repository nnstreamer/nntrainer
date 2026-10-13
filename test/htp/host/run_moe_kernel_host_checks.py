#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check actual MoE expansion, worker scheduling, dequantization and SiLU.

The HVX lane model checks arithmetic and memory behavior, not DSP instruction
timing or all QF32 edge encodings. Device bit parity remains a separate gate.
No Hexagon SDK or stored historical implementation is needed.
"""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import tempfile


def worker_source(source, marker, end_marker):
    """Take the current private worker and its immediately preceding context."""
    function = source.index(marker)
    start = source.rfind("typedef struct {", 0, function)
    if start < 0:
        raise ValueError("worker context not found: " + marker)
    end = source.index(end_marker, function)
    return source[start:end]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sanitize", action="store_true",
                        help="enable AddressSanitizer and UndefinedBehaviorSanitizer")
    parser.add_argument("--repo-root", type=Path,
                        help="source tree to check (defaults to this checkout)")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    root = args.repo_root.resolve() if args.repo_root else here.parents[2]
    host = root / "test/htp/host"
    backend = root / "nntrainer/tensor/htp_backend"
    hvx = backend / "hvx"
    source = (backend / "hmx/hexkl_mm_u8i4_moe.c").read_text()
    flags = ["-std=gnu11", "-O1", "-g1", "-Wall", "-Wextra", "-Werror",
             "-Wno-unknown-pragmas", "-Wno-unused-parameter", "-ffp-contract=off",
             "-ffunction-sections", "-fdata-sections", "-pthread"]
    if args.sanitize:
        flags += ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
    compiler = shlex.split(os.environ.get("CC", "cc"))
    runtime_env = os.environ.copy()
    if args.sanitize:
        runtime_env["UBSAN_OPTIONS"] = (
            runtime_env.get("UBSAN_OPTIONS", "") + ":halt_on_error=1")
    with tempfile.TemporaryDirectory(prefix="nntr-moe-kernel-") as name:
        work = Path(name)
        # The older shared emulator assumes float-aligned unaligned vectors.
        # Expansion legally uses byte offsets: model their alignment accurately
        # without changing the shared emulator for unrelated existing tests.
        types = (host / "hvx_emu/hexagon_types.h").read_text()
        aligned = "typedef HVX_Vector HVX_UVector;"
        unaligned = "typedef HVX_Vector HVX_UVector __attribute__((aligned(1)));"
        if types.count(aligned) != 1 and types.count(unaligned) != 1:
            raise ValueError("review the current HVX_UVector emulation")
        (work / "hexagon_types.h").write_text(types.replace(aligned, unaligned))
        (work / "hvx_hexagon_protos.h").write_text(
            (host / "hvx_emu/hvx_hexagon_protos.h").read_text())
        (work / "moe_dn_under_test.inc").write_text(worker_source(
            source, "static void moe_dn_worker(",
            "/** @brief The gate_up epilogue job"))
        (work / "moe_expand_under_test.inc").write_text(worker_source(
            source, "static void moe_expand_unit(",
            "/**\n * @brief Queues a chunk"))
        includes = [work, here, host / "stub", backend.parent,
                    backend, backend / "hmx", hvx]
        common = compiler + flags + ["-I" + str(p) for p in includes]

        def object_file(stem, extra=()):
            obj = work / (stem + ".o")
            subprocess.run(common + list(extra) +
                           ["-c", str(hvx / (stem + ".c")), "-o", str(obj)],
                           check=True)
            return str(obj)

        expand = object_file("hvx_expand_i2i4", [
            "-D__hexagon__", "-include", str(here / "moe_kernel_emu.h")])
        dequant = object_file("hvx_dequant_i32")
        scatter = object_file("hvx_scale_add_f32")
        quant = object_file("hvx_quant_u8")
        unpack = object_file("hvx_gemv_u8i2_lut")
        cases = [
            (here / "moe_expand_host_check.c", [expand]),
            (here / "moe_schedule_host_check.c",
             [expand, dequant, scatter, quant, unpack]),
            (here / "moe_epilogue_host_check.c", [dequant, scatter]),
            (host / "dequant_planes_host_check.c", [dequant, scatter]),
        ]
        for fixture, objects in cases:
            binary = work / fixture.stem
            subprocess.run(common + [str(fixture)] + objects +
                           ["-Wl,--gc-sections", "-lm", "-o", str(binary)],
                           check=True)
            subprocess.run([str(binary)], check=True, env=runtime_env)


if __name__ == "__main__":
    main()
