#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check prepared router leases, publication errors, and exact Q8 packing."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import tempfile


def function(source, marker):
    """Read an actual top-level C function, whose final brace starts a line."""
    start = source.index(marker)
    end = source.index("\n}", start) + 2
    return source[start:end] + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sanitize", action="store_true",
                        help="enable AddressSanitizer and UndefinedBehaviorSanitizer")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    root = here.parents[2]
    backend = root / "nntrainer/tensor/htp_backend"
    source = (here.parent / "nntr_hvx_mm_u8i4.c").read_text()
    start = source.index("/* Bounded internal byte packing")
    last = source.index("int nntr_hvx_router_q4_prefill_norm_prepare(", start)
    end = source.index("\n}", last) + 2
    actual = function(source, "static int check_moe_layer_args(")
    actual += function(source, "static int check_moe_row_totals(")
    actual += source[start:end] + "\n"
    actual += function(source, "int nntr_hvx_router_q4_prefill(")
    layer = (root / "Applications/CausalLM/models/lfm2_moe/lfm2_moe_layer.cpp").read_text()
    scope_start = layer.index("class MoeInputReuseScope {")
    scope_end = layer.index("\n};", scope_start) + 3
    flags = ["-O1", "-g", "-Wall", "-Wextra", "-Werror", "-ffp-contract=off"]
    if args.sanitize:
        flags += ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
    with tempfile.TemporaryDirectory(prefix="nntr-prepared-router-") as name:
        work = Path(name)
        (work / "prepared_router_under_test.inc").write_text(actual)
        (work / "prepared_scope_under_test.inc").write_text(layer[scope_start:scope_end])
        runtime_env = os.environ.copy()
        if args.sanitize:
            runtime_env["UBSAN_OPTIONS"] = runtime_env.get("UBSAN_OPTIONS", "") + ":halt_on_error=1"
        check = work / "prepared_router"
        includes = [work, here / "hvx_emu", here / "stub", backend.parent,
                    backend, backend / "hmx", backend / "hvx"]
        subprocess.run(shlex.split(os.environ.get("CC", "cc")) +
                       ["-std=gnu11", "-Wno-unknown-pragmas"] + flags +
                       ["-I" + str(p) for p in includes] +
                       [str(here / "prepared_router_host_check.c"),
                        str(backend / "hvx/hvx_quant_u8.c"),
                        str(backend / "hvx/hvx_rmsnorm_rows_f32.c"),
                        "-lm", "-o", str(check)], check=True)
        subprocess.run([str(check)], check=True, env=runtime_env)
        lease = work / "prepared_lease"
        subprocess.run(shlex.split(os.environ.get("CXX", "c++")) +
                       ["-std=c++17", "-pthread"] + flags +
                       ["-I" + str(work), "-I" + str(backend),
                        str(here / "prepared_router_lease_host_check.cpp"),
                        "-o", str(lease)], check=True)
        subprocess.run([str(lease)], check=True, env=runtime_env)


if __name__ == "__main__":
    main()
