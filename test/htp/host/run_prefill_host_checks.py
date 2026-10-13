#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check prefill RPC dispatch and generation-tail scope without a device."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sanitize", action="store_true",
                        help="enable AddressSanitizer and UndefinedBehaviorSanitizer")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    root = here.parents[2]
    backend = root / "nntrainer/tensor/htp_backend"
    source = (here.parent / "nntr_hvx_small_ops.c").read_text()
    start = source.index("typedef struct {\n  uint32_t rows, qw, kw;")
    end = source.index("\nint nntr_hvx_inference_math_f32(", start)
    flags = ["-O1", "-g", "-Wall", "-Wextra", "-Werror", "-ffp-contract=off"]
    if args.sanitize:
        flags += ["-fsanitize=address,undefined", "-fno-omit-frame-pointer"]
    with tempfile.TemporaryDirectory(prefix="nntr-prefill-check-") as name:
        work = Path(name)
        (work / "rope_prefill_under_test.inc").write_text(source[start:end])
        rope = work / "rope_prefill_host_check"
        subprocess.run(shlex.split(os.environ.get("CC", "cc")) +
                       ["-std=gnu11"] + flags +
                       ["-I" + str(work), "-I" + str(backend),
                        str(here / "rope_prefill_host_check.c"), "-lm",
                        "-o", str(rope)], check=True)
        subprocess.run([str(rope)], check=True)
        tail = work / "generation_prefill_tail_host_check"
        subprocess.run(shlex.split(os.environ.get("CXX", "c++")) +
                       ["-std=c++17", "-pthread"] + flags +
                       ["-I" + str(root / "Applications/CausalLM/layers"),
                        str(here / "generation_prefill_tail_host_check.cpp"),
                        str(here / "generation_prefill_tail_host_peer.cpp"),
                        "-o", str(tail)], check=True)
        subprocess.run([str(tail)], check=True)


if __name__ == "__main__":
    main()
