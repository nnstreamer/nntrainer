#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual composite DSP, ARM bridge and MHA dispatch control flow.

The old numerical kernels and platform/graph boundaries are fixture callbacks.
This checks state, aliases and error ordering, not HVX numerical parity. No
device, SDK, implementation copy, or generated FastRPC stub is required.
"""

import argparse
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile

from run_expert_host_staging_checks import body_end, extract_method


def extract_function(source, name):
    match = re.search(r"(?m)^(?:int|bool|void|HtpPrefillAttentionResult)\s+" +
                      re.escape(name) + r"\(", source)
    if not match:
        raise ValueError("Missing function: " + name)
    opening = source.index("{", match.end())
    return source[match.start():body_end(source, opening)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--sanitize", action="store_true")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    root = args.repo_root.resolve() if args.repo_root else here.parents[2]
    backend = root / "nntrainer/tensor/htp_backend"
    arm = (backend / "htp_compute_ops.cpp").read_text()
    dsp = (root / "test/htp/nntr_hvx_attn_q.c").read_text()
    mha = (root / "Applications/CausalLM/layers/mha_core.cpp").read_text()
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    flags = ["-std=c++17", "-O1" if args.sanitize else "-O2", "-Wall",
             "-Wextra", "-Werror", "-Wno-unused-parameter", "-pthread",
             "-I" + str(backend)]
    env = os.environ.copy()
    if args.sanitize:
        flags += ["-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-g1"]
        env["ASAN_OPTIONS"] = env.get("ASAN_OPTIONS", "") + ":halt_on_error=1:detect_leaks=1"
        env["UBSAN_OPTIONS"] = env.get("UBSAN_OPTIONS", "") + ":halt_on_error=1"
    with tempfile.TemporaryDirectory(prefix="nntr-attn-composite-") as path:
        tmp = Path(path)
        wrapper = extract_function(dsp, "nntr_hvx_attn_q8_prefill_prepare")
        # Only rename the exported symbol, so the ARM boundary can inject an
        # uncertain transport failure after the actual DSP body has executed.
        (tmp / "dsp.inc").write_text(wrapper.replace(
            "nntr_hvx_attn_q8_prefill_prepare(", "fixture_dsp_composite(", 1))
        (tmp / "arm.inc").write_text(extract_method(arm, "prefillQ8Attention"))
        (tmp / "bridge.inc").write_text(extract_function(
            arm, "htp_rope_q8_attention_prefill"))
        helper = extract_function(mha, "MHACoreLayer::try_prefill_q8_composite")
        caller = extract_function(mha, "MHACoreLayer::one_batch_incremental_forwarding")
        # Preserve the real caller through both accelerator attempts. The CPU
        # continuation is outside this fixture and must never be reached.
        boundary = caller.index("  if (htpStrict())")
        caller = caller[:boundary] + '  throw std::runtime_error("CPU continuation");\n}'
        release = extract_function(mha, "MHACoreLayer::release_quantized_cache")
        (tmp / "mha.inc").write_text(helper + "\n" + release + "\n" + caller)
        binary = tmp / "attention_composite_host_check"
        subprocess.run(compiler + flags + ["-I" + str(tmp),
                       str(here / "attention_composite_host_check.cpp"),
                       "-o", str(binary)], check=True)
        subprocess.run([str(binary)], check=True, env=env, timeout=60)
        portable = tmp / "portable.cpp"
        portable.write_text('#include "htp_rope_prefill.h"\n'
                            'int main() { return int(nntrainer::htp_rope_q8_attention_prefill('
                            '0, 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr)); }\n')
        subprocess.run(compiler + ["-std=c++17", "-Wall", "-Wextra", "-Werror",
                       "-I" + str(backend), "-fsyntax-only", str(portable)], check=True)
        print("PASS: non-HTP composite header")
        subprocess.run(compiler + ["-std=c++17", "-Wall", "-Wextra", "-Werror",
                       "-U__linux__", "-U__ANDROID__", "-fsyntax-only",
                       str(here / "attention_composite_host_check.cpp")], check=True)
        print("PASS: environment-injection fixture platform guard")


if __name__ == "__main__":
    main()
