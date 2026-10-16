#!/usr/bin/env python3
"""Deliberately weaken guards/change packet order and require actual assertions."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--checked", type=Path, required=True,
               help="Completed run_check.py output directory")
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
command = json.loads((a.checked / "commands.json").read_text())[-1]
original = a.checked / "candidate"
source = (original / "hmx/kernel.c").read_text()
changes = {
    "m_guard": ("M <= 1u || ", "", "review_align_queries>=packets"),
    "k_guard": ("K != 2048u || ", "", "review_align_queries>=packets"),
    "affine_guard": ("!h->w_offset ||", "0 ||", "review_align_queries>=packets"),
    "base_alignment": ("!hexkl_hmx_mm_aligned(vtcm, vtcm)", "(hexkl_hmx_mm_aligned(vtcm, vtcm), 0)",
                       "((uintptr_t)a&2047u)==0"),
    "weight_alignment": ("(wbuf[0] % HEXKL_HMX_WEIGHTS_ALIGNMENT) == 0u", "1",
                         "!review_direct_guard"),
    "weight_extent": ("(uint64_t)wbuf[1] + wb_max <= arena", "1",
                      "!review_direct_guard"),
    "shape_guard": ("if (!((nh == 1u &&", "if (0 && !((nh == 1u &&",
                    "review_align_queries>=packets"),
    "reverse_k_packets": ("(size_t)kt * wstep", "(size_t)(k_tiles - 1u - kt) * wstep",
                          "a==gb+ao&&w==gb+wo"),
}
results = {}
env = os.environ.copy()
env.update(ASAN_OPTIONS="detect_leaks=1:halt_on_error=1", UBSAN_OPTIONS="halt_on_error=1")
for name, (old, new, assertion) in changes.items():
    begin = source.index("static int lfm2_fc_hmx_issue_ok(") if name != "reverse_k_packets" else 0
    end = source.index("\nint hexkl_mm_u8i4_layer_run", begin)
    region = source[begin:end]
    assert region.count(old) == 1, (name, region.count(old))
    out = a.output / name
    (out / "hmx").mkdir(parents=True, exist_ok=True)
    changed = source[:begin] + region.replace(old, new) + source[end:]
    (out / "hmx/kernel.c").write_text(changed)
    shutil.copy2(original / "hmx/hexkl_hmx_mm.h", out / "hmx")
    shutil.copy2(original / "htp_tl2_opts.h", out)
    cc = [part.replace(str(original), str(out)) for part in command]
    built = subprocess.run(cc, capture_output=True, text=True)
    (out / "command.json").write_text(json.dumps(cc, indent=2) + "\n")
    (out / "build.log").write_text(built.stdout + built.stderr)
    if built.returncode:
        raise RuntimeError(name + " did not compile: " + built.stderr)
    run = subprocess.run([str(out / "check")], capture_output=True, text=True, env=env)
    (out / "run.log").write_text(run.stdout + run.stderr)
    assert run.returncode != 0, name + " unexpectedly passed"
    compact = "".join(run.stderr.split())
    assert "Assertion" in run.stderr and assertion in compact, (name, run.stderr)
    results[name] = {"rejected": True, "returncode": run.returncode,
                     "assertion": assertion,
                     "source_sha256": hashlib.sha256(changed.encode()).hexdigest()}
    print(name, "expected assertion PASS", flush=True)
(a.output / "result.json").write_text(json.dumps(results, indent=2) + "\n")
print(f"FC_ISSUE_MUTATIONS {len(results)} detected PASS", flush=True)
