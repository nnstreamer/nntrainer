#!/usr/bin/env python3
"""Reject damaged actual MoE scope/bounds/packet ordering via whole-caller checks."""
import argparse
import json
import os
from pathlib import Path
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--checks", type=Path, required=True,
               help="Successful run_whole.py output directory")
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
commands = json.loads((a.checks / "commands.json").read_text())
source = a.checks / "candidate/hmx/kernel.c"
original = source.read_text()
compile_cmd = next(c for c in commands if "-c" in c and str(source) in c)
link_cmd = next(c for c in commands if str(a.checks / "candidate/check") in c and "-o" in c)
mutations = {
    "alignment_guard": ("!hmx_inline ||", ""),
    "prefill_guard": ("M <= 1u ||", ""),
    "dimension_guard": ("K != 2048u ||", ""),
    "expert_upper_guard": ("n_experts > 64u ||", ""),
    "expert_lower_guard": ("n_experts == 0u ||", ""),
    "native_guard": ("n_tl2 != n_with_rows ||", ""),
    "activation_guard": ("act != HVX_GLU_SILU", "0"),
    "act_span": ("(uint64_t)L->act_off + act_bytes <= L->w_gu_off",
                 "((void)act_bytes, 1)"),
    "gu_span": ("(uint64_t)L->w_gu_off + gu_bytes <= L->w_dn_off",
                "((void)gu_bytes, 1)"),
    "dn_span": ("(uint64_t)L->w_dn_off + dn_bytes <= L->gate_off",
                "((void)dn_bytes, 1)"),
    "gate_mid_span": ("L->gate_off <= L->mid_off", "1"),
    "mid_span": ("(uint64_t)L->mid_off + 2u * mid_bytes <= L->result_off",
                 "((void)mid_bytes, 1)"),
    "arena_bound": ("L->result_off <= arena;", "1;"),
    "packet_order": ("a + (size_t)kt * HEXKL_HMX_ACTIVATION_ALIGNMENT,",
                     "a + (size_t)(k_tiles - 1u - kt) * HEXKL_HMX_ACTIVATION_ALIGNMENT,"),
}
env = os.environ.copy()
env.update(ASAN_OPTIONS="detect_leaks=1:halt_on_error=1",
           UBSAN_OPTIONS="halt_on_error=1")
results = []
for name, (old, new) in mutations.items():
    work = a.output / name
    work.mkdir(parents=True, exist_ok=True)
    # Guard edits are scoped to the new helper, not other shape checks.
    lo = original.index("lfm2_moe_issue_aligned(")
    hi = original.index("int hexkl_mm_u8i4_moe_layer_run(", lo)
    prefix = original[lo:hi]
    assert prefix.count(old) == 1, (name, prefix.count(old))
    changed = original[:lo] + prefix.replace(old, new, 1) + original[hi:]
    mutant = work / "kernel.c"
    mutant.write_text(changed)
    obj = work / "kernel.o"
    binary = work / "check"
    cc = [str(mutant) if x == str(source) else str(obj)
          if x == str(a.checks / "candidate/kernel.o") else x for x in compile_cmd]
    # Preserve the copied production-relative includes and test packet wrapper.
    cc[1:1] = ["-I", str(source.parent)]
    link = [str(obj) if x == str(a.checks / "candidate/kernel.o") else str(binary)
            if x == str(a.checks / "candidate/check") else x for x in link_cmd]
    (work / "commands.json").write_text(json.dumps([cc, link], indent=2) + "\n")
    for label, command in (("build", cc), ("link", link)):
        with (work / (label + ".log")).open("w") as log:
            rc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                env=env).returncode
        if rc:
            raise RuntimeError(f"{name} {label} failed: {work}")
    with (work / "run.log").open("w") as log:
        rc = subprocess.run([str(binary), str(work / "output.bin")], stdout=log,
                            stderr=subprocess.STDOUT, env=env).returncode
    out = (work / "run.log").read_text()
    assert rc != 0 and "Assertion" in out and "failed" in out, (name, rc, out)
    results.append({"mutation": name, "returncode": rc,
                    "assertion": next(line for line in out.splitlines() if "Assertion" in line)})
    print(name, "intended assertion PASS", flush=True)
(a.output / "result.json").write_text(json.dumps(results, indent=2) + "\n")
print(f"MOE_ISSUE {len(results)} intended-assertion mutants PASS", flush=True)
