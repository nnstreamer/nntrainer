#!/usr/bin/env python3
"""Exercise whole production native MoE before/after with scalar primitives."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

p = argparse.ArgumentParser(description=__doc__)
for name in ("repo", "source", "reference", "output"):
    p.add_argument("--" + name, type=Path, required=True)
p.add_argument("--sanitize", action="store_true")
a = p.parse_args()
here = Path(__file__).parent
backend = a.repo / "nntrainer/tensor/htp_backend"
host = a.repo / "test/htp/host"
relative = Path("nntrainer/tensor/htp_backend/hmx/hexkl_mm_u8i4_moe.c")
a.output.mkdir(parents=True, exist_ok=True)
commands = []
hashes = {}
flags = ["-std=gnu11", "-O2", "-g", "-Wall", "-Wextra", "-Werror",
         "-Wno-unused-parameter", "-ffunction-sections", "-fdata-sections"]
if a.sanitize:
    flags += ["-fsanitize=address,undefined", "-fno-sanitize-recover=all"]
for inc in (host, host / "stub", host / "standin", host.parent,
            backend.parent, backend, backend / "hmx", backend / "hvx"):
    flags += ["-I", str(inc)]
env = os.environ.copy()
env.update(ASAN_OPTIONS="detect_leaks=1:halt_on_error=1", UBSAN_OPTIONS="halt_on_error=1")


def execute(command, log):
    commands.append(command)
    (a.output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    with log.open("w") as output:
        rc = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT,
                            env=env).returncode
    if rc:
        raise RuntimeError(f"Command failed {rc}: {log}\n" + log.read_text())


common = []
for source in (host / "standin/hvx_scalar.c", backend / "hmx/hexkl_dma_trace.c",
               backend / "hvx/hvx_expand_i2i4.c", backend / "hvx/hvx_gemv_u8tl2.c"):
    obj = a.output / (source.stem + ".o")
    extra = []
    if a.sanitize and source.name == "hvx_scalar.c":
        # Pre-existing swiglu_det.h signed-negative exponent shift in the
        # scalar stand-in. Both actual MoE TUs retain all sanitizer classes.
        extra += ["-fno-sanitize=shift-base"]
    execute(["gcc", *flags, *extra, "-c", str(source), "-o", str(obj)],
            a.output / (source.stem + "-build.log"))
    common.append(obj)

export = """
int review_guard(int in, uint32_t M, uint32_t K, uint32_t I, uint32_t N,
                 uint32_t E, uint32_t nt, uint32_t nr, uint32_t na, uint32_t act,
                 const hexkl_moe_layout *layout, uint32_t arena) {
  return lfm2_moe_hmx_issue_ok(in, M, K, I, N, E, nt, nr, na, act, layout, arena);
}
"""
for name, source in (("baseline", a.reference), ("candidate", a.source)):
    work = a.output / name
    (work / "hmx").mkdir(parents=True, exist_ok=True)
    code = (source / relative).read_text()
    hashes[name] = hashlib.sha256(code.encode()).hexdigest()
    definitions = []
    if name == "candidate":
        pos = code.index("lfm2_moe_issue_aligned(")
        brace = code.index("{", pos)
        code = code[:brace + 1] + "\n  review_leaf_observe(k_tiles, wstep);\n" + code[brace + 1:]
        code = "void review_leaf_observe(unsigned, unsigned);\n" + code + export
        definitions = ["-DREVIEW_CANDIDATE=1"]
    (work / "hmx/kernel.c").write_text(code)
    shutil.copy2(here / "hexkl_hmx_mm.h", work / "hmx/hexkl_hmx_mm.h")
    shutil.copy2(source / relative.parent.parent / "htp_tl2_opts.h", work)
    objects = []
    for kind, path in (("kernel", work / "hmx/kernel.c"),
                       ("fixture", here / "whole_check.c")):
        obj = work / (kind + ".o")
        extra = (["-Wno-error=missing-field-initializers", "-Wno-error=maybe-uninitialized"]
                 if kind == "fixture" else [])
        execute(["gcc", *flags, *definitions, *extra, "-c", str(path), "-o", str(obj)],
                work / (kind + "-build.log"))
        objects.append(obj)
    command = ["gcc", *flags, *map(str, objects + common), "-Wl,--gc-sections",
               "-Wl,--wrap=hexkl_micro_hmx_acc_clear_int32",
               "-Wl,--wrap=hexkl_micro_hmx_mm_u8i4",
               "-Wl,--wrap=hexkl_micro_hmx_acc_read_int32", "-lm", "-o",
               str(work / "check")]
    execute(command, work / "link.log")
    execute([str(work / "check"), str(work / "output.bin")], work / "run.log")
    print(name, (work / "run.log").read_text().splitlines()[-1], flush=True)
before = (a.output / "baseline/output.bin").read_bytes()
after = (a.output / "candidate/output.bin").read_bytes()
assert before == after, "Whole native MoE FP32 output mismatch"
assert (a.output / "baseline/run.log").read_bytes() == (a.output / "candidate/run.log").read_bytes(), \
    "Whole MoE packet/control events differ"
(a.output / "result.json").write_text(json.dumps({
    "pass": True, "whole_cases": 11, "guard_cases": 25,
    "source_sha256": hashes, "output_bytes": len(after),
    "output_sha256": hashlib.sha256(after).hexdigest(),
    "limits": ["HMX/HVX/quant use the existing scalar stand-ins",
               "Host does not enable Hexagon-only i24 staging",
               "Only pre-existing scalar exponent shift sanitizer is disabled",
               "No device/cache/timing claim"]
}, indent=2) + "\n")
print("MOE_ISSUE full-caller output and packet/control equality PASS", flush=True)
