#!/usr/bin/env python3
"""Compare full production callers using the existing scalar math stand-ins."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

p = argparse.ArgumentParser(description=__doc__)
for arg in ("repo", "source", "reference", "output"):
    p.add_argument("--" + arg, type=Path, required=True)
p.add_argument("--sanitize", action="store_true")
a = p.parse_args()
here = Path(__file__).parent
backend = a.repo / "nntrainer/tensor/htp_backend"
host = a.repo / "test/htp/host"
relative = Path("nntrainer/tensor/htp_backend/hmx/hexkl_mm_u8i4_dma.c")
commands = []
hashes = {}
for name, source in (("baseline", a.reference), ("candidate", a.source)):
    out = a.output / name
    (out / "hmx").mkdir(parents=True, exist_ok=True)
    payload = (source / relative).read_bytes()
    hashes[name] = hashlib.sha256(payload).hexdigest()
    (out / "hmx/kernel.c").write_bytes(payload)
    shutil.copy2(here / "hexkl_hmx_mm.h", out / "hmx/hexkl_hmx_mm.h")
    shutil.copy2(source / relative.parent.parent / "htp_tl2_opts.h", out)
    command = ["gcc", "-std=c99", "-O2", "-g", "-Wall", "-Wextra", "-Werror",
               "-Wno-unused-parameter", "-ffunction-sections", "-fdata-sections"]
    if a.sanitize:
        command += ["-fsanitize=address,undefined", "-fno-sanitize-recover=all"]
    if name == "candidate":
        command += ["-DREVIEW_CANDIDATE=1"]
    for inc in (host, host / "stub", host / "standin", backend.parent,
                backend / "hmx", backend / "hvx"):
        command += ["-I", str(inc)]
    command += [str(here / "numeric_check.c"), str(host / "hvx_scalar_stubs.c"),
                str(host / "standin/hvx_scalar.c"), str(out / "hmx/kernel.c"),
                "-Wl,--gc-sections", "-lm", "-o", str(out / "check")]
    commands.append(command)
    built = subprocess.run(command, capture_output=True, text=True)
    (out / "build.log").write_text(built.stdout + built.stderr)
    if built.returncode:
        raise RuntimeError(built.stdout + built.stderr)
    env = os.environ.copy()
    env.update(ASAN_OPTIONS="detect_leaks=1:halt_on_error=1",
               UBSAN_OPTIONS="halt_on_error=1")
    run = subprocess.run([str(out / "check"), str(out / "output.bin")],
                         capture_output=True, text=True, env=env)
    (out / "run.log").write_text(run.stdout + run.stderr)
    if run.returncode:
        raise RuntimeError(run.stdout + run.stderr)
    print(name, run.stdout.splitlines()[-1], flush=True)
before = (a.output / "baseline/output.bin").read_bytes()
after = (a.output / "candidate/output.bin").read_bytes()
assert before == after, "FP32 scalar output bytes differ"
(a.output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
(a.output / "result.json").write_text(json.dumps({
    "pass": True, "cases": 7, "source_sha256": hashes,
    "output_bytes": len(after), "output_sha256": hashlib.sha256(after).hexdigest(),
    "limitations": ["Whole caller with scalar HMX/HVX/quant stand-ins",
                    "Does not replace device arithmetic/cache/concurrency tests"]
}, indent=2) + "\n")
print("Full-caller scalar FP32 byte equality PASS", flush=True)
