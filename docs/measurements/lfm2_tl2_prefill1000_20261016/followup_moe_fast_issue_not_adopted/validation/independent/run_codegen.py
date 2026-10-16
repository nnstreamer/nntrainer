#!/usr/bin/env python3
"""Independently compile frozen complete MoE TUs and assemble their v81 output."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--author", type=Path, required=True)
p.add_argument("--frozen", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
commands = {}

for variant in ("baseline", "candidate"):
    original = json.loads((a.frozen / "codegen" /
                           f"{variant}-command.json").read_text())
    command = []
    for word in original:
        if word == str(a.author / f"{variant}.s"):
            word = str(a.output / f"{variant}.s")
        elif word.startswith(str(a.author / variant) + "/"):
            word = str(a.frozen / variant) + word[len(str(a.author / variant)):]
        command.append(word)
    compiler = command[0]
    commands[variant] = [
        command,
        [compiler, "-mv81", "-mhvx", "-mhvx-length=128B", "-mhmx",
         "-fPIC", "-c", str(a.output / f"{variant}.s"), "-o",
         str(a.output / f"{variant}.o")],
    ]

(a.output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")


def run_variant(variant):
    for index, command in enumerate(commands[variant]):
        with (a.output / f"{variant}-{index}.log").open("w") as log:
            rc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
        if rc:
            raise RuntimeError(f"{variant} command {index} failed: {rc}")
    frames = [line for line in (a.output / f"{variant}.su").read_text().splitlines()
              if "hexkl_mm_u8i4_moe_layer_run" in line or
              "lfm2_moe_issue_aligned" in line]
    obj = a.output / f"{variant}.o"
    return {"variant": variant, "frames": frames,
            "object_sha256": hashlib.sha256(obj.read_bytes()).hexdigest()}


with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    results = list(executor.map(run_variant, commands))
(a.output / "result.json").write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps(results, indent=2))
