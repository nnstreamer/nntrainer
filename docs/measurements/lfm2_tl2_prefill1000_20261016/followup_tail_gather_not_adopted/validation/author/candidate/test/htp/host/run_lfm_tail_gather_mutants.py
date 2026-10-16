#!/usr/bin/env python3
"""Reuse a completed gather-check build to kill bounded production mutants."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--checks", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    source, checks, out = (p.resolve() for p in
                           (args.source, args.checks, args.output))
    out.mkdir(parents=True, exist_ok=True)
    runner = source / "test/htp/host/run_lfm_tail_gather_checks.py"
    spec = importlib.util.spec_from_file_location("gather_checks", runner)
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    path = source / "nntrainer/tensor/htp_backend/hmx/hexkl_mm_u8i4_moe.c"
    text = path.read_text()
    build = json.loads((checks / "commands.json").read_text())
    mutants = [
        ("floor4", "d + ROUND_UP_U32(row_count[e], 4u)",
         "d + (row_count[e] & ~3u)", False),
        ("restore16", "d + ROUND_UP_U32(row_count[e], 4u)",
         "d + ROUND_UP_U32(row_count[e], 16u)", False),
        ("offset4", "const uint32_t first = u * HEXKL_MOE_PACK_UNIT_ROWS;",
         "const uint32_t first = u * HEXKL_MOE_PACK_UNIT_ROWS + 4u;", False),
        ("generic_skip", "if (c->skip && c->skip[u])",
         "if (c->skip && c->skip[u] == 1u)", False),
        ("expert65", "n_experts <= 64u && n_with_rows",
         "n_experts <= 65u && n_with_rows", False),
        ("mixed_guard", "n_tl2 == n_with_rows && !n_affine && act == HVX_GLU_SILU;",
         "n_tl2 != 0u && !n_affine && act == HVX_GLU_SILU;", False),
        ("wrong_dispatch", "lfm_prefill_pack ? moe_lfm_pack_bg_worker : hexkl_moe_pack_bg_worker;",
         "hexkl_moe_pack_bg_worker;", True),
    ]
    env = dict(os.environ, ASAN_OPTIONS="detect_leaks=1:halt_on_error=1",
               UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1")
    results = []
    for name, old, new, whole in mutants:
        assert old in text, (name, old)
        # The first occurrence is the dedicated native-prefill condition.
        mutated = text.replace(old, new, 1)
        d = out / name; d.mkdir(exist_ok=True)
        c = d / path.name; c.write_text(mutated)
        (d / "actual-tail-workers.inc").write_text(helpers.extract(mutated))
        if whole:
            cc = next(cmd[:] for cmd in build if str(path) in cmd)
            cc[cc.index(str(path))] = str(c)
            oldobj = str(checks / "candidate.o")
            cc[cc.index(oldobj)] = str(d / "candidate.o")
            link = next(cmd[:] for cmd in build if cmd[0] == "g++" and
                        str(checks / "whole-check") in cmd)
            link[link.index(oldobj)] = str(d / "candidate.o")
        else:
            fixture = str(source / "test/htp/host/lfm_tail_gather_check.c")
            cc = next(cmd[:] for cmd in build if fixture in cmd)
            cc.insert(1, "-I" + str(d))
            oldobj = str(checks / "helper.o")
            cc[cc.index(oldobj)] = str(d / "helper.o")
            link = next(cmd[:] for cmd in build if cmd[0] == "g++" and
                        str(checks / "helper-check") in cmd)
            link[link.index(oldobj)] = str(d / "helper.o")
        link[link.index("-o") + 1] = str(d / "check")
        commands = [cc, link, [str(d / "check")]]
        (d / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        for i, cmd in enumerate(commands[:2]):
            with (d / f"build{i}.log").open("w") as f:
                subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                               check=True, env=env)
        with (d / "run.log").open("w") as f:
            run = subprocess.run(commands[2], stdout=f, stderr=subprocess.STDOUT,
                                 env=env)
        log = (d / "run.log").read_text()
        assert run.returncode != 0, f"SURVIVED: {name}"
        assert "Assertion" in log or "ERROR: AddressSanitizer" in log, log
        results.append({"name": name, "returncode": run.returncode,
                        "killed": True, "whole_tu": whole,
                        "mutated_source_sha256": hashlib.sha256(c.read_bytes()).hexdigest(),
                        "evidence": log.splitlines()[-1]})
    (out / "results.json").write_text(json.dumps({
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "checks_sha256": hashlib.sha256((checks / "results.json").read_bytes()).hexdigest(),
        "results": results}, indent=2) + "\n")
    print(f"TAIL_GATHER_MUTANTS {len(results)}/{len(mutants)} killed")


if __name__ == "__main__":
    main()
