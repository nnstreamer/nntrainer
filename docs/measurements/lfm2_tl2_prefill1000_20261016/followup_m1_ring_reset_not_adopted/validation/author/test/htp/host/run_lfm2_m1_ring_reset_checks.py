#!/usr/bin/env python3
"""Check the actual DMA ring and optional whole native-M1 caller on Linux.

The ring TU is included byte-for-byte; only hardware primitives are emulated.
The optional caller reuses the existing actual-quantizer/native-M1 fixture,
with reset-call observations and both reference/candidate padding semantics
set to the already-retained M1 dead-setup implementation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def replace_one(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f"source anchor changed: {old!r}")
    return text.replace(old, new)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True, type=Path)
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--reference", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--sanitize", action="store_true")
    ap.add_argument("--mutations", action="store_true")
    ap.add_argument("--hexagon-tools", type=Path,
                    help="also run the whole-M1 actual-quantizer fixture")
    args = ap.parse_args()
    if sys.platform != "linux":
        ap.error("this host runner currently supports Linux only")
    if args.mutations and not args.sanitize:
        ap.error("--mutations requires --sanitize")
    root, source, reference, out = (
        p.resolve() for p in (args.repo, args.source, args.reference, args.output))
    rel = Path("nntrainer/tensor/htp_backend/hmx")
    out.mkdir(parents=True, exist_ok=True)
    stub = out / "stub"
    stub.mkdir(exist_ok=True)
    for name in ("hexagon_protos.h", "hexagon_types.h", "hmx_hexagon_protos.h"):
        (stub / name).write_text("/* Hardware primitives supplied by the fixture. */\n")
    original = (source / rel / "hexkl_dma_ring.c").read_text()
    header = (source / rel / "hexkl_dma_ring.h").read_bytes()
    fixture = Path(__file__).resolve().parent / "lfm2_m1_ring_reset_check.c"
    (out / "hexkl_dma_ring.h").write_bytes(header)
    commands = []
    env = os.environ.copy()
    env["ASAN_OPTIONS"] = "detect_leaks=1:halt_on_error=1"
    env["UBSAN_OPTIONS"] = "halt_on_error=1:print_stacktrace=1"

    def run(command, name, expected_failure=None):
        commands.append(command)
        (out / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        result = subprocess.run(command, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, env=env)
        (out / f"{name}.log").write_text(result.stdout)
        if expected_failure is None:
            if result.returncode:
                raise RuntimeError(f"{name} failed: {result.returncode}; see log")
        elif result.returncode == 0 or not any(
                marker in result.stdout for marker in expected_failure):
            raise RuntimeError(f"{name} mutant was not caught as intended")
        return result

    cc = os.environ.get("CC", "gcc")
    flags = ["-std=gnu11", "-O1", "-g", "-Wall", "-Wextra", "-Werror",
             "-I", str(out), "-I", str(stub)]
    if args.sanitize:
        flags += ["-fsanitize=address,undefined", "-fno-sanitize-recover=undefined",
                  "-fno-omit-frame-pointer", "-fno-pie", "-no-pie"]
    (out / "ring-under-test.c").write_text(original)
    exe = out / "ring_check"
    run([cc, *flags, str(fixture), "-o", str(exe)], "build-ring")
    result = run([str(exe)], "ring")
    print(result.stdout, end="")

    mutant_results = []
    if args.mutations:
        mutants = (
            ("reserved_zero", "  d->reserved1 = 0;", "  /* omitted reserved1 */"),
            ("done_clear", "  d->done = 0;", "  /* omitted done clear */"),
            ("completion_guard", "!hexkl_dma_ring_is_done(last)", "0"),
            ("type_guard", "g_tail->desc_type != 9u", "0"),
            ("size_guard", "g_tail->desc_size != 1u", "0"),
            ("tail_guard", "g_tail != &g_ring[last]", "0"),
            ("whole_clear", "  g_tail = NULL;", "  memset(g_ring, 0, sizeof(g_ring));\n  g_tail = NULL;"),
            ("start_state", "  g_tail = NULL;\n  g_started = 0;", "  g_tail = NULL;"),
        )
        for name, old, new in mutants:
            changed = replace_one(original, old, new)
            (out / "ring-under-test.c").write_text(changed)
            mexe = out / f"mutant-{name}"
            run([cc, *flags, str(fixture), "-o", str(mexe)], f"build-{name}")
            r = run([str(mexe)], f"mutant-{name}", ("Assertion", "AddressSanitizer"))
            mutant_results.append({"name": name, "exit": r.returncode})
        (out / "ring-under-test.c").write_text(original)
        print(f"{len(mutant_results)} ring guard/fill/state mutants rejected")

    caller = None
    caller_mutants = []
    if args.hexagon_tools:
        existing = root / "test/htp/host"
        stub_path = source / "test/htp/host/moe_layer_host_check.c"
        if not stub_path.is_file():
            stub_path = existing / "moe_layer_host_check.c"
        (out / "moe_layer_host_check.c").write_bytes(stub_path.read_bytes())
        text = (existing / "lfm2_m1_setup_check.c").read_text()
        text = replace_one(text, '#define main nntr_existing_moe_check_main',
                           '#define hexkl_dma_ring_reset fixture_generic_reset\n'
                           '#define hexkl_dma_ring_try_reset fixture_default_try_reset\n'
                           '#define main nntr_existing_moe_check_main')
        text = replace_one(text, '#undef main',
                           '#undef main\n#undef hexkl_dma_ring_reset\n'
                           '#undef hexkl_dma_ring_try_reset')
        text = replace_one(text, 'static int g_candidate, g_fast, g_observed, g_watch_padding;',
                           'static unsigned try_reset_calls, generic_reset_calls;\n'
                           'static int try_reset_result;\n'
                           'int hexkl_dma_ring_try_reset(void) {\n'
                           '  ++try_reset_calls; return try_reset_result;\n}\n'
                           'void hexkl_dma_ring_reset(void) {\n'
                           '  ++generic_reset_calls; fixture_generic_reset();\n}\n'
                           'static int g_candidate, g_fast, g_observed, g_watch_padding;')
        text = replace_one(text, '    g_candidate = candidate;',
                           '    g_candidate = 1; /* both sources retain dead setup */\n'
                           '    try_reset_calls = generic_reset_calls = 0;\n'
                           '    try_reset_result = (cases & 1u) == 0u;')
        text = replace_one(text, '    unpoison_all();',
                           '    assert(try_reset_calls == (unsigned)(candidate && g_fast && !bad_row));\n'
                           '    if (!bad_row && (active || !rows_out))\n'
                           '      assert(generic_reset_calls == (unsigned)!(candidate && g_fast && try_reset_result));\n'
                           '    else\n'
                           '      assert(generic_reset_calls == 0u);\n'
                           '    unpoison_all();')
        text = replace_one(text, 'if (candidate && g_fast)', 'if (g_fast)')
        cfile = out / "ring_m1_caller.c"
        cfile.write_text(text)
        runner = (existing / "lfm2_m1_setup_check.py").read_text()
        runner = replace_one(runner, 'here / "lfm2_m1_setup_check.c"',
                             f'Path({str(cfile)!r})')
        runner_path = out / "run_caller.py"
        runner_path.write_text(runner)
        caller = out / "caller"
        caller.mkdir(exist_ok=True)
        (caller / "hexkl_dma_ring.h").write_bytes(header)
        command = [sys.executable, str(runner_path), "--repo", str(root),
                   "--source", str(source / rel / "hexkl_mm_u8i4_moe.c"),
                   "--reference", str(reference / rel / "hexkl_mm_u8i4_moe.c"),
                   "--output", str(caller), "--hexagon-tools", str(args.hexagon_tools)]
        if args.sanitize:
            command.append("--sanitize")
        result = run(command, "caller")
        print(result.stdout, end="")
        if args.mutations:
            candidate_moe = (source / rel / "hexkl_mm_u8i4_moe.c").read_text()
            old = 'if (!lfm_m1_setup || !hexkl_dma_ring_try_reset())'
            for name, new in (
                    ("unguarded", 'if (!hexkl_dma_ring_try_reset())'),
                    ("unused_fast_path", 'if (!lfm_m1_setup || 1)')):
                msource = out / f"caller-mutant-{name}.c"
                msource.write_text(replace_one(candidate_moe, old, new))
                mout = out / f"caller-mutant-{name}"
                mout.mkdir(exist_ok=True)
                (mout / "hexkl_dma_ring.h").write_bytes(header)
                mcommand = command.copy()
                mcommand[mcommand.index("--source") + 1] = str(msource)
                mcommand[mcommand.index("--output") + 1] = str(mout)
                r = run(mcommand, f"caller-mutant-{name}", ("check failed",))
                failure_log = (mout / "check.log").read_text()
                if "Assertion" not in failure_log or "try_reset_calls" not in failure_log:
                    raise RuntimeError(f"{name}: caller mutant failed for the wrong reason")
                caller_mutants.append({"name": name, "exit": r.returncode})
            print(f"{len(caller_mutants)} caller dispatch mutants rejected")

    hashes = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (source / rel).glob("*") if p.is_file()}
    (out / "validation.json").write_text(json.dumps({
        "source_sha256": hashes,
        "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
        "sanitize": args.sanitize, "mutants": mutant_results,
        "caller_mutants": caller_mutants,
        "actual_ring_source": True,
        "caller": str(caller) if caller else None,
        "limits": ["host hardware primitives are delayed ordered emulation",
                   "no DMA cache visibility, device timing, or speedup proof",
                   "original timeout fallback is unchanged and not repaired",
                   "caller retains existing stand-in-only shift-base suppression"],
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
