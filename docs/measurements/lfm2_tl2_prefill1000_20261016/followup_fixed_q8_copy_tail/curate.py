#!/usr/bin/env python3
"""Create a compact, new-only KV correctness evidence archive from frozen inputs."""
import argparse
import difflib
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def digest(data):
    return hashlib.sha256(data).hexdigest()


def stream_hash(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "author", "review", "root-evidence", "common-build", "device-review", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    D, O, P, K, R, I = (args.author, args.review, args.root_evidence,
                        args.common_build, args.repo, args.device_review)
    origins = {}

    def write(relative, data):
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, str):
            data = data.encode()
        target.write_bytes(data)

    def save(relative, value):
        write(relative, json.dumps(value, indent=2) + "\n")

    def copy(source, relative, compressed=False):
        data = source.read_bytes()
        if data.startswith(b"\x7fELF") or source.suffix in (".o", ".so", ".a", ".bin"):
            raise ValueError("No binary copy allowed: " + str(source))
        # Historical source snapshots are evidence, not new translation units:
        # retain exact bytes without subjecting them to all-new-line CI format.
        if source.suffix in (".c", ".h", ".cpp") and relative != "tests/kv_fixed_tail_check.c":
            compressed = True
            if not relative.endswith(".gz"):
                relative += ".gz"
        write(relative, gzip.compress(data, mtime=0) if compressed else data)
        origins[relative] = {"source": str(source.resolve()), "original_sha256": digest(data),
                             "original_bytes": len(data), "encoding": "gzip" if compressed else "identity"}

    canonical = R / "test/htp/host/kv_fixed_tail_check.c"
    author_test = D / "candidate/test/htp/host/kv_fixed_tail_check.c"
    assert b"".join(canonical.read_bytes().split()) == b"".join(author_test.read_bytes().split())
    assert digest((D / "production.patch").read_bytes()) == "c86726cb86782f33b820dafa9af5120c6831506ff991b8195d1f17b0a9cbf6ac"
    copy(D / "production.patch", "source/production.patch.gz", True)
    for name in ("baseline", "candidate"):
        copy(D / name / "nntrainer/tensor/htp_backend/hmx/hexkl_kv_q.c",
             "source/" + ("corrected" if name == "candidate" else name) + "/hexkl_kv_q.c")
    copy(canonical, "tests/kv_fixed_tail_check.c")
    copy(R / "test/htp/host/run_kv_fixed_tail_check.py", "tests/run_kv_fixed_tail_check.py")
    copy(author_test, "author/test-source.c")
    copy(D / "tests.patch", "author/tests-at-freeze.patch")
    copy(D / "README.txt", "author/README-at-freeze.txt")
    copy(D / "validation.json", "author/validation-at-freeze.json")
    copy(D / "source-hashes.json", "author/source-hashes.json")
    patch = ""
    for name in ("kv_fixed_tail_check.c", "run_kv_fixed_tail_check.py"):
        patch += "".join(difflib.unified_diff([], (out / "tests" / name).read_text().splitlines(True),
                                             fromfile="/dev/null", tofile="b/test/htp/host/" + name))
    write("source/tests-canonical.patch", patch)
    format_diff = "".join(difflib.unified_diff(
        author_test.read_text().splitlines(True), canonical.read_text().splitlines(True),
        fromfile="author/test-source.c", tofile="tests/kv_fixed_tail_check.c"))
    write("source/test-format-only.diff.gz", gzip.compress(format_diff.encode(), mtime=0))

    dependencies = ["nntrainer/tensor/htp_backend/hmx/hexkl_kv_q.h",
                    "nntrainer/tensor/htp_backend/hvx/hvx_kv_quant.c",
                    "nntrainer/tensor/htp_backend/hvx/hvx_kv_quant.h",
                    "nntrainer/tensor/htp_backend/hvx/hvx_tile_f16.h",
                    "nntrainer/tensor/htp_backend/hvx/hvx_worker_pool.c",
                    "nntrainer/tensor/htp_backend/hvx/hvx_worker_pool.h",
                    "test/htp/host/attn_qprep_native_unaligned.h"]
    dependencies += [str(p.relative_to(R)) for p in (R / "test/htp/host/stub").glob("*.h")]
    save("source/host-dependencies.json", {rel: digest((R / rel).read_bytes()) for rel in sorted(dependencies)})
    for name in ("review.txt", "validation.json", "manifest.json", "static-contract.json",
                 "portable-runner-review.json"):
        copy(O / name, "independent/" + name)
    for label, folder in (("author", D / "checks/portable-final"),
                          ("independent", O / "checks/strict"),
                          ("root", P / "root-kv-tail-host-check")):
        for name in ("results.json", "candidate-check.log", "baseline-copy-overflow.log",
                     "baseline-worker-order.log"):
            compressed = name == "baseline-copy-overflow.log"
            copy(folder / name, "host/" + label + "/" + name + (".gz" if compressed else ""), compressed)
        copy(folder / "commands.json", "host/" + label + "/commands.json.gz", True)
    for name in ("constant-host-validation.json", "commands.json", "constant-host-commands.json"):
        copy(O / "codegen" / name, "codegen/" + name + (".gz" if "commands" in name else ""), "commands" in name)
    for side in ("baseline", "candidate"):
        copy(O / "codegen" / side / "hexkl_kv_q-object.su", "codegen/" + side + ".su")
        copy(O / "codegen" / side / "constant-copy.c", "codegen/" + side + "-constant-copy.c")
        copy(O / "codegen" / side / "constant-copy.s", "codegen/" + side + "-constant-copy.s.gz", True)

    for folder, dest in ((P / "device-fixture", "original"),
                         (P / "device-fixture-fixed", "corrected")):
        for name in ("control.log", "control.memory.json"):
            copy(folder / name, "device/" + dest + "/" + name + ".gz", True)
        copy(folder / "control.execution.json", "device/" + dest + "/control.execution.json")
    copy(P / "device-fixture/failure.json", "device/original/failure.json")
    for name in ("executions.json", "diagnostic.trace.json"):
        copy(P / "device-fixture-fixed" / name, "device/corrected/" + name)
    copy(P / "device-fixture-fixed/comparison.json", "device/corrected/comparison.json.gz", True)
    for name in ("validation.json", "device-readback.json", "dispatch-readback.json"):
        copy(I / "fixed-readback" / name, "device/independent/" + name + ".gz", True)
    for name in ("review.txt", "manifest.json"):
        copy(I / "fixed-readback" / name, "device/independent/" + name)

    dumps = {}
    for key, path in [("original/control.partial", P / "device-fixture/control.partial.bin"),
                      *[("corrected/" + arm, P / "device-fixture-fixed" / (arm + ".bin"))
                        for arm in ("control", "candidate", "diagnostic")]]:
        dumps[key] = {"sha256": stream_hash(path), "bytes": path.stat().st_size,
                      "source_path": str(path), "archived": False}
    assert dumps["original/control.partial"]["sha256"] == "776062a1a0b33b33f70821c21091c4bf0f2a867c80dda81399dad294b6857fb3"
    assert all(dumps["corrected/" + arm]["sha256"] == "e64b0a2342d8132bd09e23763ca8958270c62a8e2a99dd720e1cb27f4478c394"
               for arm in ("control", "candidate", "diagnostic"))
    save("device/dump-hashes.json", dumps)
    memory = {}
    for label, folder, arm in (("original/control", P / "device-fixture", "control"),
                               *[("corrected/" + a, P / "device-fixture-fixed", a)
                                 for a in ("control", "candidate", "diagnostic")]):
        data = json.loads((folder / (arm + ".memory.json")).read_text())
        memory[label] = {key: data[key] for key in (
            "pid", "child_waited", "child_exit_code", "sample_count", "composite_available_sample_count",
            "requested_period_ms", "dma_buffer_identity_verified",
            "peak_anon_plus_fdinfo_dedup_or_raw_fallback_bytes", "limitations")}
        memory[label]["fdinfo_error_sample_count"] = sum(bool(s.get("fdinfo_read_errors")) for s in data["samples"])
        memory[label]["unavailable_composite_sample_count"] = sum(not s.get("composite_available") for s in data["samples"])
    save("device/memory-scope.json", {"scope": "Whole short fixture; original failed early. Not comparable model peak/performance data.", "runs": memory})

    for name in ("validation.json", "commands.json", "active-objects.json",
                 "external-input-hashes.json", "parent-file-hashes.json"):
        copy(K / "build" / name, "build/" + name + ".gz", True)
    for name in ("root-command.json", "common-fix-config.json", "root-build.log"):
        copy(K / name, "build/" + name)
    copy(P / "build_common_kv_fix.py", "build/build_common_kv_fix.py")
    copy(P / "root-run-device-fixture-fixed.py", "device/recorded-root-invocation.py")

    for name in ("verify.py", "reproduce_host.py", "README.txt"):
        copy(Path(__file__).resolve().parent / name, name)
    copy(Path(__file__).resolve(), "curate.py")
    summary = {
        "status": "CORRECTNESS_FIX_VALIDATED", "production_patch_sha256": digest((D / "production.patch").read_bytes()),
        "production_source_sha256": digest((D / "candidate/nntrainer/tensor/htp_backend/hmx/hexkl_kv_q.c").read_bytes()),
        "canonical_test_sha256": digest(canonical.read_bytes()),
        "canonical_runner_sha256": digest((out / "tests/run_kv_fixed_tail_check.py").read_bytes()),
        "root_canonical_test_path": str(canonical),
        "canonical_tests_patch_sha256": digest((out / "source/tests-canonical.patch").read_bytes()),
        "historical_c_storage": "Exact gzip-wrapped bytes; only final canonical test C remains a plain C file.",
        "author_test_diff": "Whitespace-only clang-format14 integration delta; root strict run passed final C 1099fec0.",
        "host_scope": "3 independently recorded actual-helper runs: author final portable, independent reviewer, root final canonical.",
        "copy_cases_per_run": 3094, "actual_worker_aba_cases_per_run": 384,
        "device_scope": "Original control failed fixed-HD64 A/B/A; same device fixture passes repaired control and both attention variants.",
        "lfm_dynamic_scope": "The current dynamic LFM path does not use the repaired fixed-Q8 row-copy branch; this fix is a separate correctness topic.",
        "device_calls_per_corrected_arm": 36, "device_probes_per_corrected_arm": 4,
        "device_output_bytes_per_corrected_arm": 231908352,
        "common_corrected_kv_all_three_arms": True,
        "performance_claim": None, "gemma_model_performance_measured": False,
        "model_memory_claim": None,
        "frozen_reports_pending_device_superseded_by": "device/corrected and device/independent final readback records",
        "excluded": "No ELF, object, SDK/vendor binary, output dump, or intermediate host repetitions."
    }
    save("summary.json", summary)
    save("provenance.json", {"copied_files": origins,
                             "generated_files": ["summary.json", "device/dump-hashes.json", "device/memory-scope.json",
                                                 "source/host-dependencies.json", "source/tests-canonical.patch",
                                                 "source/test-format-only.diff.gz", "manifest.json", "provenance.json"]})
    files = {}
    for path in sorted(out.rglob("*")):
        if path.is_file():
            data = path.read_bytes()
            record = {"sha256": digest(data), "bytes": len(data)}
            if path.suffix == ".gz":
                raw = gzip.decompress(data)
                record.update(uncompressed_sha256=digest(raw), uncompressed_bytes=len(raw))
            files[str(path.relative_to(out))] = record
    save("manifest.json", {"schema": 1, "files": files})
    subprocess.run([sys.executable, "-B", str(out / "verify.py")], check=True)
    print(json.dumps({"archive": str(out), "payloads": len(files), "files": len(files) + 1,
                      "bytes": sum(p.stat().st_size for p in out.rglob("*") if p.is_file()),
                      "manifest_sha256": digest((out / "manifest.json").read_bytes())}))


if __name__ == "__main__":
    main()
