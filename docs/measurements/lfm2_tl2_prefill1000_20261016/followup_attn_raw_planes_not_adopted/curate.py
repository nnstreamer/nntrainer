#!/usr/bin/env python3
"""Create a compact rejected-raw4plane archive; all inputs are read-only."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("author", "pair", "kv-pair", "root-results", "device-review", "review", "evidence", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    a = p.parse_args(); out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    A, PF, KF, P, H, O, E = (a.author, a.pair, a.kv_pair, a.root_results,
                            a.device_review, a.review, a.evidence)
    M = H / "model-stage-review"; O = O / "candidate-v2"
    origins = {}; replay = {"files": {x: {} for x in ("evidence", "results", "fixed-build", "fixture")},
                            "stage_jobs": [], "model_dispatch": []}

    def write(rel, data):
        f = out / rel; f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(data.encode() if isinstance(data, str) else data)

    def save(rel, value):
        write(rel, json.dumps(value, indent=2) + "\n")

    def cp(src, rel, compress=None):
        raw = src.read_bytes()
        if raw.startswith(b"\x7fELF") or src.suffix in (".o", ".so", ".bin", ".a"):
            raise ValueError("Binary excluded: " + str(src))
        if compress is None:
            compress = src.suffix in (".c", ".h", ".cpp", ".patch", ".log", ".s", ".su", ".inc") or len(raw) > 8192
        if compress and not rel.endswith(".gz"):
            rel += ".gz"
        write(rel, gzip.compress(raw, mtime=0) if compress else raw)
        origins[rel] = {"source": str(src.resolve()), "original_sha256": sha(raw),
                        "original_bytes": len(raw), "encoding": "gzip" if compress else "identity"}
        return rel

    def replay_copy(src, rel, namespace, filename=None, compress=None):
        target = cp(src, rel, compress)
        replay["files"][namespace][filename or src.name] = target
        return target

    # Twenty-four normal runs and six profile runs remain complete and separate.
    normal_jobs = []
    for group in ("normal", "confirmation"):
        for kind in ("jobs", "executions", "summary"):
            name = "root-" + group + "-fixed-" + kind + ".json"
            replay_copy(P / name, "root/" + name, "results")
        jobs = json.loads((P / ("root-" + group + "-fixed-jobs.json")).read_text())
        executions = json.loads((P / ("root-" + group + "-fixed-executions.json")).read_text())
        assert len(jobs) == len(executions) == 12
        normal_jobs += jobs
    replay_copy(P / "root-confirmation-plan.json", "root/root-confirmation-plan.json", "results", compress=False)
    for name in ("root-all-fixed-summary.json", "root-decision-and-restore.json", "root-baseline.json",
                 "root-source-apply.json", "root-fixed-build-check.json", "root-fixed-kv-deployments.json",
                 "root-model-diagnostic-fixed-summary.json", "root-stage-fixed-executions.json", "root-stage-fixed-jobs.json"):
        replay_copy(P / name, "root/" + name, "results")
    stage_jobs = json.loads((P / "root-stage-fixed-jobs.json").read_text())
    model_dispatch = json.loads((P / "root-model-diagnostic-fixed-summary.json").read_text())
    assert len(stage_jobs) == 4 and len(model_dispatch) == 2
    jobs = [(j, "normal") for j in normal_jobs] + [(j, "profiles") for j in stage_jobs]
    jobs += [({"tag": r["tag"]}, "profiles") for r in model_dispatch]
    for job, category in jobs:
        tag = job["tag"]
        for ext in ("json", "ids", "log", "memory.json"):
            filename = tag + "." + ext
            replay_copy(E / filename, "runs/" + category + "/" + filename, "evidence", compress=ext != "ids")
        for ext in ("logcat.txt",):
            src = E / (tag + "." + ext)
            if src.exists():
                replay_copy(src, "runs/" + category + "/" + src.name, "evidence", compress=True)
        cp(P / (tag + ".runner.log"), "runs/" + category + "/" + tag + ".runner.log", True)
    variants = sorted({j["variant"] for j in normal_jobs + stage_jobs})
    variants += ["lfm-attn-raw-planes-fixed-diagnostic", "lfm-kv-tail-fix-1"]
    for variant in variants:
        replay_copy(E / (variant + "-artifacts.json"), "runtime/" + variant + "-artifacts.json", "evidence", compress=False)

    for name in ("parse_stages.py", "audit_model_records.py", "audit_diagnostics.py", "check_stage_parser.py"):
        cp(M / name, "parsers/" + name, False)
    cp(A / "diagnostic/parse.py", "parsers/parse_dispatch.py", False)
    for name in ("pooled-normal-readback.json", "initial-normal-readback.json", "parser-checks.json", "stage-commands.json"):
        cp(M / name, "analysis/" + name, name.endswith("readback.json"))
    for name in ("diagnostic-audit-command.json", "memory-outlier-readback.json",
                 "model-audit-commands.json", "model-dispatch-readback.json",
                 "root-pooled-crosscheck.json", "source-timing-boundaries.json", "stage-readback.json"):
        cp(M / name, "analysis/" + name)
    for job in stage_jobs:
        tag = job["tag"]
        rel = cp(M / (tag + ".parsed.json"), "analysis/stages/" + tag + ".parsed.json", True)
        replay["stage_jobs"].append({"tag": tag, "tokens": 2048 if job["model"] == "model2048" else 1024, "expected": rel})
    for record in model_dispatch:
        tag = record["tag"]
        for ext in ("live.logcat.txt", "expected.json", "parsed.json"):
            replay_copy(P / (tag + "." + ext), "diagnostic/model/" + tag + "." + ext, "results", compress=ext.endswith("txt"))
        replay["model_dispatch"].append(record)
    for name in ("review.txt", "validation.json", "manifest.json"):
        source = M / name
        if source.exists():
            cp(source, "analysis/final-independent/" + name)

    # Production v2 and final actual-helper fixtures; v1 is only design history.
    for name in ("README.txt", "validation.json", "manifest.json", "source-hashes.json", "test-hashes.json",
                 "production.patch", "tests.patch"):
        cp(A / name, "author/" + name)
    for rel in ("nntrainer/tensor/htp_backend/hmx/hexkl_attn_q.c",):
        for side in ("baseline", "candidate"):
            cp(A / side / rel, "source/" + side + "/" + rel)
    for rel in json.loads((A / "test-hashes.json").read_text()):
        cp(A / rel, "source/" + rel)
    for name in ("production.patch", "source-hashes.json"):
        cp(A / "history/v1" / name, "history/v1/" + name)
    for label, folder in (("author", A / "checks/host-san-v2"), ("independent", O / "checks/strict")):
        for name in ("results.json", "commands.json", "attn_raw_planes_check.log", "attn_raw_planes_overlap_check.log"):
            cp(folder / name, "host/" + label + "/" + name)
    for label, folder in (("author", A), ("independent", O)):
        cp(folder / "checks/mutants/results.json", "host/" + label + "/mutants.json")
    for name in ("review.txt", "validation.json", "manifest.json", "codegen-summary.json", "unchanged-function-proof.json", "commands.json"):
        cp(O / name, "independent/" + name)
    cp(O / "layout-probe/validation.json", "independent/layout-validation.json")
    cp(A / "codegen/commands.json", "codegen/author-commands.json")
    for side in ("baseline", "candidate"):
        for name in ("phase_qk", "dequant_worker"):
            cp(O / (side + "-" + name + ".asm.txt"), "codegen/" + side + "-" + name + ".asm.txt", True)
    cp(O / "candidate-qk_planes_available.asm.txt", "codegen/qualifier.asm.txt", True)
    for name in ("diagnostic.patch", "parse.py", "parser_check.py", "parser-validation.json", "format_check.c",
                 "check.py", "manifest.json", "device-trace-expected.json"):
        cp(A / "diagnostic" / name, "diagnostic/source/" + name)
    for name in ("commands.json",):
        cp(A / "diagnostic/codegen" / name, "diagnostic/codegen/" + name)
    for name in ("results.json", "commands.json", "attn_raw_planes_overlap_check.log"):
        cp(A / "diagnostic/host-check/run" / name, "diagnostic/host/" + name)
    cp(O / "diagnostic/parser-validation.json", "diagnostic/independent-parser-validation.json")

    # Controlled links: exact inventory/provenance, no objects/ELFs/vendor bytes.
    for name in ("build_fixed_objects.py", "root-command.json", "candidate-config.json", "preparation.json"):
        cp(PF / name, "build/parent/" + name)
    for src in sorted((PF / "build").glob("*.json")):
        cp(src, "build/parent/" + src.name, True)
    for name in ("validation.json", "commands.json", "overlay.json", "imports.log", "object-imports.log", "diagnostic.patch"):
        cp(PF / "diagnostic" / name, "build/diagnostic/" + name)
    for src in sorted((KF / "build").glob("*.json")):
        rel = cp(src, "build/common-kv/" + src.name, True)
        if src.name == "validation.json":
            replay["files"]["fixed-build"]["validation.json"] = rel
    for name in ("root-command.json", "common-fix-config.json", "root-build.log"):
        cp(KF / name, "build/common-kv/" + name)
    cp(P / "build_common_kv_fix.py", "build/common-kv/build_common_kv_fix.py")
    for name in ("validation.json", "post-recovery-validation.json", "own-link-section-differences.json"):
        cp(O / "fixed-object-review" / name, "build/independent/" + name)

    # All real-DSP fixture logs and sampled memory, dumps represented by SHA only.
    F = P / "device-fixture-fixed"
    executions = json.loads((F / "executions.json").read_text())
    for arm in ("control", "candidate", "diagnostic"):
        for ext in ("log", "memory.json", "execution.json"):
            cp(F / (arm + "." + ext), "device/" + arm + "." + ext, ext != "execution.json")
    for name in ("executions.json", "diagnostic.trace.json"):
        cp(F / name, "device/" + name, False)
    cp(F / "comparison.json", "device/comparison.json", True)
    replay_copy(F / "diagnostic.live.logcat.txt", "device/diagnostic.live.logcat.txt", "fixture", compress=True)
    replay_copy(A / "diagnostic/device-trace-expected.json", "device/device-trace-expected.json", "fixture", compress=False)
    dump_hashes = {r["arm"]: {"sha256": r["dump_sha256"], "bytes": (F / (r["arm"] + ".bin")).stat().st_size,
                                    "archived": False, "basis": "Root transfer SHA and independent complete-byte readback"} for r in executions}
    save("device/dump-hashes.json", dump_hashes)
    for name in ("review.txt", "validation.json", "manifest.json", "device-readback.json", "dispatch-readback.json"):
        cp(H / "fixed-readback" / name, "device/independent/" + name)
    for name in ("caseplan.json", "caseplan.inc", "attention_raw_planes_device_check.cpp", "build.py", "build-command.json", "manifest.json"):
        cp(A / "device" / name, "device/source/" + name)
    for name in ("compare_device.py", "check_parser.py", "review.txt", "validation.json"):
        cp(H / name, "device/review/" + name)
    cp(H / "parser-checks-final/validation.json", "device/review/parser-validation.json")

    # The prior artifact inspection error is preserved without mutated ELF copies.
    cp(O / "artifact-rewrite-incident.json", "incident/reviewer-incident.json")
    for name in ("recovery.json", "root-deploy-control.log", "lfm-attn-raw-planes-control-artifacts.json"):
        cp(P / "elf-inspection-recovery" / name, "incident/" + name)
    cp(O / "fixed-object-review/post-recovery-validation.json", "incident/independent-post-recovery.json")
    save("references/kv-fix.json", {"commit": "e3eed8f40b504caa314b5329775e82d831f3b903",
          "path": "docs/measurements/lfm2_tl2_prefill1000_20261016/followup_fixed_q8_copy_tail",
          "scope": "Separate committed fixed-Q8 copy-tail repair; original failed control and corrected KV source/tests are retained there. No duplication of that full archive here.",
          "source_sha256": "fa750e41854303b716990c5d722e8b14a0c85e1776e26096ff880acaf007c291"})
    for name in ("review.txt", "audit.json", "manifest.json"):
        cp(Path("/tmp/nntr-lfm2-moe-input-prep-audit-20261017") / name, "related-audit/moe-input-prep/" + name)
    for name in ("stage-analysis.json", "source-and-input-hashes.json", "analyze_stages.py",
                 "review.txt", "prefill-io-followup.txt", "manifest.json"):
        cp(Path("/tmp/nntr-lfm2-attn-raw-planes-phase-audit-20261017") / name, "analysis/phase-audit/" + name)
    for name in ("root-run-normal-fixed.py", "root-run-confirmation-fixed.py", "root-run-model-diagnostic-fixed.py",
                 "root-run-stages-fixed.py", "root-run-device-fixture-fixed.py"):
        cp(P / name, "provenance-scripts/" + name, True)
    cp(E / "bench.py", "provenance-scripts/bench.py", True)

    pooled = json.loads((M / "pooled-normal-readback.json").read_text())
    assert pooled["all_runs"] == 24 and pooled["prespecified_adoption_metric_pass"] is False
    summary = {"adopted": False, "initial": pooled["groups"]["normal"]["summary"],
               "confirmation": pooled["groups"]["confirmation"]["summary"], "pooled": pooled["pooled"],
               "normal_runs": 24, "profile_runs": 6, "production_patch_sha256": sha((A / "production.patch").read_bytes()),
               "scope": "Predeclared both-length criterion failed; one diagnostic sample per arm/length does not establish whole-model gain.",
               "all_ids_and_peak_checks_pass": pooled["all_ids_and_peak_checks_pass"],
               "model_memory_limits": pooled["limitations"],
               "full_fixture_dump_bytes_omitted": True, "gemma_model_performance_measured": False}
    save("summary.json", summary); save("replay-index.json", replay)
    for name in ("verify.py", "reproduce_host.py", "README.template.txt"):
        cp(Path(__file__).resolve().parent / name, name, False)
    cp(Path(__file__).resolve(), "curate.py", False)
    template = (Path(__file__).resolve().parent / "README.template.txt").read_text()
    table = []
    for group in ("initial", "confirmation", "pooled"):
        for row in summary[group]:
            f = row["prefill_tps"]
            table.append(f"{group:12s} {row['tokens']:4d}  {f['median_control']:9.3f} -> {f['median_candidate']:9.3f}  {f['median_pct']:+7.3f}%  {f['pairs_faster']}/{row['pairs']} paired wins")
    write("README.txt", template.replace("@@TABLE@@", "\n".join(table)))
    save("provenance.json", {"copied_files": origins, "generated": ["README.txt", "summary.json", "replay-index.json", "references/kv-fix.json", "device/dump-hashes.json", "provenance.json", "manifest.json"]})
    files = {}
    for f in sorted(out.rglob("*")):
        if f.is_file():
            data = f.read_bytes(); row = {"bytes": len(data), "sha256": sha(data)}
            if f.suffix == ".gz":
                raw = gzip.decompress(data); row.update(uncompressed_bytes=len(raw), uncompressed_sha256=sha(raw))
            files[str(f.relative_to(out))] = row
    save("manifest.json", {"schema": 1, "files": files})
    subprocess.run([sys.executable, "-B", str(out / "verify.py")], check=True)
    print(json.dumps({"archive": str(out), "payloads": len(files), "total_files": len(files) + 1,
                      "bytes": sum(f.stat().st_size for f in out.rglob("*") if f.is_file()),
                      "manifest_sha256": sha((out / "manifest.json").read_bytes())}))


if __name__ == "__main__":
    main()
