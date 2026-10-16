#!/usr/bin/env python3
"""Read-only archive verification and replay of strict recorded-data parsers."""
import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, default=Path(__file__).resolve().parent)
    a = p.parse_args(); root = a.archive.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    actual = {str(f.relative_to(root)) for f in root.rglob("*") if f.is_file()}
    need(actual == set(manifest["files"]) | {"manifest.json"}, "Missing/unlisted archive file")
    for rel, row in manifest["files"].items():
        f = root / rel
        need(not f.is_symlink() and root in f.resolve().parents, "Unsafe archive path")
        data = f.read_bytes()
        need(len(data) == row["bytes"] and sha(data) == row["sha256"], "Payload identity: " + rel)
        need(not data.startswith(b"\x7fELF"), "ELF payload excluded")
        need(f.suffix not in (".c", ".h", ".cpp", ".o", ".so", ".a", ".bin"), "Historical source/binary packaging")
        if f.suffix == ".gz":
            raw = gzip.decompress(data)
            need(len(raw) == row["uncompressed_bytes"] and sha(raw) == row["uncompressed_sha256"], "Raw payload identity")

    def read(rel):
        data = (root / rel).read_bytes()
        return gzip.decompress(data) if rel.endswith(".gz") else data

    def doc(rel):
        return json.loads(read(rel))

    for rel, origin in doc("provenance.json")["copied_files"].items():
        need(sha(read(rel)) == origin["original_sha256"], "Origin mismatch: " + rel)
    index = doc("replay-index.json")
    summary = doc("summary.json")
    need(summary["adopted"] is False, "This candidate was not retained")
    with tempfile.TemporaryDirectory(prefix="nntr-raw4-readback-") as work:
        temp = Path(work)
        for namespace, records in index["files"].items():
            for filename, relative in records.items():
                target = temp / namespace / filename
                need(not Path(namespace).is_absolute() and not Path(filename).is_absolute()
                     and ".." not in Path(namespace).parts and ".." not in Path(filename).parts
                     and temp in target.resolve().parents, "Unsafe replay path")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(read(relative))
        E, P, K = (temp / n for n in ("evidence", "results", "fixed-build"))

        def run(script, args):
            q = subprocess.run([sys.executable, "-B", str(root / "parsers" / script), *map(str, args)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            need(q.returncode == 0, script + " failed:\n" + q.stdout + q.stderr)

        out = temp / "readback"
        run("audit_model_records.py", ["--evidence", E, "--results", P, "--fixed-build", K,
                                       "--output", out, "--include-confirmation"])
        pooled = json.loads((out / "pooled-normal-readback.json").read_text())
        need(pooled == doc("analysis/pooled-normal-readback.json.gz"), "Strict 24-run readback differs")
        need(pooled["all_runs"] == 24 and not pooled["prespecified_adoption_metric_pass"], "Prespecified decision")
        need(pooled["pooled"] == summary["pooled"] and
             pooled["groups"]["normal"]["summary"] == summary["initial"] and
             pooled["groups"]["confirmation"]["summary"] == summary["confirmation"], "Series/pooled summary mismatch")
        need(sha((P / "root-confirmation-plan.json").read_bytes()) == pooled["prespecified_confirmation_plan_sha256"],
             "Plan changed")

        for job in index["stage_jobs"]:
            tag = job["tag"]; output = out / (tag + ".parsed.json")
            run("parse_stages.py", ["--logcat", E / (tag + ".logcat.txt"),
                 "--log", E / (tag + ".log"), "--metadata", E / (tag + ".json"),
                 "--memory", E / (tag + ".memory.json"), "--ids", E / (tag + ".ids"),
                 "--tokens", job["tokens"], "--output", output])
            need(json.loads(output.read_text()) == doc(job["expected"]), "Stage readback changed: " + tag)

        # Recompute all six profile runs' runtime/memory identities and derive
        # dispatch counts from geometry independently of archived expectations.
        run("audit_diagnostics.py", ["--evidence", E, "--results", P,
            "--fixed-build", K, "--trace-parser", root / "parsers/parse_dispatch.py",
            "--output", out])
        for name in ("stage-readback.json", "model-dispatch-readback.json"):
            need(json.loads((out / name).read_text()) == doc("analysis/" + name + ".gz"),
                 "Independent diagnostic readback differs: " + name)

        ids_hash = {1024: "b6c1b801f95caded5399ec685cce0f900a20c3f9df760482dd7b5c9a62a5b92e",
                    2048: "2bc7afc3c7d8e4387c8d9c7378d6792a1ef0fb56c2635f73b95e8650ac7fef33"}
        sys.path.insert(0, str(root / "parsers"))
        import audit_model_records
        build = json.loads((K / "validation.json").read_text())
        for record in index["model_dispatch"]:
            tag = record["tag"]; meta = json.loads((E / (tag + ".json")).read_text())
            mem_path = E / (tag + ".memory.json"); mem = json.loads(mem_path.read_text())
            audit_model_records.memory_readback(mem)
            need(meta["profile"] and meta["instrumented"] and not meta["performance_comparison_eligible"], "Diagnostic TPS exclusion")
            need(meta["valid"] and meta["ids_match_reference"] and meta["returncode"] == 0, "Diagnostic success")
            need(sha(mem_path.read_bytes()) == meta["full_run_memory_report_sha256"], "Diagnostic memory identity")
            ids = (E / (tag + ".ids")).read_bytes()
            need(len(ids.splitlines()) == 65 and sha(ids) == ids_hash[meta["prefill_tokens"]] == meta["ids_sha256"], "Diagnostic 65 IDs")
            need(record["runtime_sha256"]["libnntr_hvx_skel.so"] == build["diagnostic"]["sha256"], "Diagnostic DSP identity")
            output = temp / (tag + ".dispatch.json")
            run("parse_dispatch.py", ["--log", P / (tag + ".live.logcat.txt"),
                 "--expected", P / (tag + ".expected.json"), "--pid", mem["pid"], "--output", output])
            need(json.loads(output.read_text()) == record["parser_result"], "Model actual-dispatch mismatch")

        fixture = doc("device/executions.json")
        dump_records = doc("device/dump-hashes.json")
        compare_path = temp / "compare_device.py"
        compare_path.write_bytes(read("device/review/compare_device.py.gz"))
        spec = importlib.util.spec_from_file_location("frozen_compare_device", compare_path)
        comparator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(comparator)
        plan_path = temp / "device-caseplan.json"
        plan_path.write_bytes(read("device/source/caseplan.json"))
        plan = comparator.read_plan(plan_path)
        parsed_fixtures = []
        for execution in fixture:
            arm = execution["arm"]
            need(execution["returncode"] == 0 and execution["runtime_sha256"]["libnntr_hvx_skel.so"] == build[arm]["sha256"], "Device fixture arm identity")
            need(execution["dump_sha256"] == dump_records[arm]["sha256"] and dump_records[arm]["bytes"] == 231908352, "Device dump identity")
            parsed_fixtures.append(comparator.parse_log(read("device/" + arm + ".log.gz").decode(), plan))
        need(parsed_fixtures[0] == parsed_fixtures[1] == parsed_fixtures[2],
             "Cross-arm actual fixture geometry/schema/input/replay/probe metadata")
        output = temp / "fixture-dispatch.json"
        run("parse_dispatch.py", ["--log", temp / "fixture/diagnostic.live.logcat.txt",
             "--expected", temp / "fixture/device-trace-expected.json", "--pid", fixture[-1]["pid"], "--output", output])
        need(json.loads(output.read_text()) == doc("device/diagnostic.trace.json"), "Fixture actual-dispatch mismatch")
        comparison = doc("device/comparison.json.gz")
        need(comparison["all_outputs_bit_exact_finite"] and comparison["all_aba_replays_exact_and_b_changed"], "Recorded byte comparison failed")
        need(len({x["sha256"] for x in dump_records.values()}) == 1, "Cross-arm dump hash mismatch")

    incident = doc("incident/recovery.json")
    need(incident["device_executions_before_detection"] == 0, "Inspection incident execution scope")
    restore = doc("root/root-decision-and-restore.json")
    need(not restore["adopted_attention_candidate"] and restore["source_delta_empty"], "Final source not restored")
    need(restore["new_accepted_dsp_sha256"] == build["control"]["sha256"], "Corrected control restore identity")
    print(json.dumps({"status": "PASS", "payloads": len(manifest["files"]), "normal_runs": 24,
                      "stage_profile_runs": 4, "actual_dispatch_profile_runs": 2, "fixture_arms": 3,
                      "prespecified_adoption_metric_pass": False,
                      "limits": "Recorded-data replay only. Full fixture dumps and ELFs are intentionally omitted; their byte comparisons are evidenced by frozen reports/hashes."}))


if __name__ == "__main__":
    main()
