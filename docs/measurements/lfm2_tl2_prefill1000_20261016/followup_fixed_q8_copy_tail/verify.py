#!/usr/bin/env python3
"""Verify this portable text archive; no compiler, adb, or SDK invocation."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.archive.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    expected = set(manifest["files"])
    actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    require(actual == expected | {"manifest.json"}, "missing or unlisted archive file")
    for rel, record in manifest["files"].items():
        path = root / rel
        require(not path.is_symlink() and root in path.resolve().parents, "unsafe path")
        data = path.read_bytes()
        require(len(data) == record["bytes"], "size mismatch: " + rel)
        require(hashlib.sha256(data).hexdigest() == record["sha256"], "hash mismatch: " + rel)
        require(not data.startswith(b"\x7fELF"), "ELF must not be archived")
        if rel.endswith(".gz"):
            raw = gzip.decompress(data)
            require(hashlib.sha256(raw).hexdigest() == record["uncompressed_sha256"],
                    "uncompressed hash mismatch: " + rel)
            require(len(raw) == record["uncompressed_bytes"], "uncompressed size mismatch")

    def read(rel):
        data = (root / rel).read_bytes()
        return gzip.decompress(data) if rel.endswith(".gz") else data

    def doc(rel):
        return json.loads(read(rel))

    provenance = doc("provenance.json")
    for rel, origin in provenance["copied_files"].items():
        require(hashlib.sha256(read(rel)).hexdigest() == origin["original_sha256"],
                "source provenance mismatch: " + rel)

    summary = doc("summary.json")
    require(summary["status"] == "CORRECTNESS_FIX_VALIDATED", "unexpected verdict")
    require(summary["performance_claim"] is None, "no TPS claim permitted")
    require(summary["gemma_model_performance_measured"] is False, "no Gemma TPS evidence")
    plain_sources = {str(p.relative_to(root)) for p in root.rglob("*")
                     if p.is_file() and p.suffix in (".c", ".h", ".cpp")}
    require(plain_sources == {"tests/kv_fixed_tail_check.c"}, "historical C must be gzip-wrapped")
    require(hashlib.sha256(read("source/corrected/hexkl_kv_q.c.gz")).hexdigest() ==
            summary["production_source_sha256"], "corrected source identity mismatch")
    require(doc("device/original/failure.json")["status"] == "CONTROL_FAILED_BEFORE_CANDIDATE",
            "original failure missing")
    old = doc("device/original/control.execution.json")
    new = doc("device/corrected/control.execution.json")
    require(old["returncode"] == 1 and new["returncode"] == 0, "control exit mismatch")
    require(old["fixture_sha256"] == new["fixture_sha256"], "fixture changed")
    for name, digest in old["runtime_sha256"].items():
        if name != "libnntr_hvx_skel.so":
            require(new["runtime_sha256"][name] == digest, "host runtime changed")
    rows = [json.loads(line) for line in read("device/corrected/control.log.gz").splitlines()]
    cases = [r for r in rows if r["record"] == "CASE"]
    probes = [r for r in rows if r["record"] == "PROBE"]
    endings = [r for r in rows if r["record"] == "SUMMARY"]
    require(len(cases) == 36 and len(probes) == 4 and len(endings) == 1, "fixture incomplete")
    require(endings[0]["bytes"] == 231908352 and endings[0]["rc"] == 0, "bad terminal record")
    require(all(c["rc"] == 0 and c["finite"] for c in cases), "failed/nonfinite case")
    fixed = [c for c in cases if c["case_id"] == 3]
    require(len(fixed) == 3 and all(c["fixed"] and c["hd"] == 64 for c in fixed),
            "fixed-HD64 cohort missing")
    require([c["repetition"] for c in fixed] == [0, 1, 2] and fixed[2]["same_repeat"] is True,
            "fixed-HD64 A/B/A did not pass")
    build = doc("build/validation.json.gz")
    require(build["status"] == "PASS" and build["no_op_overlay_object_exact"], "bad common build")
    require(build["active_objects_per_arm"] == 63 and
            build["common_objects_between_normal_arms"] == 62 and
            build["retained_original_common_objects"] == 61, "common object scope changed")
    require(build["common_correction"]["patch_sha256"] ==
            hashlib.sha256(read("source/production.patch.gz")).hexdigest(), "wrong KV patch")
    executions = doc("device/corrected/executions.json")
    dumps = doc("device/dump-hashes.json")
    require(len(executions) == 3, "missing corrected arm")
    for execution in executions:
        arm = execution["arm"]
        require(execution["returncode"] == 0, "failed corrected arm")
        require(execution["runtime_sha256"]["libnntr_hvx_skel.so"] == build[arm]["sha256"],
                "deployed DSP does not match corrected build")
        require(execution["dump_sha256"] == dumps["corrected/" + arm]["sha256"],
                "dump identity mismatch")
    require(len({dumps["corrected/" + a]["sha256"] for a in ("control", "candidate", "diagnostic")}) == 1,
            "corrected arm dump hashes differ")
    require(all(dumps["corrected/" + a]["bytes"] == 231908352
                for a in ("control", "candidate", "diagnostic")), "dump extent mismatch")
    canonical = read("tests/kv_fixed_tail_check.c")
    author = read("author/test-source.c.gz")
    require(b"".join(canonical.split()) == b"".join(author.split()), "test formatting changed tokens")
    require(doc("host/root/results.json")["hashes"][summary["root_canonical_test_path"]] ==
            hashlib.sha256(canonical).hexdigest(), "root tested another canonical fixture")
    for name in ("author", "independent", "root"):
        result = doc("host/" + name + "/results.json")
        require(result["candidate-check"]["returncode"] == 0, "host test failed: " + name)
        require(result["baseline-copy-overflow"]["returncode"] != 0 and
                result["baseline-worker-order"]["returncode"] != 0, "baseline defects not reproduced")
    print(json.dumps({"status": "PASS", "payloads": len(expected),
                      "stored_bytes": sum(x["bytes"] for x in manifest["files"].values()),
                      "scope": "Archive hashes and recorded correctness relations; omitted dumps are not re-compared."}))


if __name__ == "__main__":
    main()
