#!/usr/bin/env python3
"""Strictly validate and summarize the two existing normal-app I/O traces."""
import argparse
import collections
import hashlib
import importlib.util
import json
import re
from pathlib import Path

FIELDS = {
    "BATCH": {"id", "experts", "jobs_ahead", "queued_us", "demand_us",
              "ready", "joined_us", "register_start_us", "register_end_us"},
    "READ": {"batch", "expert", "off_gu", "start_us", "end_us", "io_us",
             "copy_us", "calls", "bytes", "cpu", "rc"},
    "COMPUTE": {"kind", "M", "K", "N", "start_us", "end_us"},
}


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def records(path):
    result = {tag: [] for tag in FIELDS}
    for line in path.read_text().splitlines():
        if "[HTP-PREFILL-" not in line:
            continue
        m = re.search(r"\[HTP-PREFILL-(BATCH|READ|COMPUTE)\] (.*)$", line)
        if not m:
            raise ValueError(f"malformed tag: {line}")
        tokens = m[2].split()
        if any(not re.fullmatch(r"\w+=-?\d+", token) for token in tokens):
            raise ValueError(f"malformed value: {line}")
        data = {k: int(v) for k, v in (token.split("=") for token in tokens)}
        if set(data) != FIELDS[m[1]] or len(data) != len(tokens):
            raise ValueError(f"duplicate/missing/unknown field: {line}")
        result[m[1]].append(data)
    batches, reads, compute = (result[k] for k in ("BATCH", "READ", "COMPUTE"))
    if [b["id"] for b in batches] != list(range(1, 66)):
        raise ValueError("expected 65 unique ordered batches")
    if len(reads) != 2304 or len(compute) != 222:
        raise ValueError("expected 2304 reads and 222 total compute records")
    if len({r["off_gu"] for r in reads}) != 2304:
        raise ValueError("duplicated expert file offset")
    for b in batches:
        rs = [r for r in reads if r["batch"] == b["id"]]
        if sorted(r["expert"] for r in rs) != list(range(b["experts"])):
            raise ValueError("missing/duplicate expert in batch")
        if not 0 <= b["ready"] <= len(rs):
            raise ValueError("invalid ready count")
        if not b["queued_us"] <= b["demand_us"] <= b["joined_us"] <= b["register_start_us"] <= b["register_end_us"]:
            raise ValueError("invalid batch time order")
        for r in rs:
            if r["rc"] != 0 or r["calls"] != 1 or not b["queued_us"] <= r["start_us"] <= r["end_us"] <= b["joined_us"]:
                raise ValueError("failed/malformed/read not joined")
            if r["io_us"] + r["copy_us"] > r["end_us"] - r["start_us"]:
                raise ValueError("invalid read timing partition")
    if sum(r["bytes"] for r in reads) != 4656726016:
        raise ValueError("unexpected total direct-read bytes")
    return result


def analyze(path, io, m0):
    rec = records(path)
    summary = io.analyze(path)
    if (summary["excluded_warmup_calls"], summary["traced_compute_calls"], summary["traced_moe_calls"]) != (102, 120, 38):
        raise ValueError("main/warmup cohort mismatch")
    summary["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    summary["m0"] = m0.profile(path, 1024)
    batches, reads = rec["BATCH"], rec["READ"]
    start, end = min(r["start_us"] for r in reads), max(r["end_us"] for r in reads)
    events = collections.defaultdict(lambda: [0, 0])
    for b in batches:
        events[b["queued_us"]][0] += b["experts"]
    for r in reads:
        events[r["start_us"]][0] -= 1
        events[r["start_us"]][1] += 1
        events[r["end_us"]][1] -= 1
    queue = busy = 0
    previous = min(events)
    timeline = []
    for t, (dq, db) in sorted(events.items()):
        a, b = max(previous, start), min(t, end)
        if b > a:
            timeline.append((a, b, queue, busy))
        queue += dq
        busy += db
        if queue < 0 or not 0 <= busy <= 6:
            raise ValueError("invalid submitted/start or busy event count")
        previous = t
    if queue != 0 or busy != 0:
        raise ValueError("undrained final jobs")
    starvation = [(a, b) for a, b, q, n in timeline if q == 0 and n == 0]
    # Queue can include up to six popped jobs whose read body has not started.
    # A positive lower bound removes this ambiguity.
    body_active_with_definite_queued_work = sum(b-a for a,b,q,n in timeline if q > 6)
    summary["queue_reconstruction"] = {
        "definite_starvation_us": sum(b-a for a,b in starvation),
        "definite_starvation_intervals": starvation,
        "some_job_definitely_still_in_deque_us": body_active_with_definite_queued_work,
        "some_job_definitely_still_in_deque_fraction": body_active_with_definite_queued_work/(end-start),
        "read_body_capacity_utilization": summary["service_ms"]/(6*summary["read_window_ms"]),
        "read_window_minus_service_div_six_ms": summary["read_window_ms"]-summary["service_ms"]/6,
        "interpretation": "No timing claim outside this recorded read window. Busy intervals cover read function bodies including pread wait/copy, not CPU utilization. Unstarted count includes at most six claimed jobs. Existing trace has no explicit held/free-slot timestamp.",
    }
    summary["late_batches_over_500us"] = [b for b in summary["per_batch"] if b["exposed_wait_us"] > 500]
    return summary, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    a = ap.parse_args()
    e = a.repo / "docs/measurements/lfm2_tl2_prefill1000_20261016"
    io = module("existing_io", e / "validation/analyze_prefill_io.py")
    own = module("existing_m0", Path(__file__).with_name("analyze_existing.py"))
    all_records = []
    compact = []
    for variant in (0, 1):
        s, rec = analyze(e / f"continue-router-arm-io1024-{variant}.log", io, own)
        all_records.append(rec)
        (a.output_dir / f"matched-{variant}-summary.json").write_text(json.dumps(s, indent=2)+"\n")
        compact.append({k:s[k] for k in ("read_window_ms", "service_ms", "exposed_wait_ms", "register_ms", "queue_reconstruction")})
    def stream(r):
        return [(x["batch"], x["expert"], x["off_gu"], x["bytes"], x["calls"])
                for x in r["READ"]]
    if stream(all_records[0]) != stream(all_records[1]):
        raise ValueError("OFF/ON batch/expert order or payload differs")
    print(json.dumps({"ordered_payload_identical": True, "variants": compact}, indent=2))


if __name__ == "__main__":
    main()
