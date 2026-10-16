#!/usr/bin/env python3
"""Read existing route/profile evidence; no production code or timing tests."""
import argparse
import hashlib
import json
import re
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def profile(path, tokens):
    lines = path.read_text().splitlines()
    rows = []
    for line in lines:
        if "[M0-PROF]" not in line:
            continue
        index = re.search(r"moe_layer\[(\d+)\]", line)
        data = {k: int(v) for k, v in re.findall(r"(\w+)=(\d+)", line)}
        required = {"tokens", "us", "setup", "router", "topk", "wksp",
                    "gather", "ffn", "route", "scatter", "other", "miss"}
        if index is None or set(data) != required or data["tokens"] != tokens:
            raise ValueError(f"malformed/foreign M0 record: {line}")
        data["layer"] = int(index[1])
        if data["us"] != sum(data[k] for k in required - {"tokens", "us", "miss"}):
            raise ValueError(f"nonconserving M0 timers: {line}")
        rows.append(data)
    if [r["layer"] for r in rows] != list(range(38)):
        raise ValueError("expected complete ordered 38-layer M0 cohort")
    return {
        "path": str(path), "sha256": sha(path), "rows": rows,
        "sum_us": {k: sum(r[k] for r in rows)
                   for k in rows[0] if k not in ("tokens", "layer", "miss")},
        "topk_range_us": [min(r["topk"] for r in rows),
                          max(r["topk"] for r in rows)],
        "prefetch_lines": [line for line in lines
                           if "[HTP-PROFILE] expert prefetch" in line],
        "interpretation": "Diagnostic only. router=0 does not measure the new CPU router; it runs outside g_m0.router. ffn includes staging, queue operations, backend call and final recency refresh. No causal OFF/ON attribution.",
    }


def routes(path):
    rows = []
    for line in path.read_text().splitlines():
        header, active, extra = line.split("|")
        layer, tokens = map(int, header.split())
        ids = list(map(int, extra.split()))
        active = list(map(int, active.split()))
        if tokens != 1024 or len(ids) != 9 * tokens or any(x < 0 or x >= 64 for x in ids):
            raise ValueError("unexpected route geometry")
        first = list(dict.fromkeys(ids))
        rows.append({"layer": layer, "tokens": tokens,
                     "full_count": len(ids), "first_count": len(first),
                     "active_count": len(active)})
    if [r["layer"] for r in rows] != list(range(38)):
        raise ValueError("expected exactly 38 prior 1024 route records")
    full = sum(r["full_count"] for r in rows)
    first = sum(r["first_count"] for r in rows)
    return {
        "path": str(path), "sha256": sha(path), "rows": rows,
        "full_entries": full, "first_entries": first,
        "removed_entries": full - first,
        "first_count_range": [min(r["first_count"] for r in rows),
                              max(r["first_count"] for r in rows)],
        "logical_int_bytes_avoided": 4 * (full - first),
        "logical_key_bytes_avoided_aarch64": 8 * (full - first),
        "interpretation": "Older saved route, not the new ARM-router diagnostic. Counts are logical materialized entries, not allocator traffic or measured time.",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    e = a.repo / "docs/measurements/lfm2_tl2_prefill1000_20261016"
    src = ["Applications/CausalLM/models/lfm2_moe/lfm2_moe_layer.cpp",
           "Applications/CausalLM/models/lfm2_moe/lfm2_moe_layer.h",
           "Applications/CausalLM/models/lfm2_moe/expert_lru.h",
           "nntrainer/tensor/htp_backend/htp_compute_ops.cpp"]
    result = {"source_sha256": {p: sha(a.repo / p) for p in src},
              "diagnostics": [profile(e / f"continue-router-arm-diagnostic{n}.log", n)
                              for n in (1024, 2048)],
              "older_routes": routes(e / "lfm2-prefill.trace")}
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(a.output),
                      "topk_us": [x["sum_us"]["topk"] for x in result["diagnostics"]],
                      "old_route_entries": [result["older_routes"][x]
                                            for x in ("full_entries", "first_entries")]}, indent=2))


if __name__ == "__main__":
    main()
