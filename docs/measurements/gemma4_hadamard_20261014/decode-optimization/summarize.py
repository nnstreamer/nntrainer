#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pool matching unprofiled E2E batches.

Run this file from any directory; pass batch directory names, or omit arguments
for the final cache-16/readers-4 actual-rotation batch. Print JSON without
modifying raw measurements. Actual and shadow experiments must stay separate.
Do not include diagnostic runs or runs using a different binary.
"""
import json
from pathlib import Path
import statistics
import sys

root = Path(__file__).resolve().parent
batches = sys.argv[1:] or ("rotation-only/final-e2e",)
rows = []
settings = []
for batch in batches:
    records = json.loads((root / batch / "runs.json").read_text())
    rows.extend(dict(r, batch=batch) for r in records)
    settings.append(json.loads((root / batch / "settings.json").read_text()))
assert all(s["profile"] is False for s in settings)
assert all(s["environment"] == settings[0]["environment"] for s in settings)
assert all(s["serial"] == "R5KL30G6MCX" for s in settings)
assert all(s.get("measurement") == settings[0].get("measurement") for s in settings)
assert all(s.get("runtime_sha256") == settings[0].get("runtime_sha256") for s in settings)
summary = {"batches": batches, "environment": settings[0]["environment"],
           "measurement": settings[0].get("measurement", "legacy_unlabelled"),
           "runtime_sha256": settings[0].get("runtime_sha256")}
keys = ("prefill_ms", "decode_ms", "inference_ms", "e2e_ms")
for variant in ("off", "both_hmx"):
    records = [r for r in rows if r["variant"] == variant]
    hashes = {r["output_ids_sha256"] for r in records}
    assert len(hashes) == 1
    summary[variant] = dict(runs=len(records), output_ids_sha256=hashes.pop())
    for key in keys:
        values = [r[key] for r in records]
        summary[variant][key] = statistics.median(values)
        summary[variant][key + "_range"] = [min(values), max(values)]
    summary[variant]["prefill_tps"] = 1024000 / summary[variant]["prefill_ms"]
    summary[variant]["decode_tps"] = 127000 / summary[variant]["decode_ms"]
summary["time_delta_pct"] = {
    k: 100 * (summary["both_hmx"][k] / summary["off"][k] - 1) for k in keys
}
summary["tps_delta_pct"] = {
    k: 100 * (summary["both_hmx"][k] / summary["off"][k] - 1)
    for k in ("prefill_tps", "decode_tps")
}
print(json.dumps(summary, indent=2))
