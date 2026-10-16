#!/usr/bin/env python3
"""Read the four frozen model stage logs; never run a benchmark."""
import argparse
import hashlib
import json
import pathlib
import re


STAGES = ("qprep", "dma", "qk", "dequant", "softmax", "pquant", "pv",
          "oupd", "store", "blocks", "total", "kcycles")
TRACE = re.compile(
    r"handle=(\d+) n_q=(\d+) cache=(\d+)\.\.(\d+) append=(\d+) "
    r"err=(0x[0-9a-fA-F]+) wall_us=(\d+) dsp_us=(\d+) append_us=(\d+) "
    r"attn_us=(\d+) quant_us=(\d+) stage_us=(\d+) bake_us=(\d+)")
TRACE_FIELDS = ("handle", "n_q", "cache_from", "cache_to", "append", "err",
                "wall_us", "dsp_us", "append_us", "attn_us", "quant_us",
                "stage_us", "bake_us")
PREFIX = re.compile(r"^\d\d-\d\d \d\d:\d\d:\d\d\.\d+\s+(\d+)\s+\d+\s+[A-Z]\s+\S+: ")


def need(condition, message):
    if not condition:
        raise ValueError(message)


def provenance(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def parse_run(evidence, n, variant):
    stem = f"continue-attn-raw-planes-fixed-stage{n}-{variant}"
    log = evidence / (stem + ".logcat.txt")
    meta_path = evidence / (stem + ".json")
    meta = json.loads(meta_path.read_text())
    need(meta["valid"] and meta["returncode"] == 0 and meta["ids_match_reference"], stem + ": failed run")
    need(meta["profile"] and meta["instrumented"] and
         meta["performance_comparison_eligible"] is False, stem + ": instrumentation scope")
    need(meta["prefill_tokens"] == n and meta["decode_tokens"] == 64 and
         meta["moe_cache_experts"] == 16, stem + ": shape/capacity")
    for key, value in {"NNTR_HTP_ATTN_TRACE": "1", "NNTR_HTP_PAGE_CACHE": "0",
                       "NNTR_MOE_CACHE_GROW": "0", "NNTR_MOE_PRELOAD_EXPERTS": "128"}.items():
        need(meta["environment"][key] == value, stem + ": " + key)
    pending = None
    pids = set()
    calls = []
    for lineno, line in enumerate(log.read_text().splitlines(), 1):
        kind = "trace" if "HTP attn trace q:" in line else "stages" if "HTP attn stages q:" in line else None
        if kind is None:
            continue
        prefix = PREFIX.match(line)
        need(prefix is not None, f"{stem}:{lineno}: missing process identity")
        pids.add(int(prefix.group(1)))
        payload = line.split(f"HTP attn {kind} q: ", 1)[1]
        if kind == "trace":
            need(pending is None, f"{stem}:{lineno}: missing stage record")
            match = TRACE.fullmatch(payload)
            need(match is not None, f"{stem}:{lineno}: malformed trace")
            pending = {k: int(v, 16) if k == "err" else int(v)
                       for k, v in zip(TRACE_FIELDS, match.groups())}
            need(pending["err"] == 0 and pending["n_q"] == n and
                 pending["cache_from"] == 0 and pending["cache_to"] == n and
                 pending["append"] == n, f"{stem}:{lineno}: wrong call cohort")
        else:
            need(pending is not None, f"{stem}:{lineno}: orphan stage record")
            expected = ["n_q", *STAGES]
            fields = payload.split(" ")
            need(len(fields) == len(expected), f"{stem}:{lineno}: wrong field count")
            stage = {}
            for key, field in zip(expected, fields):
                m = re.fullmatch(re.escape(key) + r"=(\d+)", field)
                need(m is not None, f"{stem}:{lineno}: malformed/out-of-order {key}")
                stage[key] = int(m.group(1))
            need(stage["n_q"] == n and stage["blocks"] == {1024: 160, 2048: 576}[n],
                 f"{stem}:{lineno}: wrong stage geometry")
            need(stage["total"] >= sum(stage[k] for k in STAGES[:9]),
                 f"{stem}:{lineno}: inconsistent nested counters")
            calls.append({"wrapper": pending, "stage": stage})
            pending = None
    need(pending is None and len(calls) == 10 and len(pids) == 1,
         stem + ": expected exactly 10 complete calls from one process")
    sums = {k: sum(x["stage"][k] for x in calls) for k in STAGES}
    sums["unbucketed_us"] = sums["total"] - sum(sums[k] for k in STAGES[:9])
    wrappers = {k: sum(x["wrapper"][k] for x in calls)
                for k in TRACE_FIELDS[6:]}
    return {"input": provenance(log), "metadata": provenance(meta_path),
            "pid": pids.pop(), "calls": calls, "sum": sums,
            "wrapper_sum": wrappers, "ids_sha256": meta["ids_sha256"],
            "profile_tps_not_a_normal_performance_result": True}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evidence", type=pathlib.Path, required=True)
    ap.add_argument("--output", type=pathlib.Path, required=True)
    a = ap.parse_args()
    out = {"scope": "One instrumented control/candidate pair per shape; 10 complete attention calls each. Not a normal TPS estimate.", "shapes": {}}
    for n in (1024, 2048):
        runs = {v: parse_run(a.evidence, n, v) for v in ("control", "candidate")}
        need(runs["control"]["ids_sha256"] == runs["candidate"]["ids_sha256"], "IDs changed")
        delta = {k: runs["candidate"]["sum"][k] - runs["control"]["sum"][k]
                 for k in runs["control"]["sum"]}
        runs["delta_candidate_minus_control"] = delta
        runs["body_change_percent"] = 100 * delta["total"] / runs["control"]["sum"]["total"]
        runs["qk_plus_softmax_delta_us"] = delta["qk"] + delta["softmax"]
        runs["pv_plus_oupd_delta_us"] = delta["pv"] + delta["oupd"]
        out["shapes"][str(n)] = runs
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({n: {"calls_per_arm": 10, "body_delta_us": s["delta_candidate_minus_control"]["total"],
                          "qk_plus_softmax_delta_us": s["qk_plus_softmax_delta_us"],
                          "pv_plus_oupd_delta_us": s["pv_plus_oupd_delta_us"]}
                      for n, s in out["shapes"].items()}, indent=2))


if __name__ == "__main__":
    main()
