#!/usr/bin/env python3
"""Strict decoder for the frozen NNTR_LFM2_PARTIAL_TRACE diagnostic.

Exit 2 on missing/malformed/inconsistent evidence, but write the independently
available aggregate summaries. Missing trace fields are null, never zero.
Read-only inputs; output is a new JSON report. Python standard library only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

MARKER = "[LFM-PARTIAL]"
TRACE_RE = re.compile(
    r"\[LFM-PARTIAL\] tok=(?P<tok>\d+) rc=(?P<rc>-?\d+) "
    r"rounds=(?P<rounds>\d+) early_rows=(?P<early_rows>\d+) "
    r"early_batches=(?P<early_batches>\d+) final_first=(?P<final_first>\d+) "
    r"ready_wait_us=(?P<ready_wait_us>\d+) final_wait_us=(?P<final_wait_us>\d+)"
)
FLOAT = r"(?:\d+(?:\.\d*)?|\.\d+)"
KNOWN_KINDS = (
    "RMSNORM", "FC", "CONV1D_GATE", "QK_NORM", "ROPE", "ATTN_M1", "ADD",
    "ROUTER_TOPK", "MOE", "DENSE_FFN", "LM_HEAD",
)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def parse_trace(text, label, expected=64, first_token=0):
    records, errors = [], []
    marker_count = 0
    for line_number, line in enumerate(text.splitlines(), 1):
        if MARKER not in line:
            continue
        marker_count += line.count(MARKER)
        tail = line[line.index(MARKER):].strip()
        match = TRACE_RE.fullmatch(tail)
        if not match:
            errors.append(f"{label}:{line_number}: malformed diagnostic: {tail!r}")
            continue
        record = {key: int(value) for key, value in match.groupdict().items()}
        records.append(record)
        token = record["tok"]
        if record["rc"] != 0:
            errors.append(f"{label}: token {token}: rc={record['rc']}")
        rounds, rows = record["rounds"], record["early_rows"]
        batches, final_first = record["early_batches"], record["final_first"]
        if not (0 <= final_first <= rounds <= 38):
            errors.append(f"{label}: token {token}: invalid rounds/final_first")
        if not (0 <= batches <= rows <= 4 * rounds):
            errors.append(f"{label}: token {token}: invalid rows/batches")
        if (rows == 0) != (batches == 0):
            errors.append(f"{label}: token {token}: rows/batches zero mismatch")
        early_rounds = rounds - final_first
        if not (early_rounds <= rows <= 4 * early_rounds):
            errors.append(f"{label}: token {token}: early rows/rounds inconsistent")
    # A second sink may legitimately have no records. The caller requires one
    # complete sink; any sink with markers must be independently complete.
    if marker_count:
        tokens = [r["tok"] for r in records]
        wanted = list(range(first_token, first_token + expected))
        if len(records) != expected:
            errors.append(f"{label}: expected {expected} records, got {len(records)}")
        if len(set(tokens)) != len(tokens):
            errors.append(f"{label}: duplicate token IDs")
        if tokens != wanted:
            errors.append(f"{label}: token IDs must be consecutive {wanted[0]}..{wanted[-1]}")
    return {"marker_count": marker_count, "records": records, "errors": errors}


def only_tail(text, marker, errors, required=True):
    tails = [line[line.index(marker) + len(marker):].strip()
             for line in text.splitlines() if marker in line]
    if len(tails) != 1:
        if required or tails:
            errors.append(f"{marker}: expected one summary, got {len(tails)}")
        return None
    return tails[0]


def match_tail(text, marker, pattern, errors, required=True):
    tail = only_tail(text, marker, errors, required)
    if tail is None:
        return None
    match = re.fullmatch(pattern, tail)
    if not match:
        errors.append(f"{marker}: malformed summary: {tail!r}")
        return None
    return match.groupdict()


def numeric(values):
    if values is None:
        return None
    return {k: float(v) if "." in v else int(v, 16 if v.startswith("0x") else 10)
            for k, v in values.items()}


def parse_summaries(text, errors):
    out = {}
    for phase in ("prefill", "generation"):
        out[phase] = numeric(match_tail(
            text, f"{phase}: ",
            rf"(?P<tokens>\d+) tokens, (?P<ms>{FLOAT}) ms, (?P<tps>{FLOAT}) TPS",
            errors))
    out["pool"] = numeric(match_tail(
        text, "[HTP] token driver: pool misses=",
        rf"(?P<misses>\d+) misses/token=(?P<misses_per_token>{FLOAT}) "
        rf"miss_wait_us/token=(?P<miss_wait_us_per_token>{FLOAT}) "
        rf"rounds=(?P<rounds>\d+) arm_ms/round=(?P<arm_ms_per_round>{FLOAT}) "
        rf"pgpgin_mib=(?P<pgpgin_mib>{FLOAT}) rss_mib=(?P<rss_mib>{FLOAT}) "
        rf"mem_available_mib=(?P<mem_available_mib>{FLOAT})", errors))
    out["close"] = numeric(match_tail(
        text, "[HTP] token driver: close ",
        rf"tokens=(?P<tokens>\d+) hops/token=(?P<hops_per_token>{FLOAT}) "
        rf"served=(?P<served>\d+) timeouts=(?P<timeouts>\d+) stale=(?P<stale>\d+) "
        rf"id_checked=(?P<id_checked>\d+) id_mismatch=(?P<id_mismatch>\d+) "
        rf"stop_err=(?P<stop_err>0x[0-9a-fA-F]+)", errors))
    out["moe_calls"] = numeric(match_tail(
        text, "[HTP] token driver: moe calls/token=",
        rf"(?P<per_token>{FLOAT}) \(one-at-a-time (?P<one_at_a_time>{FLOAT})\)",
        errors))
    out["round_us"] = numeric(match_tail(
        text, "[HTP] token driver: round us: ",
        r"notice=(?P<notice>\d+) pre=(?P<pre>\d+) read=(?P<read>\d+) "
        r"answer=(?P<answer>\d+) \(arm total\) \| dsp overlap=(?P<overlap>\d+) "
        r"wait=(?P<wait>\d+) anslat=(?P<anslat>\d+) rebind=(?P<rebind>\d+)", errors))
    out["driver_us_per_token"] = numeric(match_tail(
        text, "[HTP] token driver: L0 us/token ",
        rf"rt=(?P<rt>{FLOAT}) dsp_wall=(?P<dsp_wall>{FLOAT}) wake=(?P<wake>{FLOAT}) "
        rf"hop_us=(?P<hop_us>{FLOAT}) arm_fwd=(?P<arm_fwd>{FLOAT}) "
        rf"arm_us=(?P<arm_us>{FLOAT}) arm_n=(?P<arm_n>\d+)", errors))
    pair = numeric(match_tail(text, "[HTP] token driver: expert_pair_reads=",
                              r"(?P<count>\d+)", errors))
    out["expert_pair_reads"] = pair["count"] if pair else None
    partial = numeric(match_tail(text, "[HTP] token driver: partial_ready_rounds=",
                                 r"(?P<count>\d+)", errors, required=False))
    partial_marker = "[HTP] token driver: partial_ready_rounds="
    out["arm_partial_ready_rounds"] = (partial["count"] if partial else
                                       None if partial_marker in text else 0)
    out["arm_partial_count_source"] = (
        "explicit" if partial else "malformed" if partial_marker in text else
        "absent; production emits this line only when nonzero")
    graph = only_tail(text, "[HTP] graph per-kind us/token: ", errors)
    out["graph_us_per_token"] = None
    if graph is not None:
        match = re.fullmatch(rf"(.+) \| ops/wall=({FLOAT}) compute_mhz=(\d+)", graph)
        if match:
            body = match[1]
            entries = list(re.finditer(r"([A-Z][A-Z0-9_]*)=(\d+)(?:\(net of miss wait (\d+)\))?", body))
            remainder = re.sub(r"([A-Z][A-Z0-9_]*)=(\d+)(?:\(net of miss wait (\d+)\))?", "", body)
            kinds = [m[1] for m in entries]
            if remainder.strip() or tuple(kinds) != KNOWN_KINDS:
                errors.append("graph per-kind: malformed, missing, duplicate, or unexpected kinds")
            elif any(m[3] is not None for m in entries if m[1] != "MOE"):
                errors.append("graph per-kind: unexpected net annotation")
            else:
                values = {m[1]: int(m[2]) for m in entries}
                moe = next(m for m in entries if m[1] == "MOE")
                if moe[3] is None:
                    errors.append("graph per-kind: missing net MoE field")
                else:
                    values["MOE_net_of_miss_wait"] = int(moe[3])
                    values["ops_wall_ratio"] = float(match[2])
                    values["compute_mhz"] = int(match[3])
                    out["graph_us_per_token"] = values
        else:
            errors.append(f"graph per-kind: malformed summary: {graph!r}")
    return out


def read_run(stem, knob, expected, first_token):
    errors, files, texts = [], {}, {}
    for suffix in (".log", ".logcat.txt", ".json", ".ids"):
        path = Path(str(stem) + suffix)
        try:
            data = path.read_bytes()
            texts[suffix] = data.decode("utf-8")
            files[suffix] = {"path": str(path), "bytes": len(data), "sha256": sha256(data)}
        except (OSError, UnicodeError) as exc:
            errors.append(f"{path}: {exc}")
            texts[suffix] = ""
    try:
        meta = json.loads(texts[".json"])
        if not isinstance(meta, dict):
            raise ValueError("metadata must be an object")
    except (ValueError, TypeError) as exc:
        errors.append(f"metadata: {exc}")
        meta = {}
    required = {"returncode": 0, "valid": True, "profile": True,
                "instrumented": True, "performance_comparison_eligible": False,
                "decode_tokens": expected, "moe_cache_experts": 16,
                "ids_match_reference": True, "concurrent_processes": []}
    for key, value in required.items():
        if key not in meta or meta[key] != value:
            errors.append(f"metadata {key}: expected {value!r}, got {meta.get(key)!r}")
    env = meta.get("environment", {})
    if not isinstance(env, dict):
        errors.append("metadata environment is not an object")
        env = {}
    for key, value in {"NNTR_HTP_PAGE_CACHE": "0", "NNTR_MOE_CACHE_GROW": "0",
                       "NNTR_MOE_PRELOAD_EXPERTS": "128", "NNTR_HTP_PROFILE": "1",
                       "NNTR_LFM2_DECODE_PARTIAL_READY": str(knob)}.items():
        if env.get(key) != value:
            errors.append(f"environment {key}: expected {value}, got {env.get(key)!r}")
    ids_text = texts[".ids"]
    ids_lines = ids_text.splitlines()
    ids = None
    if len(ids_lines) != expected + 1 or any(not re.fullmatch(r"\d+", x) for x in ids_lines):
        errors.append(f"IDs: expected exactly {expected + 1} decimal lines")
    else:
        ids = [int(x) for x in ids_lines]
    if files.get(".ids", {}).get("sha256") != meta.get("ids_sha256"):
        errors.append("IDs: actual SHA256 does not match metadata")
    summaries = parse_summaries(texts[".log"], errors)
    for phase, prefix in (("prefill", "prefill"), ("generation", "decode")):
        summary = summaries[phase]
        if summary:
            for key in ("tokens", "ms", "tps"):
                if summary[key] != meta.get(f"{prefix}_{key}"):
                    errors.append(f"{phase}: {key} differs between log and metadata")
    close = summaries["close"]
    if close:
        for key, value in {"tokens": expected, "served": expected, "timeouts": 0,
                           "stale": 0, "id_mismatch": 0, "stop_err": 0}.items():
            if close[key] != value:
                errors.append(f"token close {key}: expected {value}, got {close[key]}")
    graph, pool = summaries["graph_us_per_token"], summaries["pool"]
    if graph and pool:
        residual = graph["MOE"] - pool["miss_wait_us_per_token"] - graph["MOE_net_of_miss_wait"]
        if abs(residual) > 1.1:
            errors.append(f"net MoE does not reconcile with aggregate miss wait: residual {residual} us/token")
    sinks = {suffix: parse_trace(texts[suffix], suffix, expected, first_token)
             for suffix in (".log", ".logcat.txt")}
    trace_errors = []
    for sink in sinks.values():
        trace_errors.extend(sink["errors"])
    populated = [suffix for suffix, sink in sinks.items() if sink["marker_count"]]
    records = None
    if not populated:
        trace_errors.append(f"missing {MARKER} diagnostics: expected {expected} tokens; both sinks contain zero markers")
    elif len(populated) == 2 and sinks[".log"]["records"] != sinks[".logcat.txt"]["records"]:
        trace_errors.append("trace sinks disagree; mirrors must contain identical ordered records")
    if populated and not trace_errors:
        records = sinks[populated[0]]["records"]
        totals = {key: sum(r[key] for r in records)
                  for key in ("rounds", "early_rows", "early_batches", "final_first",
                              "ready_wait_us", "final_wait_us")}
        if not knob and any(totals[key] for key in ("rounds", "early_rows", "early_batches", "final_first", "ready_wait_us")):
            trace_errors.append("partial knob off but partial diagnostic counters are nonzero")
        arm_rounds = summaries["arm_partial_ready_rounds"]
        if arm_rounds is not None and totals["rounds"] < arm_rounds:
            trace_errors.append("ARM publications exceed eligible DSP partial rounds")
        if pool and (totals["early_rows"] > pool["misses"] or totals["rounds"] > pool["rounds"]):
            trace_errors.append("DSP partial counts exceed total miss rows/rounds")
    aggregate = None
    if records and not trace_errors:
        aggregate = {"totals": totals,
                     "per_token": {k: v / expected for k, v in totals.items()},
                     "tokens_with_early_compute": sum(r["early_rows"] > 0 for r in records),
                     "early_rounds": totals["rounds"] - totals["final_first"]}
        if pool:
            aggregate["wait_reconciliation_us_per_token"] = (
                (totals["ready_wait_us"] + totals["final_wait_us"]) / expected
                - pool["miss_wait_us_per_token"])
    errors.extend(trace_errors)
    return {"stem": str(stem), "knob": knob, "files": files, "errors": errors,
            "valid": not errors, "metadata": {k: meta.get(k) for k in (
                "model_config_sha256", "ids_sha256", "ids_match_reference", "instrumented",
                "performance_comparison_eligible", "full_run_sampled_peak_kib",
                "full_run_peak_scope", "full_run_memory_sample_count",
                "full_run_memory_mean_interval_ms")},
            "environment": env, "ids": ids, "summaries": summaries,
            "trace": {"valid": not trace_errors, "sinks": sinks,
                      "source_sinks": populated, "aggregate": aggregate,
                      "records": records}}


def compare(off, on):
    errors = []
    if off["ids"] is None or off["ids"] != on["ids"]:
        errors.append("off/on output IDs differ or are unavailable")
    if off["metadata"]["model_config_sha256"] != on["metadata"]["model_config_sha256"]:
        errors.append("off/on model configuration hashes differ")
    ignored_env = {"NNTR_GENERATED_IDS", "NNTR_LFM2_DECODE_PARTIAL_READY"}
    for key in (set(off["environment"]) | set(on["environment"])) - ignored_env:
        if off["environment"].get(key) != on["environment"].get(key):
            errors.append(f"off/on environment differs: {key}")
    metrics = {
        "decode_tps": ("generation", "tps"),
        "prefill_tps": ("prefill", "tps"),
        "miss_wait_us_per_token": ("pool", "miss_wait_us_per_token"),
        "moe_gross_us_per_token": ("graph_us_per_token", "MOE"),
        "moe_net_us_per_token": ("graph_us_per_token", "MOE_net_of_miss_wait"),
        "moe_calls_per_token": ("moe_calls", "per_token"),
        "dsp_wall_us_per_token": ("driver_us_per_token", "dsp_wall"),
        "arm_read_us_per_round": ("round_us", "read"),
        "arm_answer_us_per_round": ("round_us", "answer"),
    }
    deltas = {}
    for name, (section, field) in metrics.items():
        a, b = off["summaries"].get(section), on["summaries"].get(section)
        if a is not None and b is not None:
            x, y = a[field], b[field]
            deltas[name] = {"off": x, "on": y, "delta": y - x,
                            "change_percent": (y / x - 1) * 100 if x else None}
    return {"errors": errors, "ids_equal": off["ids"] is not None and off["ids"] == on["ids"],
            "deltas": deltas}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--off-stem", type=Path, required=True)
    parser.add_argument("--on-stem", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-tokens", type=int, default=64)
    parser.add_argument("--first-token", type=int, default=0)
    args = parser.parse_args(argv)
    if args.expected_tokens < 1:
        parser.error("--expected-tokens must be positive")
    off = read_run(args.off_stem, 0, args.expected_tokens, args.first_token)
    on = read_run(args.on_stem, 1, args.expected_tokens, args.first_token)
    comparison = compare(off, on)
    report = {"schema": 1, "valid": off["valid"] and on["valid"] and not comparison["errors"],
              "off": off, "on": on, "comparison": comparison,
              "limits": [
                  "Instrumented runs describe mechanism; use separate normal repeated runs for performance decisions.",
                  "Net MoE is the aggregate per-token mean from graph per-kind summary; per-token net compute is unavailable.",
                  "ARM partial_ready_rounds counts publications; DSP early_rows/batches count successful early compute.",
                  "Missing diagnostics are errors with null aggregates, never evidence of zero early compute.",
                  "Trace wait and pool miss wait use separate timer reads; tiny rounding/instrumentation offsets are expected.",
                  "IDs are independently compared between these runs; reference match is also checked as reported by the runner.",
              ]}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"valid": report["valid"], "output": str(args.output),
                      "off_errors": off["errors"], "on_errors": on["errors"],
                      "comparison": comparison}, indent=2))
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    sys.exit(main())
