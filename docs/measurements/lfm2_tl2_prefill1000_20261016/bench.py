#!/usr/bin/env python3
"""Measure one isolated S26 run and retain its command, output and thermal data."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time

ROOT = Path(__file__).resolve().parent
SETTINGS_PATH = ROOT / "device.json"
SETTINGS = json.loads(SETTINGS_PATH.read_text()) if SETTINGS_PATH.exists() else {}
SERIAL = os.environ.get("ANDROID_SERIAL", SETTINGS.get("serial"))
DEVICE = os.environ.get("NNTR_BENCH_DEVICE_ROOT", SETTINGS.get("device_root"))
if not SERIAL or not DEVICE:
    raise SystemExit("Set ANDROID_SERIAL and NNTR_BENCH_DEVICE_ROOT (or a local device.json)")
ADB = ["adb", "-s", SERIAL]
INFERENCE_PROCESSES = "nntr_causallm nntrainer_causallm"
DEVICE_CHECKS = ("gemma4_prefill_device_check repeat_prompt_check "
                 "lfm2_snapshot_check lfm2_prefill_replay "
                 "lfm2_conv_postgate_check lfm2_conv_ion_read_check")


def shell(command):
    return subprocess.check_output(ADB + ["shell", command], text=True, timeout=20)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant")
    parser.add_argument("tag")
    parser.add_argument("--model", default="model")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--memory", action="store_true", help="Sample the full process lifetime, including loading")
    parser.add_argument("--memory-period-ms", type=int, default=20)
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("--unset-env", action="append", default=[])
    args = parser.parse_args()
    config_text = shell("cat " + shlex.quote(DEVICE + "/" + args.model + "/nntr_config.json"))
    config = json.loads(config_text)
    if config.get("moe_cache_experts") != 16:
        parser.error("The model must use moe_cache_experts=16")
    idle_since = None
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        busy = shell(f"pidof {INFERENCE_PROCESSES} {DEVICE_CHECKS} || true").strip()
        if busy:
            idle_since = None
        elif idle_since is None:
            idle_since = time.monotonic()
        elif time.monotonic() - idle_since >= 5:
            break
        time.sleep(1)
    else:
        raise RuntimeError("Device stayed busy; no benchmark started")
    runtime = f"{DEVICE}/{args.variant}"
    ids_file = f"{DEVICE}/{args.tag}.ids"
    environment = {
        "LD_LIBRARY_PATH": runtime,
        "ADSP_LIBRARY_PATH": runtime,
        "NNTR_NUM_THREADS": "8",
        "NNTR_HTP_PAGE_CACHE": "0",
        "NNTR_LOAD_PREFETCH": "0",
        "NNTR_MOE_CACHE_GROW": "0",
        "NNTR_MOE_PRELOAD_EXPERTS": "128",
        "NNTR_GENERATED_IDS": ids_file,
        "NNTR_INIT_TRACE": "1",
    }
    if args.profile:
        environment.update(NNTR_LAYER_PROFILE="1", NNTR_HTP_PROFILE="1", NNTR_OP_TIME="prefill")
    environment.update(dict(value.split("=", 1) for value in args.env))
    for key in args.unset_env:
        environment.pop(key, None)
    for key in ("NNTR_HTP_PAGE_CACHE", "NNTR_MOE_CACHE_GROW"):
        if environment.get(key) != "0":
            parser.error(f"{key}=0 is required for the fixed-cache comparison")
    if environment.get("NNTR_MOE_PRELOAD_EXPERTS") != "128":
        parser.error("NNTR_MOE_PRELOAD_EXPERTS=128 is required for the comparison")
    memory_file = DEVICE + "/" + args.tag + ".memory.json"
    invocation = [runtime + "/nntr_causallm", DEVICE + "/" + args.model]
    if args.memory:
        invocation = [DEVICE + "/fixture-support/memory_monitor", "--output", memory_file,
                      "--period-ms", str(args.memory_period_ms), "--"] + invocation
    command = "cd " + shlex.quote(DEVICE) + " && " + shlex.join(
        ["env"] + [f"{key}={value}" for key, value in environment.items()]
        + ["timeout", "150"] + invocation
    )
    before = shell("dumpsys thermalservice")
    timestamp = shell("date '+%m-%d %H:%M:%S.000'").strip()
    start = time.monotonic()
    log_path = ROOT / f"{args.tag}.log"
    log = log_path.open("wb")
    process = subprocess.Popen(ADB + ["shell", command], stdout=log, stderr=subprocess.STDOUT)
    observed = []
    while process.poll() is None:
        pids = shell(f"pidof {INFERENCE_PROCESSES} || true").split()
        checks = shell(f"pidof {DEVICE_CHECKS} || true").split()
        if len(pids) > 1 or checks:
            observed.append({"elapsed_s": time.monotonic() - start,
                             "pids": pids, "device_check_pids": checks})
        if time.monotonic() - start > 170:
            process.kill()
            break
        time.sleep(0.5)
    process.wait()
    log.close()
    output = log_path.read_bytes()
    after = shell("dumpsys thermalservice")
    if args.profile:
        logcat = shell("logcat -d -v threadtime -T " + shlex.quote(timestamp))
        # Retain accelerator diagnostics without unrelated phone-app logs.
        selected = [line for line in logcat.splitlines()
                    if re.search(r"nntr|hexkl|HTP attn trace|LFM2-MOE-STAGES|LFM-PARTIAL", line, re.I)]
        (ROOT / f"{args.tag}.logcat.txt").write_text("\n".join(selected) + "\n")
    text = output.decode(errors="replace")
    prefill = re.search(r"prefill: (\d+) tokens, ([\d.]+) ms, ([\d.]+) TPS", text)
    decode = re.search(r"generation: (\d+) tokens, ([\d.]+) ms, ([\d.]+) TPS", text)
    peak = re.search(r"peak memory: (\d+) KB", text)
    footprint = re.search(r"honest RssAnon\+accel: (\d+) KB, ru_maxrss: (\d+) KB", text)
    result = dict(tag=args.tag, variant=args.variant, model=args.model, profile=args.profile,
                  command=command, environment=environment, returncode=process.returncode,
                  concurrent_processes=observed, thermal_before=before, thermal_after=after,
                  wall_seconds=time.monotonic()-start, unix_time=time.time(),
                  device_logcat_start=timestamp,
                  moe_cache_experts=config["moe_cache_experts"],
                  model_config_sha256=hashlib.sha256(config_text.encode()).hexdigest())
    if peak:
        result["app_peak_memory_kib"] = int(peak[1])
        result["app_peak_scope"] = "CausalLM::run only; excludes model loading"
        result["app_peak_sample_interval_ms"] = int(environment.get("NNTR_FOOTPRINT_MS", "100"))
    if footprint:
        result["app_anon_plus_accel_peak_kib"] = int(footprint[1])
        result["process_lifetime_ru_maxrss_kib"] = int(footprint[2])
    if args.memory and not process.returncode:
        memory_text = shell("cat " + shlex.quote(memory_file))
        memory = json.loads(memory_text)
        (ROOT / f"{args.tag}.memory.json").write_text(memory_text)
        composite = memory["peak_anon_plus_fdinfo_dedup_or_raw_fallback_bytes"]
        result["full_run_sampled_peak_kib"] = composite / 1024 if composite is not None else None
        result["full_run_peak_scope"] = "Process RssAnon + deduplicated fdinfo size allocations; includes loading; excludes unreported DSP/driver memory"
        result["full_run_memory_sample_count"] = memory["sample_count"]
        result["full_run_memory_mean_interval_ms"] = memory["sample_interval_ms"]["mean"]
        result["full_run_memory_report_sha256"] = hashlib.sha256(memory_text.encode()).hexdigest()
    if prefill and decode and not process.returncode:
        ids = shell("cat " + shlex.quote(ids_file))
        (ROOT / f"{args.tag}.ids").write_text(ids)
        result.update(prefill_tokens=int(prefill[1]), prefill_ms=float(prefill[2]),
                      prefill_tps=float(prefill[3]), decode_tokens=int(decode[1]),
                      decode_ms=float(decode[2]), decode_tps=float(decode[3]),
                      ids_sha256=hashlib.sha256(ids.encode()).hexdigest())
    expected_ids = {
        "model": "b6c1b801f95caded5399ec685cce0f900a20c3f9df760482dd7b5c9a62a5b92e",
        "model2048": "2bc7afc3c7d8e4387c8d9c7378d6792a1ef0fb56c2635f73b95e8650ac7fef33",
        "gemma": "d0ca48a0dab9c2a745edb099df3766971d30b69ce4ea72744fa6e9e61d5ed3f3",
    }
    result["ids_match_reference"] = result.get("ids_sha256") == expected_ids.get(args.model)
    result["valid"] = bool(prefill and decode and peak and not process.returncode
                           and not observed and result["ids_match_reference"]
                           and (not args.memory or result.get("full_run_sampled_peak_kib") is not None))
    result["instrumented"] = any(environment.get(key, "0") != "0" for key in
                                 ("NNTR_LAYER_PROFILE", "NNTR_HTP_PROFILE", "NNTR_HTP_PREFILL_IO_TRACE"))
    result["performance_comparison_eligible"] = result["valid"] and not result["instrumented"]
    (ROOT / f"{args.tag}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k:v for k,v in result.items() if k not in
                     {"command", "environment", "thermal_before", "thermal_after"}}), flush=True)
    if not result["valid"]:
        raise SystemExit("Invalid measurement; see retained log")


if __name__ == "__main__":
    main()
