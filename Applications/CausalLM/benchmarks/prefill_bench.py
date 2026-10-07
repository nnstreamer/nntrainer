#!/usr/bin/env python3
"""
prefill_bench.py — pure prefill (num_to_generate=0) speed of the deployed
Qwen3-0.6B Q4_0 model on Android via the nntrainer CausalLM CLI binary,
at exact input token lengths.

Purpose: inference-prefill upper-bound baseline for the QAT training speed
analysis (training step = fwd + ~2x bwd ~= 3x prefill compute), measured on
the same device / model / thread count the QAT runs used (NNTR_NUM_THREADS=4).

Usage (run on the Ubuntu PC, device connected via adb):
  python3 prefill_bench.py --lengths 32,64,96,128,192,256,320,384,512 --threads 4
  python3 prefill_bench.py --lengths 192,192,192 --threads 4        # thermal drift tail
  python3 prefill_bench.py --lengths 192 --threads 1,2,6            # thread sweep

Appends one JSON line per run to --out.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time

MODEL_DIR = "/data/local/tmp/nntrainer/causallm/models/qwen3-0.6b"
WORKDIR = "/data/local/tmp/nntrainer/causallm"
BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
TMP = os.path.join(BENCH_DIR, ".prefill_bench_tmp")
os.makedirs(TMP, exist_ok=True)

PREFILL_RE = re.compile(r"prefill:\s*(\d+)\s+tokens,\s*(\d+)\s*ms,\s*([-\d\.eE+]+)\s*TPS")
GEN_RE = re.compile(r"generation:\s*(\d+)\s+tokens,\s*(\d+)\s*ms")
TOTAL_RE = re.compile(r"total:\s*(\d+)\s*ms")
MEM_RE = re.compile(r"peak memory:\s*(\d+)\s*KB")


def adb_shell(cmd, timeout=1800):
    r = subprocess.run(["adb", "shell", cmd], capture_output=True, text=True, timeout=timeout)
    return r.stdout + r.stderr


class ExactLenText:
    """Builds prompts whose host-side token count is exactly L."""

    def __init__(self):
        os.makedirs(TMP, exist_ok=True)
        local = os.path.join(TMP, "tokenizer.json")
        if not os.path.exists(local):
            subprocess.run(["adb", "pull", f"{MODEL_DIR}/tokenizer.json", local],
                           check=True, capture_output=True)
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(local)
        corpus = ("The quick brown fox jumps over the lazy dog. "
                  "Large language models run on-device for privacy and low latency. "
                  "오늘 날씨가 좋아서 산책을 나갔습니다. 온디바이스 모델이 문장을 생성합니다. ") * 60
        self.ids = self.tok.encode(corpus).ids

    def get(self, L):
        n, text, got = L, None, None
        for _ in range(24):
            text = self.tok.decode(self.ids[:n])
            got = len(self.tok.encode(text).ids)
            if got == L:
                return text, got
            n = max(4, n + (L - got))
        return text, got  # caller records the achieved count; app prints the true one


def fetch_device_config():
    return json.loads(adb_shell(f"cat {MODEL_DIR}/nntr_config.json"))


def push_config(cfg):
    local = os.path.join(TMP, "nntr_config_push.json")
    with open(local, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False)
    subprocess.run(["adb", "push", local, f"{MODEL_DIR}/nntr_config.json"],
                   check=True, capture_output=True)


def start_sampler(interval=1.0):
    """Sample per-cluster cpufreq + cpu thermal zones while a run is active."""
    samples = []
    stop = threading.Event()

    def snap():
        cmd = ("for c in /sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq; do "
               "echo -n \"$(cat $c 2>/dev/null) \"; done; echo '|'; "
               "for z in /sys/class/thermal/thermal_zone*; do "
               "t=$(cat $z/type 2>/dev/null); case $t in cpuss*|cpu-*) "
               "echo -n \"$t:$(cat $z/temp 2>/dev/null) \";; esac; done")
        out = adb_shell(cmd, timeout=30)
        return out.strip().replace("\r", "").replace("\n", " ")

    def loop():
        while not stop.is_set():
            try:
                samples.append((time.time(), snap()))
            except Exception:
                pass
            stop.wait(interval)

    th = threading.Thread(target=loop, daemon=True)
    th.start()
    return samples, stop, th


def summarize_samples(samples):
    """Per-run: min freq per core index + max temp per zone."""
    if not samples:
        return {}
    freq_min, freq_max, temps_max = None, None, {}
    for _, s in samples:
        if "|" not in s:
            continue
        fpart, tpart = s.split("|", 1)
        freqs = [int(x) for x in fpart.split() if x.strip().isdigit()]
        if freqs:
            freq_min = freqs if freq_min is None else [min(a, b) for a, b in zip(freq_min, freqs)]
            freq_max = freqs if freq_max is None else [max(a, b) for a, b in zip(freq_max, freqs)]
        for m in re.finditer(r"(cpuss-\d|cpu-\d-\d):(-?\d+)", tpart):
            temps_max[m.group(1)] = max(temps_max.get(m.group(1), -273000), int(m.group(2)))
    return {
        "freq_min_khz": freq_min,
        "freq_max_khz": freq_max,
        "temp_max_c": {k: round(v / 1000.0, 1) for k, v in temps_max.items()},
        "n_samples": len(samples),
    }


def run_once(threads, L, text, rep, out_f, base_cfg, record=True, taskset=None):
    cfg = dict(base_cfg)
    cfg["init_seq_len"] = L
    cfg["num_to_generate"] = 0
    cfg["batch_size"] = 1
    cfg["sample_input"] = text
    cfg.pop("lora_q4_file_name", None)   # pure base model, no adapter
    cfg["lora_rank"] = 0
    cfg["lora_alpha"] = 0
    cfg["lora_target"] = []
    push_config(cfg)

    samples, stop_ev, th = start_sampler()
    t_wall0 = time.time()
    pin = f"taskset {taskset} " if taskset else ""
    out = adb_shell(
        f"cd {WORKDIR} && LD_LIBRARY_PATH={WORKDIR} "
        f"OMP_NUM_THREADS={threads} NNTR_NUM_THREADS={threads} "
        f"{pin}./nntrainer_causallm {MODEL_DIR}")
    wall = time.time() - t_wall0
    stop_ev.set()
    th.join(timeout=10)

    rec = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "length_target": L,
        "threads": threads,
        "rep": rep,
        "taskset": taskset,
        "wall_s": round(wall, 1),
    }
    m = PREFILL_RE.search(out)
    if m:
        rec.update({"prefill_tokens": int(m.group(1)), "prefill_ms": int(m.group(2)),
                    "prefill_tps": float(m.group(3))})
    g = GEN_RE.search(out)
    if g:
        rec["gen_tokens"] = int(g.group(1))
    t = TOTAL_RE.search(out)
    if t:
        rec["total_ms"] = int(t.group(1))   # includes model load
    mm = MEM_RE.search(out)
    if mm:
        rec["peak_mem_kb"] = int(mm.group(1))
    rec["device_samples"] = summarize_samples(samples)
    if not m:
        rec["error"] = out[-2000:]

    if record:
        with open(out_f, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if m:
        print(f"[{rec['ts']}] L={rec['prefill_tokens']:>4} t={threads} rep={rep} "
              f"prefill={rec['prefill_ms']:>6} ms  {rec['prefill_tps']:>7.1f} tok/s  "
              f"wall={rec['wall_s']}s", flush=True)
    else:
        print(f"[{rec['ts']}] L={L} t={threads} rep={rep} FAILED (see {out_f})", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", default="32,64,96,128,192,256,320,384,512")
    ap.add_argument("--threads", default="4")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--taskset", default=None,
                    help="hex CPU affinity mask passed to taskset (e.g. F8)")
    ap.add_argument("--warmup", type=int, default=0,
                    help="N unrecorded warmup runs at L=192 before the sweep")
    ap.add_argument("--out", default=os.path.join(BENCH_DIR, "prefill_results.jsonl"))
    args = ap.parse_args()

    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    threads_list = [int(x) for x in args.threads.split(",") if x.strip()]

    print(f"Pulling tokenizer + config from device...", flush=True)
    builder = ExactLenText()
    base_cfg = fetch_device_config()

    warm_text, _ = builder.get(192)
    for i in range(args.warmup):
        print(f"== warmup {i + 1}/{args.warmup}", flush=True)
        run_once(threads_list[0], 192, warm_text, 0, args.out, base_cfg, record=False)

    for threads in threads_list:
        for L in lengths:
            for rep in range(1, args.reps + 1):
                text, got = builder.get(L)
                print(f"== L={L} (host tokens={got}) threads={threads} rep={rep}", flush=True)
                run_once(threads, L, text, rep, args.out, base_cfg, taskset=args.taskset)

    # restore a benign config on device (original sample_input, adapter removed)
    base_cfg["num_to_generate"] = 0
    push_config(base_cfg)
    print("Done. Results in", args.out, flush=True)


if __name__ == "__main__":
    main()
