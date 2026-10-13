#!/usr/bin/env python3
"""Measure timed host staging and verify that its payloads were consumed.

Memory/profile runs are diagnostic screens, never clean TPS samples.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import threading
import time

import measure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--count', type=int, required=True)
    parser.add_argument('--length', type=int, choices=[1024, 2048], default=1024)
    parser.add_argument('--memory', action='store_true')
    parser.add_argument('--io-profile', action='store_true')
    parser.add_argument('--aligned', action='store_true',
                        help='Read staged weights directly into aligned host buffers')
    parser.add_argument('--cpu-mask', help='Main-process taskset mask')
    parser.add_argument('--reader-cpus', help='Explicit comma-separated reader CPUs')
    args = parser.parse_args()
    assert 0 <= args.count <= 160
    assert not args.aligned or args.count > 0
    out = Path(__file__).resolve().parent
    env = json.loads((out / 'v13-br256-r7-clean-x1.json').read_text())['env']
    env['NNTR_MOE_HOST_STAGE_EXPERTS'] = str(args.count)
    env['NNTR_MOE_IO_PROFILE'] = '1' if args.io_profile else '0'
    env['NNTR_MOE_HOST_STAGE_ALIGNED'] = '1' if args.aligned else '0'
    if args.reader_cpus is not None:
        assert re.fullmatch(r'\d+(,\d+)*', args.reader_cpus)
        assert all(int(cpu) < 32 for cpu in args.reader_cpus.split(','))
        env['NNTR_MOE_PREFETCH_CPUS'] = args.reader_cpus
    if args.length == 2048:
        measure.REMOTE = '/data/local/tmp/lfm2_prefill1000_20261013'
    log_path = out / (args.label + '.log')
    stop = threading.Event()
    samples = []

    def sample():
        while not stop.wait(0.4):
            if not log_path.exists():
                continue
            match = re.search(r'^NNTR_BENCH_PID=(\d+)$',
                              log_path.read_text(errors='replace'), re.M)
            if not match:
                continue
            pid = match[1]
            try:
                result = subprocess.run(
                    measure.ADB + ['shell', f'cat /proc/{pid}/status /proc/meminfo'],
                    capture_output=True, text=True, timeout=10)
                if result.returncode:
                    continue
                values = {key: int(value) for key, value in re.findall(
                    r'^(VmRSS|VmHWM|RssAnon|RssFile|RssShmem|MemAvailable):\s*(\d+)',
                    result.stdout, re.M)}
                samples.append({'monotonic': time.monotonic(), 'pid': pid,
                                'kb': values})
            except subprocess.TimeoutExpired:
                continue

    worker = threading.Thread(target=sample) if args.memory else None
    if worker:
        worker.start()
    try:
        row = measure.run(args.variant, args.length, 'clean', args.label, env,
                          args.cpu_mask)
    finally:
        stop.set()
        if worker:
            worker.join()
        if args.memory:
            measure.save(out / (args.label + '-memory.json'), samples)
    raw = log_path.read_text()
    active = re.findall(r'\[HTP-HOST-STAGING-ACTIVE\] ([^\n]+)', raw)
    done = re.findall(r'\[HTP-HOST-STAGING-DONE\] ([^\n]+)', raw)
    parse = lambda text: {k: int(v) for k, v in re.findall(r'(\w+)=(\d+)', text)}
    row['host_staging'] = {'requested': args.count,
                           'aligned_requested': args.aligned,
                           'active': [parse(s) for s in active],
                           'done': [parse(s) for s in done]}
    verified = not active and not done if args.count == 0 else (
        len(active) == len(done) == 1 and
        parse(active[0]).get('count') == args.count and
        parse(done[0]).get('count') == args.count and
        parse(done[0]).get('read') == args.count and
        parse(done[0]).get('consumed') == args.count)
    if args.aligned and verified:
        allocation = parse(active[0])
        verified = (allocation.get('aligned') == 1 and
                    allocation['payload_bytes'] <=
                    allocation.get('allocated_bytes', 0) <= 384 * 1024 * 1024)
    row['host_staging']['verified'] = verified
    row['valid'] = row['valid'] and verified
    row['clean_timing_sample'] = not args.memory and not args.io_profile
    row['memory_observer'] = args.memory
    row['io_profile'] = args.io_profile
    if samples:
        row['memory_peak_kb'] = {
            key: max(s['kb'].get(key, 0) for s in samples)
            for key in ['VmRSS', 'VmHWM', 'RssAnon', 'RssFile', 'RssShmem']}
        row['min_available_kb'] = min(
            s['kb']['MemAvailable'] for s in samples if 'MemAvailable' in s['kb'])
    measure.save(out / (args.label + '.json'), row)
    print('STAGING EVIDENCE', row['host_staging'],
          'clean_timing_sample=', row['clean_timing_sample'],
          'memory_peak_kb=', row.get('memory_peak_kb'), flush=True)
    assert verified, 'Staging lifecycle/count mismatch'


if __name__ == '__main__':
    main()
