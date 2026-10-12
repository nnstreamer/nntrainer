#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measure a deployed LFM2 runtime with decode disabled in an isolated config.

Preserves the original model/runtime. Full logs, commands, hashes, temperatures
and concurrency checks are retained. HTP aggregate counters include load-time
warmups; OP-TIME=prefill and M0 rows isolate the actual model forward.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--serial', required=True)
    p.add_argument('--runtime', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--environment', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repeats', type=int, default=3)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    adb = ['adb', '-s', a.serial]

    def read(cmd):
        return subprocess.check_output(adb + ['shell', cmd], text=True,
                                       timeout=30).strip()

    def busy():
        r = subprocess.run(adb + ['shell', 'pidof nntrainer_causallm '
            'unittest_hvx_mm_u8i4 unittest_hvx_attn_q unittest_hvx_attn_f16 '
            'unittest_compute_ops_dispatch'], capture_output=True, text=True,
            timeout=10)
        if r.returncode not in (0, 1):
            raise RuntimeError(r.stderr)
        return r.stdout.split()

    def thermal():
        raw = read('dumpsys thermalservice')
        section = raw.split('Current temperatures from HAL:')[-1].split(
            'Current cooling devices')[0]
        return {'status': int(re.search(r'Thermal Status: (\d+)', raw)[1]),
                'temperatures': {n: float(v) for v, n in re.findall(
                    r'Temperature\{mValue=([\d.]+), mType=\d+, mName=(\w+)',
                    section)}, 'raw': raw}

    def hashes():
        return read('cd ' + shlex.quote(a.runtime) + ' && sha256sum '
            'nntrainer_causallm libnntrainer.so libcausallm_core.so '
            'libnntr_hvx_skel.so libccapi-nntrainer.so libc++_shared.so libsdkl.so')

    base = json.loads(a.environment.read_text())
    for key in list(base):
        if any(x in key for x in ('PROFILE', 'TRACE', 'OP_TIME', 'PPL')):
            del base[key]
    original = json.loads(read('cat ' + shlex.quote(a.model + '/nntr_config.json')))
    cfg = dict(original, num_to_generate=0)
    (a.output / 'original-nntr_config.json').write_text(json.dumps(original, indent=2))
    (a.output / 'nntr_config.json').write_text(json.dumps(cfg, indent=2))
    for name in ('config.json', 'generation_config.json'):
        (a.output / name).write_text(read('cat ' + shlex.quote(a.model + '/' + name)))
    remote = '/data/local/tmp/nntr_prefill_audit_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    read('mkdir ' + shlex.quote(remote))
    names = read('ls -1 ' + shlex.quote(a.model)).splitlines()
    for name in names:
        if name != 'nntr_config.json':
            read('ln -s ' + shlex.quote(a.model + '/' + name) + ' ' + shlex.quote(remote + '/' + name))
    subprocess.run(adb + ['push', str(a.output / 'nntr_config.json'), remote + '/nntr_config.json'], check=True, capture_output=True)
    report = {'started': datetime.datetime.now().astimezone().isoformat(),
        'serial': a.serial, 'device': read('getprop ro.product.model'),
        'soc': read('getprop ro.soc.model'), 'runtime': a.runtime,
        'original_model': a.model, 'isolated_model': remote,
        'binary_sha256': hashes(), 'boot_id': read('cat /proc/sys/kernel/random/boot_id'),
        'method': '1024 input tokens, zero decode steps; fresh process per run, no page cache flush; 20s quiet gaps; thermal status zero, AP <=40C, BAT <=32C, SKIN <36C. Profiles separated from clean timing. HTP aggregate includes load warmups.',
        'runs': []}
    (a.output / 'runner.py').write_text(Path(__file__).read_text())

    def save():
        (a.output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')

    order = [('clean', i + 1) for i in range(a.repeats)] + [('profile', i + 1) for i in range(2)]
    for mode, repeat in order:
        quiet = time.monotonic()
        while True:
            if busy():
                quiet = time.monotonic()
            t = thermal()
            temps = t['temperatures']
            if time.monotonic() - quiet >= 20 and t['status'] == 0 and temps.get('AP', 999) <= 40 and temps.get('BAT', 999) <= 32 and temps.get('SKIN', 999) < 36:
                break
            time.sleep(2)
        env = dict(base, NNTR_HTP_PROFILE='0', NNTR_OP_TIME='0', NNTR_HTP_DMA_TRACE='0')
        if mode == 'profile':
            env.update(NNTR_HTP_PROFILE='2', NNTR_OP_TIME='prefill', NNTR_M0_PROFILE='1', NNTR_HTP_ATTN_TRACE='1', NNTR_HTP_MATH_PROFILE='1')
        command = 'cd ' + shlex.quote(a.runtime) + ' && env ' + ' '.join(k + '=' + shlex.quote(str(v)) for k, v in env.items()) + ' ./nntrainer_causallm ' + shlex.quote(remote)
        name = f'{mode}-{repeat}'
        (a.output / (name + '-thermal-before.txt')).write_text(t['raw'])
        print('START', name, temps, flush=True)
        tracefile = (a.output / (name + '-logcat.txt')).open('w')
        lc = subprocess.Popen(adb + ['logcat', '-T', '1', '-v', 'threadtime', 'nntrainer:I', '*:S'], stdout=tracefile, stderr=subprocess.STDOUT)
        overlap = False
        try:
            with (a.output / (name + '.log')).open('w') as f:
                proc = subprocess.Popen(adb + ['shell', command], stdout=f, stderr=subprocess.STDOUT)
                start = time.monotonic()
                while proc.poll() is None:
                    overlap |= len(busy()) > 1
                    if time.monotonic() - start > 300:
                        proc.terminate()
                        raise RuntimeError('Timed out: inspect device process before retrying')
                    time.sleep(.5)
            time.sleep(.4)
        finally:
            lc.terminate()
            lc.wait(timeout=10)
            tracefile.close()
        after = thermal()
        (a.output / (name + '-thermal-after.txt')).write_text(after['raw'])
        raw = (a.output / (name + '.log')).read_text()
        m = re.search(r'^prefill: (\d+) tokens, (\d+) ms, ([\d.]+) TPS', raw, re.M)
        dec = re.search(r'^generation: (\d+) tokens,', raw, re.M)
        row = {'name': name, 'mode': mode, 'env': env, 'command': command,
               'returncode': proc.returncode, 'overlap': overlap,
               'thermal_before': {k: v for k, v in t.items() if k != 'raw'},
               'thermal_after': {k: v for k, v in after.items() if k != 'raw'},
               'log_sha256': hashlib.sha256(raw.encode()).hexdigest()}
        if m:
            row.update(tokens=int(m[1]), prefill_ms=int(m[2]), tps=float(m[3]))
        report['runs'].append(row)
        save()
        assert proc.returncode == 0 and m and dec and int(dec[1]) == 0, name
        assert int(m[1]) == 1024 and not overlap, name
        assert report['boot_id'] == read('cat /proc/sys/kernel/random/boot_id')
        print('DONE', name, row['prefill_ms'], 'ms', flush=True)
    assert hashes() == report['binary_sha256']
    report['summary'] = {}
    for mode in ('clean', 'profile'):
        times = [r['prefill_ms'] for r in report['runs'] if r['mode'] == mode]
        report['summary'][mode] = {'ms': times, 'median_ms': statistics.median(times), 'mean_ms': statistics.mean(times), 'min_ms': min(times), 'max_ms': max(times)}
    report['finished'] = datetime.datetime.now().astimezone().isoformat()
    save()
    print(json.dumps(report['summary']), flush=True)


if __name__ == '__main__':
    main()
