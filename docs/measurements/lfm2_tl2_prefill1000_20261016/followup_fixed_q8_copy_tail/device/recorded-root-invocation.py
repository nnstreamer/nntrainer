#!/usr/bin/env python3
"""Sequential real-DSP correctness runs; these are not TPS measurements."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import threading
import time

D = Path(__file__).resolve().parent
A = Path('/tmp/nntr-lfm2-attn-raw-planes-20261017')
R = Path('/home/jungone-lee/workspace/nntrainer-lfm2-tl2-prefill1000')
E = R / 'docs/measurements/lfm2_tl2_prefill1000_20261016'
F = Path('/tmp/nntr-lfm2-attn-planes-kv-fixed-objects-20261017')
ADB = ['adb', '-s', 'R3CY70LV96T']
PHONE = '/data/local/tmp/nntr_lfm2_prefill1000_20261016'
BINARY = 'attention_raw_planes_device_check'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def shell(argv):
    return subprocess.check_output(ADB + ['shell', shlex.join(argv)], text=True, timeout=30)

out = D / 'device-fixture-fixed'
out.mkdir(exist_ok=False)
manifest = json.loads((A / 'device/manifest.json').read_text())
assert manifest[BINARY] == '351ca73b417e7b908790e9e041389aee662e9ba719a19e765e4d6a09aaadef2c'
assert all(sha(A / 'device' / name) == h for name, h in manifest.items())
fixed = json.loads((F / 'build/validation.json').read_text())
diag = fixed['diagnostic']
expected_dsp = {'control': fixed['control']['sha256'],
                'candidate': fixed['candidate']['sha256'],
                'diagnostic': diag['sha256']}
all_records = []
for arm, variant in [('control', 'lfm-attn-raw-planes-fixed-control'),
                     ('candidate', 'lfm-attn-raw-planes-fixed'),
                     ('diagnostic', 'lfm-attn-raw-planes-fixed-diagnostic')]:
    busy = subprocess.check_output(ADB + ['shell',
        'pidof nntr_causallm nntrainer_causallm attention_raw_planes_device_check || true'],
        text=True).strip()
    assert not busy, ('Device busy; no fixture started', busy)
    runtime = PHONE + '/' + variant
    m = json.loads((E / (variant + '-artifacts.json')).read_text())
    assert m['binaries']['libnntr_hvx_skel.so'] == expected_dsp[arm]
    expected = m['binaries']
    raw = shell(['sha256sum', *[runtime + '/' + name for name in expected]])
    actual = {Path(line.split()[1]).name: line.split()[0] for line in raw.splitlines()}
    assert actual == expected
    subprocess.run(ADB + ['push', '-Z', str(A / 'device' / BINARY), runtime + '/' + BINARY],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    assert shell(['sha256sum', runtime + '/' + BINARY]).split()[0] == manifest[BINARY]
    shell(['chmod', '755', runtime + '/' + BINARY])
    remote_dump = runtime + '/raw-planes-fixture-output.bin'
    remote_memory = runtime + '/raw-planes-fixture-memory.json'
    collector = None
    if arm == 'diagnostic':
        stamp = shell(['date', '+%m-%d %H:%M:%S.000']).strip()
        live = (out / 'diagnostic.live.logcat.txt').open('w')
        collector = subprocess.Popen(ADB + ['logcat', '-v', 'threadtime', '-T', stamp],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        def consume():
            for line in collector.stdout:
                if '[LFM-RAW4]' in line:
                    live.write(line)
            live.flush()
        thread = threading.Thread(target=consume)
        thread.start()
    invocation = ['env', 'LD_LIBRARY_PATH=' + runtime, 'ADSP_LIBRARY_PATH=' + runtime,
        'NNTR_NUM_THREADS=8', 'NNTR_HTP_PAGE_CACHE=0', 'NNTR_MOE_CACHE_GROW=0',
        'timeout', '150', PHONE + '/fixture-support/memory_monitor', '--output', remote_memory,
        '--period-ms', '20', '--', runtime + '/' + BINARY, '--dump', remote_dump]
    command = ADB + ['shell', shlex.join(invocation)]
    started = time.monotonic()
    try:
        with (out / (arm + '.log')).open('wb') as log:
            proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=170)
        if collector is not None:
            time.sleep(2)
    finally:
        if collector is not None:
            collector.terminate()
            collector.wait(timeout=10)
            thread.join(timeout=10)
            assert not thread.is_alive()
            live.close()
            (out / 'diagnostic.collector.stderr').write_text(collector.stderr.read())
    record = {'arm': arm, 'variant': variant, 'command': command,
              'returncode': proc.returncode, 'elapsed_seconds': time.monotonic() - started,
              'runtime_sha256': actual, 'fixture_sha256': manifest[BINARY],
              'performance_measurement': False}
    (out / (arm + '.execution.json')).write_text(json.dumps(record, indent=2) + '\n')
    assert proc.returncode == 0, (arm, 'Fixture failed; raw log preserved')
    for remote, suffix in [(remote_dump, '.bin'), (remote_memory, '.memory.json')]:
        local = out / (arm + suffix)
        subprocess.run(ADB + ['pull', '-Z', remote, str(local)], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        assert shell(['sha256sum', remote]).split()[0] == sha(local)
    memory = json.loads((out / (arm + '.memory.json')).read_text())
    assert memory['child_waited'] and memory['child_exit_code'] == 0
    assert (out / (arm + '.bin')).stat().st_size == 231908352
    record.update(pid=memory['pid'], dump_sha256=sha(out / (arm + '.bin')),
                  memory_sha256=sha(out / (arm + '.memory.json')),
                  log_sha256=sha(out / (arm + '.log')))
    if arm == 'diagnostic':
        parse = ['python3', str(A / 'diagnostic/parse.py'), '--log',
            str(out / 'diagnostic.live.logcat.txt'), '--expected',
            str(A / 'diagnostic/device-trace-expected.json'), '--pid', str(memory['pid']),
            '--output', str(out / 'diagnostic.trace.json')]
        subprocess.run(parse, check=True)
        record['trace_parser_command'] = parse
    # Only our verified, copied output files are removed; retain runtime and logs.
    shell(['rm', remote_dump, remote_memory])
    record['verified_fixture_outputs_removed_from_phone'] = True
    all_records.append(record)
    (out / 'executions.json').write_text(json.dumps(all_records, indent=2) + '\n')
    print({'arm': arm, 'pid': record['pid'], 'dump_sha256': record['dump_sha256'],
           'bytes': 231908352, 'status': 'completed; cross-arm comparison pending'}, flush=True)
