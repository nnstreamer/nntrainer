#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path
import subprocess
import threading
import time

D = Path(__file__).resolve().parent
R = Path('/home/jungone-lee/workspace/nntrainer-lfm2-tl2-prefill1000')
E = R / 'docs/measurements/lfm2_tl2_prefill1000_20261016'
A = Path('/tmp/nntr-lfm2-attn-strided-qprep-20261017')
ADB = ['adb', '-s', 'R3CY70LV96T']
jobs = json.loads((D / 'root-normal-jobs.json').read_text())
variant = 'lfm-attn-qprep-group4-diagnostic'
manifest = json.loads((E / (variant + '-artifacts.json')).read_text())
assert manifest['binaries']['libnntr_hvx_skel.so'] == 'ca3c46323c7ccf7d5c263fd530e4aefbab00ba089be7cc22941fbeec3337c1ed'
all_records = []

for rows, model in [(2048, 'model2048'), (1024, 'model')]:
    job = next(j for j in jobs if j['model'] == model)
    tag = f'continue-attn-qprep-group4-diagnostic{rows}'
    assert not (E / (tag + '.json')).exists()
    timestamp = subprocess.check_output(ADB + ['shell', "date '+%m-%d %H:%M:%S.000'"], text=True).strip()
    live_path = D / (tag + '.live.logcat.txt')
    live = live_path.open('w')
    collector = subprocess.Popen(ADB + ['logcat', '-v', 'threadtime', '-T', timestamp],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def consume():
        for line in collector.stdout:
            if '[nntr-qprep-group4]' in line:
                live.write(line)
        live.flush()

    thread = threading.Thread(target=consume)
    thread.start()
    command = ['python3', str(E / 'bench.py'), variant, tag,
               '--model', model, '--memory', '--profile', '--env', 'NNTR_HTP_ATTN_TRACE=1']
    for key, value in job['env'].items():
        command += ['--env', key + '=' + str(value)]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (D / (tag + '.runner.log')).write_text(result.stdout)
        assert result.returncode == 0, 'Diagnostic model failed; raw logs retained'
        time.sleep(2)
    finally:
        collector.terminate()
        collector.wait(timeout=10)
        thread.join(timeout=10)
        assert not thread.is_alive()
        live.close()
        (D / (tag + '.collector.stderr')).write_text(collector.stderr.read())
    meta = json.loads((E / (tag + '.json')).read_text())
    memory = json.loads((E / (tag + '.memory.json')).read_text())
    assert meta['valid'] and meta['ids_match_reference'] and meta['decode_tokens'] == 64
    assert meta['returncode'] == 0 and memory['child_waited'] and memory['child_exit_code'] == 0
    parse_cmd = ['python3', str(A / 'diagnostic/parse_trace.py'), str(live_path),
                 '--pid', str(memory['pid']), '--model', str(rows)]
    parsed = json.loads(subprocess.check_output(parse_cmd, text=True))
    parsed.update(tag=tag, model_command=command, parser_command=parse_cmd,
                  live_log_sha256=hashlib.sha256(live_path.read_bytes()).hexdigest(),
                  parser_sha256=hashlib.sha256((A / 'diagnostic/parse_trace.py').read_bytes()).hexdigest(),
                  trace_enabled_binary_verified=True, normal_performance_eligible=False,
                  boundary='Fresh single supervised child; live capture begins before launch and ends after normal child completion.')
    (D / (tag + '.parsed.json')).write_text(json.dumps(parsed, indent=2) + '\n')
    all_records.append(parsed)
    (D / 'root-diagnostic-summary.json').write_text(json.dumps(all_records, indent=2) + '\n')
    print({k: parsed[k] for k in ['tag', 'unique_calls', 'transport_duplicates', 'logical_gather_bytes_skipped']}, flush=True)
