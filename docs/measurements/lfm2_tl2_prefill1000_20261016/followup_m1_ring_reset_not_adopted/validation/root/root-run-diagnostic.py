import hashlib
import json
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_EVEN
from pathlib import Path
import re
import subprocess
import threading
import time

D = Path(__file__).parent
R = Path('/home/jungone-lee/workspace/nntrainer-lfm2-tl2-prefill1000')
E = R / 'docs/measurements/lfm2_tl2_prefill1000_20261016'
F = Path('/tmp/nntr-lfm2-ring-fixed-objects-20261017')
ADB = ['adb', '-s', 'R3CY70LV96T']
jobs = json.loads((D / 'root-normal-jobs.json').read_text())

for rows, model in [(1024, 'model'), (2048, 'model2048')]:
    job = next(j for j in jobs if j['model'] == model)
    tag = f'continue-m1-ring-reset-diagnostic{rows}'
    assert not (E / (tag + '.json')).exists()
    timestamp = subprocess.check_output(
        ADB + ['shell', "date '+%m-%d %H:%M:%S.000'"], text=True).strip()
    live_path = D / (tag + '.live.logcat.txt')
    live = live_path.open('w')
    collector = subprocess.Popen(
        ADB + ['logcat', '-v', 'threadtime', '-T', timestamp],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def consume():
        for line in collector.stdout:
            if 'nntr-ring-reset ' in line:
                live.write(line)
        live.flush()

    thread = threading.Thread(target=consume)
    thread.start()
    command = ['python3', str(E / 'bench.py'),
               'lfm-m1-ring-reset-diagnostic', tag, '--model', model,
               '--memory', '--profile']
    for key, value in job['env'].items():
        command += ['--env', key + '=' + str(value)]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        (D / (tag + '.runner.log')).write_text(result.stdout)
        if result.returncode:
            raise RuntimeError('Diagnostic model failed; raw logs retained')
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
    output = (E / (tag + '.log')).read_text()
    assert meta['valid'] and meta['ids_match_reference']
    assert meta['decode_tokens'] == 64 and meta['returncode'] == 0
    assert memory['child_waited'] and memory['child_exit_code'] == 0
    assert '[HTP] graph: forward calls=64 tokens=64 calls/token=1.00' in output
    assert re.search(r'\[HTP\] dspq: close calls=64 served=64 bad=0 .*stop_err=0x0', output)
    rates = re.findall(r'\[HTP\] token driver: moe calls/token=(\d+\.\d{2}) ', output)
    assert len(rates) == 1
    rate = Decimal(rates[0])
    low = int(((rate - Decimal('.005')) * 64).to_integral_value(rounding=ROUND_FLOOR))
    high = int(((rate + Decimal('.005')) * 64).to_integral_value(rounding=ROUND_CEILING))
    totals = [n for n in range(low, high + 1)
              if (Decimal(n) / 64).quantize(Decimal('.01'), rounding=ROUND_HALF_EVEN) == rate]
    assert len(totals) == 1, totals
    parse_cmd = ['python3', str(F / 'diagnostic/parse.py'), str(live_path),
                 '--pid', str(memory['pid']), '--expected-calls', str(totals[0])]
    parsed = json.loads(subprocess.check_output(parse_cmd, text=True))
    parsed.update(model=model, tag=tag, token_driver_calls_per_token=rates[0],
                  independently_derived_calls=totals[0],
                  count_scope='Single native LFM graph prompt, 64 successful tokens, no M1 load warmup or extra model call; same-run token-driver count only.',
                  model_command=command, parser_command=parse_cmd,
                  live_log_sha256=hashlib.sha256(live_path.read_bytes()).hexdigest())
    (D / (tag + '.parsed.json')).write_text(json.dumps(parsed, indent=2) + '\n')
    print({k: parsed[k] for k in ['tag', 'observed_unique_attempts', 'hits',
                                 'fallbacks', 'independently_derived_calls']}, flush=True)
