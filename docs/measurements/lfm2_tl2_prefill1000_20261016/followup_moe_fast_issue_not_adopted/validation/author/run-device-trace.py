from pathlib import Path
import hashlib
import json
import subprocess
import sys

R = Path('/home/jungone-lee/workspace/nntrainer-lfm2-tl2-prefill1000')
E = R / 'docs/measurements/lfm2_tl2_prefill1000_20261016'
D = Path(__file__).parent
rows = int(sys.argv[1])
assert rows in (1024, 2048)
variant = 'lfm-moe-fast-issue-trace'
tag = 'continue-moe-issue-trace' + str(rows)
model = 'model' if rows == 1024 else 'model2048'
env = json.loads((D / 'root-normal-jobs.json').read_text())[0]['env']
command = [sys.executable, str(E / 'bench.py'), variant, tag,
           '--model', model, '--memory', '--profile']
for key, value in env.items():
    command += ['--env', key + '=' + str(value)]
subprocess.run(command, check=True, stdout=subprocess.PIPE, text=True)
result = json.loads((E / (tag + '.json')).read_text())
raw = subprocess.check_output(['adb', '-s', 'R3CY70LV96T', 'logcat', '-d',
                               '-v', 'threadtime', '-T', result['device_logcat_start']], text=True)
selected = [line for line in raw.splitlines() if '[LFM-MISSUE]' in line]
path = E / (tag + '.issue.logcat.txt')
path.write_text('\n'.join(selected) + '\n')
memory_path = E / (tag + '.memory.json')
memory = json.loads(memory_path.read_text())
peak = memory['peak_dedup_sample']
expected = {1024: 'b6c1b801f95caded5399ec685cce0f900a20c3f9df760482dd7b5c9a62a5b92e',
            2048: '2bc7afc3c7d8e4387c8d9c7378d6792a1ef0fb56c2635f73b95e8650ac7fef33'}
assert result['valid'] and result['returncode'] == 0 and result['ids_match_reference']
assert result['profile'] and result['instrumented'] and not result['performance_comparison_eligible']
assert not result['concurrent_processes'] and result['moe_cache_experts'] == 16
assert result['prefill_tokens'] == rows and result['decode_tokens'] == 64
assert result['ids_sha256'] == expected[rows]
assert hashlib.sha256((E / (tag + '.ids')).read_bytes()).hexdigest() == expected[rows]
for key, value in {'NNTR_HTP_PAGE_CACHE': '0', 'NNTR_MOE_CACHE_GROW': '0',
                   'NNTR_MOE_PRELOAD_EXPERTS': '128', **env}.items():
    assert result['environment'][key] == str(value)
assert memory['child_exit_code'] == 0 and memory['child_waited']
assert memory['requested_period_ms'] == 20 and peak['dedup_complete_for_readable_size_fds']
assert not any(peak[k] for k in ['fdinfo_read_errors', 'unknown_identity_fds',
                               'inode_size_conflicts', 'invalid_fields', 'counter_overflow'])
assert hashlib.sha256(memory_path.read_bytes()).hexdigest() == result['full_run_memory_report_sha256']
assert result['full_run_sampled_peak_kib'] == memory['peak_anon_plus_fdinfo_dedup_or_raw_fallback_bytes'] / 1024
subprocess.run([sys.executable, str(D / 'diagnostic/parse.py'), str(path), '--rows', str(rows),
                '--output', str(D / ('root-trace-summary' + str(rows) + '.json'))], check=True)
print(tag, 'IDs, fixed settings, memory and strict dispatch checks PASS', flush=True)
