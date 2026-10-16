from pathlib import Path
import json
import subprocess
import sys

root = Path('/home/jungone-lee/workspace/nntrainer-lfm2-tl2-prefill1000')
evidence = root / 'docs/measurements/lfm2_tl2_prefill1000_20261016'
out = Path(__file__).parent
diagnostic = out.parent / 'diagnostic-v2'
env = json.loads(Path('/tmp/nntr-lfm2-tl2-expand-noalias-20261016/root-normal-jobs.json').read_text())[0]['env']
for n in (1, 2):
    tag = 'continue-affine-fc-trace-v2-r' + str(n)
    command = [sys.executable, str(evidence / 'bench.py'), 'affine-fc-trace-v2', tag, '--model', 'model', '--memory', '--profile']
    for key, value in env.items():
        command += ['--env', key + '=' + str(value)]
    subprocess.run(command, check=True, stdout=subprocess.PIPE, text=True)
    run = json.loads((evidence / (tag + '.json')).read_text())
    # The normal harness does not retain this new tag. Read from the same
    # timestamp immediately after the run; preserve only our diagnostic lines.
    logcat = subprocess.check_output(['adb', '-s', 'R3CY70LV96T', 'logcat', '-d', '-v', 'threadtime', '-T', run['device_logcat_start']], text=True)
    selected = [line for line in logcat.splitlines() if 'LFM-AFC2' in line or 'LFM-AFFINE-FC' in line]
    path = evidence / (tag + '.affine.logcat.txt')
    path.write_text('\n'.join(selected) + '\n')
    subprocess.run([sys.executable, str(diagnostic / 'parse_affine_fc.py'), str(path), '--output', str(out / (tag + '.phases.json'))], check=True)
    memory = json.loads((evidence / (tag + '.memory.json')).read_text())
    peak = memory['peak_dedup_sample']
    assert run['valid'] and run['profile'] and run['instrumented'] and not run['performance_comparison_eligible']
    assert run['ids_match_reference'] and not run['concurrent_processes'] and run['moe_cache_experts'] == 16
    for k, v in {'NNTR_HTP_PAGE_CACHE': '0', 'NNTR_MOE_CACHE_GROW': '0', 'NNTR_MOE_PRELOAD_EXPERTS': '128'}.items():
        assert run['environment'][k] == v
    assert memory['child_exit_code'] == 0 and peak['dedup_complete_for_readable_size_fds']
    assert not any(peak[k] for k in ('fdinfo_read_errors', 'unknown_identity_fds', 'inode_size_conflicts'))
    print(tag, 'all65 IDs, C16/directIO/complete peak PASS; diagnostic excluded from normal TPS', flush=True)
