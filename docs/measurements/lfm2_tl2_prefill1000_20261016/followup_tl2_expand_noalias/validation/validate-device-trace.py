from pathlib import Path
import json
import re
import sys

evidence = Path(sys.argv[1])
output = Path(sys.argv[2])
pattern = re.compile(r'\[nntr-tl2-disjoint\] M=(\d+) K=(\d+) I=(\d+) N=(\d+) E=(\d+) count=(\d+)$')
result = {}
for context in (1024, 2048):
    tag = f'continue-expand-noalias-trace{context}'
    lines = [line for line in (evidence / (tag + '.logcat.txt')).read_text().splitlines() if '[nntr-tl2-disjoint]' in line]
    assert len(lines) == 39, (tag, len(lines))
    rows = []
    for index, line in enumerate(lines):
        match = pattern.search(line)
        assert match, line
        row = dict(zip(('M', 'K', 'I', 'N', 'active_experts', 'count'), map(int, match.groups())))
        assert (row['K'], row['I'], row['N']) == (2048, 1536, 2048)
        # gemma4_htp_weights.cpp shared loader uses max(512, INIT_SEQ_LEN).
        assert row['M'] == context
        assert 1 <= row['active_experts'] <= 64
        assert row['count'] == row['active_experts'] * 784
        if index == 0:
            assert row['active_experts'] == 64
        rows.append(row)
    run = json.loads((evidence / (tag + '.json')).read_text())
    memory = json.loads((evidence / (tag + '.memory.json')).read_text())
    peak = memory['peak_dedup_sample']
    assert run['valid'] and run['profile'] and run['instrumented']
    assert not run['performance_comparison_eligible']
    assert run['ids_match_reference'] and not run['concurrent_processes']
    assert run['moe_cache_experts'] == 16
    for key, val in {'NNTR_HTP_PAGE_CACHE': '0', 'NNTR_MOE_CACHE_GROW': '0', 'NNTR_MOE_PRELOAD_EXPERTS': '128', 'NNTR_LFM2_ROUTER_INPUT_HANDOFF': '1'}.items():
        assert run['environment'][key] == val
    assert memory['child_exit_code'] == 0 and peak['dedup_complete_for_readable_size_fds']
    assert not any(peak[k] for k in ('fdinfo_read_errors', 'unknown_identity_fds', 'inode_size_conflicts'))
    result[str(context)] = {
        'tag': tag, 'warmup_calls': 1, 'main_calls': 38,
        'main_active_experts': sum(row['active_experts'] for row in rows[1:]),
        'actual_disjoint_main_calls': sum(row['count'] for row in rows[1:]),
        'all_counts_equal_784_per_active_expert': True,
        'ids_match_reference': True, 'trace_excluded_from_perf': True,
        'rows': rows,
    }
    print(context, 'real MoE38/38 used disjoint entry; main calls=', result[str(context)]['actual_disjoint_main_calls'])
output.write_text(json.dumps(result, indent=2) + '\n')
