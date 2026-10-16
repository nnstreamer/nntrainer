#!/usr/bin/env python3
"""Parser sensitivity checks using already captured records, no device runs."""
import argparse
import json
from pathlib import Path
import re
import sys
sys.dont_write_bytecode = True
from parse_stages import parse, need, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    cases = []
    for tokens in (1024, 2048):
        stem = a.evidence / f'continue-attn-raw-planes-fixed-stage{tokens}-control'
        cat = Path(str(stem) + '.logcat.txt').read_text()
        out = Path(str(stem) + '.log').read_text()
        pid = json.loads(Path(str(stem) + '.memory.json').read_text())['pid']
        parse(cat, out, pid, tokens); cases.append({'tokens':tokens, 'check':'original', 'passed':True})
        lines = cat.splitlines(); selected = [i for i, x in enumerate(lines) if 'HTP attn trace q:' in x or 'HTP attn stages q:' in x]
        payloads = [re.sub(r'^.*?I nntrainer: ', '', lines[i]) for i in selected]
        parse(cat, out + '\n' + '\n'.join(payloads), pid, tokens)
        cases.append({'tokens':tokens, 'check':'complete_stdout_mirror', 'passed':True})
        def replace_first(text, pattern, replacement):
            changed, n = re.subn(pattern, replacement, text, count=1)
            need(n == 1 and changed != text, 'Mutation failed to change its intended field')
            return changed
        negative = {
            'missing_wrapper': ('\n'.join(x for i,x in enumerate(lines) if i != selected[0]), out, pid),
            'missing_final_stage': ('\n'.join(x for i,x in enumerate(lines) if i != selected[-1]), out, pid),
            'extra_pair': (cat + '\n' + lines[selected[0]] + '\n' + lines[selected[1]], out, pid),
            'duplicate_single_record': (cat + '\n' + lines[selected[0]], out, pid),
            'wrong_expected_pid': (cat, out, pid+1),
            'mixed_pid': (cat.replace(f' {pid} {pid} I nntrainer: HTP attn trace q:', f' {pid+1} {pid} I nntrainer: HTP attn trace q:', 1), out, pid),
            'wrong_tid': (cat.replace(f' {pid} {pid} I nntrainer: HTP attn trace q:', f' {pid} {pid+1} I nntrainer: HTP attn trace q:', 1), out, pid),
            'error': (cat.replace('err=0x0', 'err=0x1', 1), out, pid),
            'wrong_length': (cat.replace(f'n_q={tokens}', 'n_q=1', 1), out, pid),
            'wrong_cache': (cat.replace(f'cache=0..{tokens}', f'cache=1..{tokens}', 1), out, pid),
            'wrong_append': (cat.replace(f'append={tokens}', 'append=0', 1), out, pid),
            'wrong_block_count': (replace_first(cat, r'blocks=\d+', 'blocks=0'), out, pid),
            'counter_overflow': (replace_first(cat, r'qprep=\d+', 'qprep=4294967296'), out, pid),
            'negative_counter': (replace_first(cat, r'qprep=\d+', 'qprep=-1'), out, pid),
            'time_identity': (replace_first(cat, r'dsp_us=\d+', 'dsp_us=0'), out, pid),
            'inner_exceeds_wrapper': (replace_first(cat, r'total=\d+', 'total=4294967295'), out, pid),
            'zero_stage': (replace_first(cat, r'qk=\d+', 'qk=0'), out, pid),
            'truncated': (cat.replace(lines[selected[-1]], lines[selected[-1]][:-5]), out, pid),
            'trailing_unrecognized_field': (cat.replace(lines[selected[-1]], lines[selected[-1]] + ' other=1'), out, pid),
            'wrong_layer': (cat, out.replace(f'LAYERPROF\t{tokens}\tlayer2_attention\t', f'LAYERPROF\t{tokens}\tlayer3_attention\t', 1), pid),
            'missing_layer': (cat, re.sub(rf'^LAYERPROF\t{tokens}\tlayer2_attention\t\d+\n', '', out, count=1, flags=re.M), pid),
            'duplicate_layer': (cat, out + '\n' + re.search(rf'^LAYERPROF\t{tokens}\tlayer2_attention\t\d+$', out, re.M).group(), pid),
            'layer_time_too_small': (cat, re.sub(rf'(LAYERPROF\t{tokens}\tlayer2_attention\t)\d+', r'\g<1>1', out, count=1), pid),
            'partial_stdout_mirror': (cat, out + '\n' + payloads[0], pid),
            'mismatched_stdout_mirror': (cat, out + '\n' + '\n'.join(payloads).replace('err=0x0','err=0x2',1), pid),
            'mixed_other_attention_path': (cat.replace('HTP attn trace q:', 'HTP attn trace q2:', 1), out, pid),
            'reported_fallback_error': (cat + '\nHTP quantized attention step failed: 0x1; CPU fallback', out, pid),
        }
        for name, (changed_cat, changed_out, changed_pid) in negative.items():
            try:
                parse(changed_cat, changed_out, changed_pid, tokens)
            except ValueError as e:
                cases.append({'tokens':tokens, 'check':name, 'rejected':True, 'reason':str(e)})
            else:
                raise AssertionError('Accepted malformed fixture: ' + name)
    result = {'passed':True, 'valid_cases':4, 'rejected_cases':sum('rejected' in x for x in cases),
              'parser_sha256':sha(Path(__file__).with_name('parse_stages.py')), 'cases':cases,
              'scope':'Strict parser sensitivity using existing logs only; no new model/DSP execution.'}
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'}))


if __name__ == '__main__':
    main()
