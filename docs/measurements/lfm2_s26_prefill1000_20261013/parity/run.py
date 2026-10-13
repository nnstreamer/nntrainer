#!/usr/bin/env python3
"""Compare all raw logits/tokens from a matched diagnostic staging runtime."""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
sys.path.insert(0, str(OUT))
import measure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--count', type=int, required=True)
    parser.add_argument('--aligned', action='store_true')
    args = parser.parse_args()
    assert re.fullmatch(r'[a-zA-Z0-9_-]+', args.label)
    assert 0 <= args.count <= 160
    assert not args.aligned or args.count > 0
    destination = HERE / ('dumps-' + args.label)
    comparison = HERE / (args.label + '-raw-output-comparison.json')
    assert not destination.exists() and not comparison.exists()
    remote = '/data/local/tmp/lfm2_prefill1000_20261013/parity-' + args.label
    # A reused directory could leave extra or stale diagnostic files.
    measure.shell('mkdir ' + shlex.quote(remote))
    env = json.loads((OUT / 'v13-br256-r7-clean-x1.json').read_text())['env']
    env.update(NNTR_MOE_HOST_STAGE_EXPERTS=str(args.count),
               NNTR_MOE_HOST_STAGE_ALIGNED='1' if args.aligned else '0',
               NNTR_MOE_IO_PROFILE='0', NNTR_PARITY_DUMP=remote,
               NNTR_PARITY_LOGIT_STEPS='65', NNTR_PARITY_LOGITS_ONLY='1')
    row = measure.run(args.variant, 1024, 'clean', args.label, env)
    row.update(diagnostic_only=True, clean_timing_sample=False)
    measure.save(OUT / (args.label + '.json'), row)
    raw = (OUT / (args.label + '.log')).read_text()
    parse = lambda s: {k: int(v) for k, v in re.findall(r'(\w+)=(\d+)', s)}
    active = [parse(s) for s in re.findall(
        r'\[HTP-HOST-STAGING-ACTIVE\] ([^\n]+)', raw)]
    done = [parse(s) for s in re.findall(
        r'\[HTP-HOST-STAGING-DONE\] ([^\n]+)', raw)]
    if args.count:
        assert len(active) == len(done) == 1
        assert active[0]['count'] == done[0]['count'] == args.count
        assert done[0]['read'] == done[0]['consumed'] == args.count
        if args.aligned:
            assert active[0].get('aligned') == 1
            assert active[0]['payload_bytes'] <= active[0]['allocated_bytes'] <= 384 * 1024 * 1024
    else:
        assert not active and not done
    subprocess.run(measure.ADB + ['pull', remote, str(destination)],
                   capture_output=True, check=True)
    baseline = HERE / 'dumps-baseline-v1'
    expected = sorted(p.name for p in baseline.iterdir() if p.is_file())
    actual = sorted(p.name for p in destination.iterdir() if p.is_file())
    assert len(expected) == 132 and expected == actual
    different = [name for name in expected
                 if (baseline / name).read_bytes() !=
                 (destination / name).read_bytes()]
    result = dict(baseline=str(baseline), candidate=str(destination),
                  files=len(expected), different=different, exact=not different,
                  staging_active=active, staging_done=done,
                  scope='65 raw logits, 65 token IDs, prompt and manifest; '
                        'diagnostic timing is excluded from TPS results.')
    measure.save(comparison, result)
    print('RAW PARITY', result, flush=True)
    assert not different


if __name__ == '__main__':
    main()
