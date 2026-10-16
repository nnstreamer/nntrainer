#!/usr/bin/env python3
"""Compile four affected DSP TUs from recorded original cc1 plans, no link."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', required=True, type=Path)
    p.add_argument('--fixed-build', required=True, type=Path)
    p.add_argument('--work', required=True, type=Path)
    a = p.parse_args()
    plans = json.loads((a.fixed_build / 'compilations.json').read_text())
    old_source = a.fixed_build / 'source'
    tracked = {}
    # Materialize just the original recorded translation-unit dependency closure.
    # Preserve edited candidate files. Generated headers come from the frozen build.
    for name, plan in plans.items():
        if not name.startswith('baseline'):
            continue
        for source in plan['dependency_sha256']:
            src = Path(source)
            if not src.is_relative_to(old_source):
                continue
            rel = src.relative_to(old_source)
            origin = a.repo / rel
            if not origin.is_file():
                origin = src
            tracked[str(rel)] = sha(origin)
            for side in ('baseline', 'candidate'):
                dst = a.work / side / rel
                if not dst.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(origin, dst)
    consumers = {}
    for header in ('hexkl_attn_q.h', 'hvx_quant_u8.h'):
        consumers[header] = [Path(v['source']).name for k, v in plans.items()
                             if k.startswith('baseline') and
                             any(Path(d).name == header for d in v['dependency_sha256'])]
    assert set(consumers['hexkl_attn_q.h']) == {
        'nntr_hvx_attn_q.c', 'hexkl_attn_q.c', 'hvx_attn_decode_q.c'}
    targets = {'hvx_quant_u8.c', *consumers['hexkl_attn_q.h']}
    records = []
    out = a.work / 'codegen'
    out.mkdir(exist_ok=True)
    for name, plan in plans.items():
        source = Path(plan['source'])
        if not source.is_absolute():
            source = old_source / 'test/htp' / source
        if not name.startswith('baseline') or source.name not in targets:
            continue
        for side in ('baseline', 'candidate'):
            root = a.work / side
            dst_dir = out / side
            dst_dir.mkdir(exist_ok=True)
            stem = source.stem
            args = [x.replace(str(old_source), str(root)) for x in plan['cc1']]
            # Keep original compile flags, apart from output/dependency locations.
            for option in ('-o', '-dependency-file', '-MT'):
                while option in args:
                    i = args.index(option)
                    del args[i:i+2]
            dep = dst_dir / (stem + '.d')
            for mode, action, suffix in [('object', '-emit-obj', '.o'), ('assembly', '-S', '.s')]:
                argv = [action if x == '-emit-obj' else x for x in args]
                target = dst_dir / (stem + suffix)
                argv += ['-o', str(target)]
                if mode == 'object':
                    argv += ['-dependency-file', str(dep), '-MT', str(target),
                             '-stack-usage-file', str(dst_dir / (stem + '.su'))]
                proc = subprocess.run(argv, cwd=root / 'test/htp', text=True,
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                log = dst_dir / (stem + '-' + mode + '.log')
                log.write_text(proc.stdout)
                rec = {'side': side, 'source': str(source.relative_to(old_source)),
                       'source_sha256': sha(root / source.relative_to(old_source)),
                       'mode': mode, 'command': argv, 'cwd': str(root / 'test/htp'),
                       'returncode': proc.returncode, 'log_sha256': sha(log)}
                if target.exists():
                    rec.update(output=str(target), output_sha256=sha(target))
                records.append(rec)
                (out / 'commands.json').write_text(json.dumps(records, indent=2) + '\n')
                if proc.returncode:
                    raise RuntimeError(f'{side} {source.name} {mode}: {log}')
                print(f'{side} {source.name} {mode}: PASS', flush=True)
    (out / 'dependency-screen.json').write_text(json.dumps({
        'source_dependency_hashes': tracked, 'recorded_baseline_tus': 63,
        'old_header_consumers': consumers, 'recompile_tus': sorted(targets),
        'reason': 'all IO layout producers/consumers, plus new helper implementation',
        'unchanged_header': 'hvx_quant_u8.h; new declaration isolated in hvx_quant_u8_group4.h',
        'full_dsp_build_or_link_performed': False,
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
