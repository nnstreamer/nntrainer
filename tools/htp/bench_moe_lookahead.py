#!/usr/bin/env python3
# @brief Paired S26 decode-prefetch benchmark using a predeployed runtime/model.
# SPDX-License-Identifier: Apache-2.0
"""Paired S26 decode-prefetch benchmark using a predeployed runtime/model.

Teacher-forced IDs and a baseline env JSON are explicit inputs. Snapshot I/O
is disabled. Profile runs log read readiness at the real expert-use deadline;
clean runs omit per-batch logs. All modes must reproduce the PPL step trace.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import statistics
import time


def fields(line):
    return dict(re.findall(r'(\w+)=([^\s]+)', line))


def parse_log(raw):
    generation = re.search(r'^generation: (\d+) tokens, (\d+) ms, ([\d.]+) TPS', raw, re.M)
    if not generation:
        raise ValueError('Missing generation summary')
    steps = re.findall(r'\[PPL\] decode step=.*', raw, re.M)
    if len(steps) != int(generation[1]):
        raise ValueError('Incomplete fixed-token decode')
    rows = [fields(line) for line in raw.splitlines()
            if line.startswith('[MOE-LOOKAHEAD]')]
    totals = {k: sum(int(row[k]) for row in rows) for k in
              ['calls', 'demand', 'hits', 'predicted', 'correct', 'reads',
               'useful', 'predict_us', 'begin_us', 'collect_us', 'window_us', 'batches']}
    deadlines = []
    last_batch = None
    for line in raw.splitlines():
        if '[MOE-PREFETCH-BATCH]' in line:
            last_batch = {k:int(v) for k,v in fields(line).items()}
        elif '[MOE-LOOKAHEAD-TAKE]' in line:
            take = fields(line)
            if last_batch is None or last_batch['n'] != int(take['n']):
                raise ValueError('Unpaired prefetch deadline')
            deadlines.append(dict(last_batch, **{k:int(v) if k!='kind' else v
                                                  for k,v in take.items() if k!='n'}))
            last_batch = None
    totals['remaining_demand_misses'] = totals['demand'] - totals['hits']
    totals['post_wait_hit_pct'] = 100 * totals['hits'] / totals['demand']
    totals['total_expert_reads'] = totals['reads'] + totals['remaining_demand_misses']
    if deadlines:
        totals['ready_reads'] = sum(d['ready'] for d in deadlines)
        totals['ready_pct'] = 100 * totals['ready_reads'] / totals['reads']
        totals['useful_ready'] = sum((d['ready_mask'] & d['useful_mask']).bit_count() for d in deadlines)
        totals['useful_late'] = totals['useful'] - totals['useful_ready']
        totals['ready_demand_hit_pct'] = 100 * (totals['hits'] - totals['useful_late']) / totals['demand']
        totals['exposed_read_wait_us'] = sum(d['wait_us'] for d in deadlines)
        totals['registration_us'] = sum(d['register_us'] for d in deadlines)
    profile = re.search(r'expert cache misses: (\d+), file read ([\d.]+) ms .*swap rpc ([\d.]+) ms',raw)
    return {'tokens':int(generation[1]), 'ms':int(generation[2]), 'tps':float(generation[3]),
            'ppl_trace_sha256':hashlib.sha256('\n'.join(steps).encode()).hexdigest(),
            'totals':totals, 'layers':rows, 'deadlines':deadlines,
            'demand_io_profile':{'misses':int(profile[1]), 'read_ms':float(profile[2]),
                                 'register_ms':float(profile[3])} if profile else None}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--serial',required=True)
    p.add_argument('--runtime',required=True)
    p.add_argument('--model',required=True)
    p.add_argument('--continuation',required=True)
    p.add_argument('--environment',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--modes',nargs='+',default=['off','block','block_ffn','previous'])
    p.add_argument('--repeats',type=int,default=1)
    p.add_argument('--profile',action='store_true')
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    adb=['adb','-s',a.serial,'shell']
    env=json.loads(a.environment.read_text())['env']
    for key in ['NNTR_MOE_LOOKAHEAD_DUMP','NNTR_MOE_TRACE','NNTR_MOE_PREFETCH_BATCH_TRACE']:
        env.pop(key,None)
    env.update(NNTR_PPL_DECODE=a.continuation, NNTR_HTP_PROFILE=str(int(a.profile)),NNTR_OP_TIME='0')
    if a.profile:
        env['NNTR_MOE_PREFETCH_BATCH_TRACE']='1'
    hashes=subprocess.check_output(adb+['cd '+shlex.quote(a.runtime)+' && sha256sum nntrainer_causallm libcausallm_core.so libnntrainer.so libnntr_hvx_skel.so'],text=True)
    report={'serial':a.serial,'runtime':a.runtime,'model':a.model,'continuation':a.continuation,
            'binary_sha256':hashes,'profile':a.profile,'runs':[]}
    parity=None
    for repeat in range(a.repeats):
        modes=a.modes if repeat%2==0 else a.modes[::-1]
        for mode in modes:
            busy=subprocess.run(adb+['pidof nntrainer_causallm'],capture_output=True,text=True,timeout=10)
            if busy.returncode!=1 or busy.stdout.strip():
                raise RuntimeError('Device busy: '+busy.stdout+busy.stderr)
            current=dict(env,NNTR_MOE_DECODE_PREFETCH=mode)
            cmd='cd '+shlex.quote(a.runtime)+' && env '+' '.join(k+'='+shlex.quote(v) for k,v in current.items())+' ./nntrainer_causallm '+shlex.quote(a.model)
            filename=f'{repeat+1}-{mode}.log'
            print('RUN '+filename,flush=True)
            battery=subprocess.check_output(adb+['dumpsys battery'],text=True,timeout=10)
            temperature=re.search(r'temperature:\s*(\d+)',battery)
            concurrent=False
            with (a.output/filename).open('w') as f:
                process=subprocess.Popen(adb+[cmd],stdout=f,stderr=subprocess.STDOUT)
                start=time.monotonic()
                while process.poll() is None:
                    check=subprocess.run(adb+['pidof nntrainer_causallm'],capture_output=True,text=True,timeout=10)
                    if check.returncode not in (0,1):
                        raise RuntimeError('Process monitoring failed')
                    concurrent |= len(check.stdout.split()) > 1
                    if time.monotonic()-start > 240:
                        raise RuntimeError('Inference exceeded 240 seconds')
                    time.sleep(.5)
                if process.returncode:
                    raise RuntimeError('Inference failed: '+filename)
            row=parse_log((a.output/filename).read_text())
            row.update(mode=mode,repeat=repeat+1,log=filename,command=cmd,env=current)
            row.update(concurrent_inference=concurrent,
                       battery_c=int(temperature[1])/10 if temperature else None)
            if parity is None:parity=row['ppl_trace_sha256']
            row['same_output']=row['ppl_trace_sha256']==parity
            report['runs'].append(row)
            (a.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
            if not row['same_output']:raise RuntimeError('PPL trace changed: '+filename)
            if concurrent:raise RuntimeError('Concurrent inference invalidates '+filename)
            print(filename+': '+json.dumps({k:row[k] for k in ['tps','same_output','totals']}),flush=True)
    report['median_tps']={m:statistics.median(r['tps'] for r in report['runs'] if r['mode']==m) for m in a.modes}
    (a.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
