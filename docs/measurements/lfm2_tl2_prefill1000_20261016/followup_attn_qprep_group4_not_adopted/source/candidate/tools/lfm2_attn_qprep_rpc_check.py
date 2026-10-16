#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Extract actual quantized-attention RPC bodies and test staging capability.

Reserve/DMA/KV append/attention computation are explicit synchronous stand-ins;
this tests ownership propagation and failure routing, not DSP coherence/math.
"""
import argparse, hashlib, json, os, re, subprocess
from pathlib import Path

def function(s, name):
    m=re.search(r'^int '+re.escape(name)+r'\([^;{]*\{',s,re.M)
    if not m: raise RuntimeError('missing function '+name)
    end,depth=m.end(),1
    while depth:
        depth+=(s[end]=='{')-(s[end]=='}'); end+=1
    return s[m.start():end]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--source',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sanitize',action='store_true')
    p.add_argument('--mutations',action='store_true')
    a=p.parse_args(); a.source=a.source or a.repo; a.output.mkdir(parents=True,exist_ok=True)
    rpc=a.source/'test/htp/nntr_hvx_attn_q.c'; source=rpc.read_text()
    start=source.index('enum {\n  QSTAT_QPREP'); end=source.index('\n};',start)+3
    h=source[start:end]+'\n'+''.join(function(source,n)+'\n' for n in ['nntr_hvx_attn_q_prefill','nntr_hvx_attn_q_decode','nntr_hvx_attn_q_step'])
    fixture=Path(__file__).resolve().parents[1]/'test/htp/host/attn_qprep_rpc_check.c'
    under=a.output/'attn_qprep_rpc_under_test.h'; under.write_text(h)
    flags=['-std=gnu11','-O1','-g','-Wall','-Wextra','-Werror']
    if a.sanitize: flags+=['-fsanitize=address,undefined','-fno-omit-frame-pointer']
    includes=[]
    for x in [a.repo/'test/htp/host/stub',a.source/'nntrainer/tensor/htp_backend/hvx',a.source/'nntrainer/tensor/htp_backend/hmx',a.output]: includes+=['-I',str(x)]
    commands=[]; results=[]
    def run(name,good=True):
        cmd=[os.environ.get('CC','gcc'),*flags,*includes,str(fixture),'-lm','-o',str(a.output/name)]
        commands.append(cmd); subprocess.run(cmd,check=True)
        env=dict(os.environ); env['ASAN_OPTIONS']='detect_leaks=1:halt_on_error=1'; env['UBSAN_OPTIONS']='halt_on_error=1'
        r=subprocess.run([str(a.output/name)],capture_output=True,text=True,env=env)
        (a.output/(name+'.log')).write_text(r.stdout+r.stderr)
        if good: assert r.returncode==0, r.stdout+r.stderr
        else: assert r.returncode==-6 and 'Assertion' in r.stderr, r.stdout+r.stderr
        results.append({'name':name,'returncode':r.returncode,'expected_success':good})
        print(r.stdout.strip() if good else name+': rejected as expected',flush=True)
    run('rpc_check')
    if a.mutations:
        for name,old,new in [('always_cached','io.q_cached = cached_out != NULL;','io.q_cached = 1;'),('never_cached','io.q_cached = cached_out != NULL;','io.q_cached = 0;'),('missing_q_dma','hexkl_moe_dma_copy(base, q_f32, q_bytes, 0, 0);','/* mutant: omitted Q copy */')]:
            assert h.count(old)==1; under.write_text(h.replace(old,new)); run(name,False)
        under.write_text(h)
    (a.output/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
    (a.output/'validation.json').write_text(json.dumps({'scope':__doc__,'results':results,'source_sha256':hashlib.sha256(rpc.read_bytes()).hexdigest(),'fixture_sha256':hashlib.sha256(fixture.read_bytes()).hexdigest()},indent=2)+'\n')
if __name__=='__main__': main()
