#!/usr/bin/env python3
"""Read existing model dispatch and stage artifacts; never execute a device."""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import sys
sys.dont_write_bytecode = True
from audit_model_records import read_run
from parse_stages import need, sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('evidence','results','fixed-build','trace-parser','output'):
        p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    spec=importlib.util.spec_from_file_location('frozen_raw4_parser',a.trace_parser)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    build=json.loads((a.fixed_build/'validation.json').read_text())
    summaries=json.loads((a.results/'root-model-diagnostic-fixed-summary.json').read_text())
    need(len(summaries)==2,'Two model dispatch runs')
    base_env=json.loads((a.results/'root-normal-fixed-jobs.json').read_text())[0]['env']
    dispatch=[]
    for i,s in enumerate(summaries):
        n=(1024,2048)[i];t=s['tag'];meta=json.loads((a.evidence/(t+'.json')).read_text())
        job={'tag':t,'variant':meta['variant'],'model':'model' if n==1024 else 'model2048','env':base_env}
        execution={'tag':t,'returncode':0,'runtime_manifest_sha256':s['manifest_sha256'],'runtime_sha256_before':s['runtime_sha256']}
        audit=read_run(a.evidence,job,execution,{'candidate':build['diagnostic']},True)
        audit['arm']='diagnostic'
        need(audit['memory']['pid']==s['pid'],'Model diagnostic supervised PID')
        raw=a.results/(t+'.live.logcat.txt');need(sha(raw)==s['live_log_sha256'],'Live PID log hash')
        # G4 br=128, bc=256, 8 KV heads. Causal kb_hi=ceil(q_end/256).
        # Each QK block has 16 row tiles x 4 column tiles = 64 readouts.
        blocks=8*sum(((qb+1)*128+255)//256 for qb in range(n//128))
        serial=blocks*64 if n==1024 else 8*(n//128)*64
        deferred=blocks*64-serial
        expected=[dict(m=n,on=1,raw_s=serial,raw_d=deferred,sdk=0,b=blocks,rc=0) for _ in range(10)]
        actual=mod.parse(raw.read_text(),expected,s['pid'])
        need(actual==s['parser_result'],'Independent expected geometry matches root raw result')
        dispatch.append({'tag':t,'pid':s['pid'],'actual':actual,'expected_per_layer':expected[0],
                         'expected_layers':10,'live_log_sha256':sha(raw),'run_audit':audit})
    (a.output/'model-dispatch-readback.json').write_text(json.dumps({'passed':True,'runs':dispatch,
        'parser_sha256':sha(a.trace_parser),'derivation':'8 KV heads * sum ceil((qb+1)*128/256) causal blocks; 16*4 QK tiles/block. 1024 is serial; 2048 serial first block/query block plus deferred remaining blocks. Ten real attention layers; no inferred warmups.'},indent=2)+'\n')

    jobs=json.loads((a.results/'root-stage-fixed-jobs.json').read_text())
    executions=json.loads((a.results/'root-stage-fixed-executions.json').read_text())
    need(len(jobs)==len(executions)==4,'Four stage runs')
    stages=[]
    for j,e in zip(jobs,executions):
        audit=read_run(a.evidence,j,e,build,True)
        parsed=json.loads((a.output/(j['tag']+'.parsed.json')).read_text())
        need(parsed['pid']==audit['memory']['pid'] and parsed['calls']==10 and parsed['ids_sha256']==audit['ids_sha256'],'Stage readback identity')
        for k,name in [('metadata','json'),('log','log'),('memory','memory.json'),('ids','ids')]:
            need(parsed['inputs'][k]['sha256']==audit['input_sha256'][name],'Parser/runtime inputs '+k)
        need(parsed['inputs']['logcat']['sha256']==sha(a.evidence/(j['tag']+'.logcat.txt')),'Stage raw hash')
        log=(a.evidence/(j['tag']+'.log')).read_text()
        patterns={
            'prefetch':r'^\[HTP-PROFILE\] expert prefetch: (\d+) experts read under the layer calls, exposed wait ([\d.]+) ms \(([\d.]+) ms/expert\), register rpc ([\d.]+) ms \(([\d.]+) ms/expert\)$',
            'progress':r'^\[HTP-PROFILE\] expert prefetch progress: (\d+) of (\d+) read when their layer asked \((\d+)%\), caller cpus (\{[^}]*\}), reader cpus (\{[^}]*\})$',
            'readers':r'^\[HTP-PROFILE\] expert prefetch readers: ([\d.]+) ms/expert on a reader, (\d+) experts$'}
        values={}
        for k,pattern in patterns.items():
            found=re.findall(pattern,log,re.M);need(len(found)==1,'Exactly one expert prefetch '+k);values[k]=found[0]
        experts=int(values['prefetch'][0]);need(experts==2304==int(values['progress'][1])==int(values['readers'][1]),'Prefetch expert counts')
        context={'experts':experts,'exposed_wait_ms':float(values['prefetch'][1]),
                 'register_rpc_ms':float(values['prefetch'][3]),'ready_experts':int(values['progress'][0]),
                 'ready_percent_logged':int(values['progress'][2]),'caller_cpus_logged':values['progress'][3],
                 'reader_cpus_logged':values['progress'][4],'reader_ms_per_expert':float(values['readers'][0])}
        stages.append({'tag':j['tag'],'run_audit':audit,'attention':parsed,'expert_prefetch_context':context})
    comparisons=[]
    for n in (1024,2048):
        rows={r['run_audit']['arm']:r for r in stages if r['run_audit']['tokens']==n}
        c=rows['control']['attention']['totals'];b=rows['candidate']['attention']['totals']
        comparisons.append({'tokens':n,'control_tag':rows['control']['tag'],'candidate_tag':rows['candidate']['tag'],
            'stage_delta_us':{k:b['stages_us'][k]-v for k,v in c['stages_us'].items()},
            'wrapper_delta_us':{k:b['wrapper_us'][k]-v for k,v in c['wrapper_us'].items()},
            'layer_profile_delta_ns':b['layer_profile_ns']-c['layer_profile_ns'],
            'expert_prefetch_context':{arm:row['expert_prefetch_context'] for arm,row in rows.items()}})
    result={'passed':True,'runs':stages,'comparisons':comparisons,
            'interpretation':[
                'The stage traces use normal controlled DSPs and profile1/NNTR_HTP_ATTN_TRACE=1. The separate dispatch traces use the diagnostic DSP. Neither set contributes normal TPS observations.',
                'us_qk covers accumulator clear, HMX issue and accumulator readout; immediate conversion is recorded in us_dequant. It is not pure HMX arithmetic time.',
                'us_softmax records submit plus exposed wait at softmax_finish_pquant. Shorter next-QK work can expose previously overlapped softmax time; it does not prove extra softmax arithmetic.',
                'Deferred raw dequant workers start only after softmax and pquant have completed, overlap PV, then join. There is no direct simultaneous softmax/dequant worker contention in this pipeline.',
                'The 1024 prefetch wait differs materially in this single pair. It is contextual evidence for whole-model variability, not a demonstrated cause or permission to adjust measured TPS.',
                'Do not sum worker or overlapping fields, treat QK delta as full-model saving, or extrapolate these four diagnostic observations to normal performance.'
            ]}
    (a.output/'stage-readback.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'model_dispatch_runs':len(dispatch),'stage_runs':len(stages),'passed':True,'comparisons':comparisons}))


if __name__=='__main__':
    main()
