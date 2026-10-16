#!/usr/bin/env python3
"""Read-only archive validation. No compile, device, SDK or model access."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import statistics
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
REL = Path('Applications/CausalLM/models/lfm2_moe')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads((ROOT/path).read_text())


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT/relative)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def patch(before, text):
    """Replay ordinary unified text hunks in memory; never trust header paths."""
    current = dict(before)
    lines = text.splitlines(keepends=True)
    i = 0
    while i < len(lines):
        if not lines[i].startswith('--- '):
            i += 1
            continue
        require(i+1 < len(lines) and lines[i+1].startswith('+++ '), 'bad patch header')
        name = Path(lines[i+1][4:].split('\t')[0].strip()).name
        require(name in current, 'unexpected patched file '+name)
        original = current[name].splitlines(keepends=True)
        output, consumed = [], 0
        i += 2
        while i < len(lines) and not lines[i].startswith(('diff --git ', '--- ')):
            if not lines[i].startswith('@@ '):
                i += 1
                continue
            m = re.match(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', lines[i])
            require(m is not None, 'bad hunk')
            start = int(m[1])-1 if int(m[1]) else 0
            oldn = int(m[2] or 1)
            newn = int(m[4] or 1)
            require(consumed <= start <= len(original), 'overlapping/out-of-range hunk')
            output.extend(original[consumed:start]); consumed = start
            oldcount = newcount = 0
            i += 1
            while i < len(lines) and lines[i][:1] in (' ', '+', '-') and not lines[i].startswith('--- '):
                kind, body = lines[i][0], lines[i][1:]
                if kind in (' ', '-'):
                    require(consumed < len(original) and original[consumed] == body, 'patch context differs')
                    consumed += 1; oldcount += 1
                if kind in (' ', '+'):
                    output.append(body); newcount += 1
                i += 1
            require((oldcount,newcount)==(oldn,newn), 'hunk extent differs')
        output.extend(original[consumed:])
        current[name] = ''.join(output)
    return current


def check_run(folder, tag, diagnostic=False):
    base = ROOT/folder
    data = json.loads((base/(tag+'.json')).read_text())
    ids = (base/(tag+'.ids')).read_bytes()
    require(len(ids.splitlines()) == 65 and all(re.fullmatch(rb'\d+', x) for x in ids.splitlines()), tag+' IDs')
    require(hashlib.sha256(ids).hexdigest()==data['ids_sha256'] and data['ids_match_reference'], tag+' IDs hash')
    require(data['returncode']==0 and data['valid'] and not data['concurrent_processes'], tag+' run validity')
    require(data['decode_tokens']==64 and data['prefill_tokens'] in (1024,2048), tag+' token count')
    require(data['moe_cache_experts']==16, tag+' C16')
    env=data['environment']
    for key,value in {'NNTR_HTP_PAGE_CACHE':'0','NNTR_LOAD_PREFETCH':'0','NNTR_MOE_CACHE_GROW':'0',
                      'NNTR_MOE_PRELOAD_EXPERTS':'128','NNTR_NUM_THREADS':'8'}.items():
        require(env.get(key)==value, tag+' fixed '+key)
    require('NNTR_MOE_PREFETCH_READERS' not in env and 'NNTR_MOE_PREFETCH' not in env, tag+' reader override')
    require(data['profile']==diagnostic and data['instrumented']==diagnostic and
            data['performance_comparison_eligible'] != diagnostic, tag+' performance cohort')
    mempath=base/(tag+'.memory.json')
    require(sha(mempath)==data['full_run_memory_report_sha256'], tag+' memory SHA')
    mem=json.loads(mempath.read_text())
    require(mem['child_waited'] and mem['child_exit_code']==0 and mem['child_signal'] is None, tag+' child completion')
    require(mem['sample_count']==len(mem['samples'])==data['full_run_memory_sample_count'], tag+' memory samples')
    require(mem['requested_period_ms']==20 and mem['sample_count']>0, tag+' sample period')
    peak=mem['peak_anon_plus_fdinfo_dedup_or_raw_fallback_bytes']
    require(peak/1024==data['full_run_sampled_peak_kib'], tag+' peak')
    require(peak==max(x['anon_plus_fdinfo_dedup_or_raw_fallback_bytes'] for x in mem['samples'] if x['composite_available']), tag+' max samples')
    log=(base/(tag+'.log')).read_text()
    for label,field in [('prefill','prefill'),('generation','decode')]:
        matches=re.findall(r'^'+label+r': (\d+) tokens, ([\d.]+) ms, ([\d.]+) TPS$',log,re.M)
        require(len(matches)==1, tag+' performance record')
        tokens,ms,tps=matches[0]
        require((int(tokens),float(ms),float(tps))==(data[field+'_tokens'],data[field+'_ms'],data[field+'_tps']),tag+' raw rate')
    return data


def validate():
    for item in read('provenance.json'):
        p=ROOT/item['archived']
        require(p.is_file() and sha(p)==item['sha256'] and p.stat().st_size==item['bytes'],'provenance '+str(p))
    for p in ROOT.rglob('*'):
        require(not p.is_symlink(),'symlink '+str(p))
        if p.is_file():
            content=p.read_bytes()
            require(not content.startswith((b'\x7fELF',b'MZ',b'!<arch>')) and b'\0' not in content,'binary '+str(p))
            require(p.suffix not in ('.o','.so','.a','.bin','.pyc'),'binary suffix '+str(p))

    before={name:(ROOT/'author/before'/REL/name).read_text() for name in ['lfm2_moe_layer.cpp','lfm2_moe_layer.h']}
    normal={name:(ROOT/'author/candidate'/REL/name).read_text() for name in before}
    require(patch(before,(ROOT/'author/production.patch').read_text())==normal,'author production replay')
    for mode in ['0','1']:
        artifact=read(f'device/lfm-prefill-recency-{mode}-artifacts.json')
        sourcepatch=ROOT/f'device/lfm-prefill-recency-{mode}.source.patch'
        require(sha(sourcepatch)==artifact['source_patch_sha256'],'normal deployed patch SHA')
        require(patch(before,sourcepatch.read_text())==normal,'normal deployed replay')
    actualdiag={name:(ROOT/'device/diagnostic'/name).read_text() for name in before}
    artifact=read('device/lfm-prefill-recency-diagnostic-artifacts.json')
    sourcepatch=ROOT/'device/lfm-prefill-recency-diagnostic.source.patch'
    require(sha(sourcepatch)==artifact['source_patch_sha256'],'diagnostic deployed patch SHA')
    require(patch(before,sourcepatch.read_text())==actualdiag,'diagnostic deployed replay from base')
    require(patch(normal,(ROOT/'device/root-diagnostic-integrated.patch').read_text())==actualdiag,'diagnostic replay from normal')
    require(sha(ROOT/'device/diagnostic/lfm2_moe_layer.cpp')==read('device/root-diagnostic-build.json')['source_sha256'],'diagnostic source SHA')
    for name in before:
        require((ROOT/'device/accepted-control'/name).read_text()==before[name],'base snapshot')
        require((ROOT/'device/normal'/name).read_text()==normal[name],'normal snapshot')
    lru=ROOT/'author/before'/REL/'expert_lru.h'
    require(lru.read_bytes()==(ROOT/'author/candidate'/REL/'expert_lru.h').read_bytes(),'generic LRU changed')

    ref=read('device/root-normal-summary.json')
    require(len(ref['runs'])==12 and ref['all_checks_pass'],'normal count')
    rows=[]
    for entry in ref['runs']:
        value=check_run('raw/normal',entry['tag'])
        for k in ['variant','prefill_tokens','prefill_tps','decode_tps','full_run_sampled_peak_kib','ids_sha256']:
            require(value[k]==entry[k],entry['tag']+' summary mismatch '+k)
        value['repetition']=entry['repetition'];rows.append(value)
    thermal=read('device/root-thermal-observations.json')
    require([x['tag'] for x in thermal]==[x['tag'] for x in rows], 'thermal normal cohort')
    cached_skin=[];hal_skin=[]
    for item,data in zip(thermal,rows):
        for phase in ['before','after']:
            text=data['thermal_'+phase]
            statuses=re.findall(r'Thermal Status: (\d+)',text)
            readings=[list(x) for x in re.findall(r'Temperature\{mValue=([^,}]+), mType=([^,}]+), mName=([^,}]+), mStatus=([^,}]+)\}',text)]
            require(item[phase]==dict(status=statuses,readings=readings), 'thermal raw summary differs')
            require(statuses==['0'],'unexpected recorded thermal status')
            require('Cached temperatures:' in text and 'Current temperatures from HAL:' in text,'thermal source sections')
            cached=text.split('Cached temperatures:',1)[1].split('HAL Ready:',1)[0]
            hal=text.split('Current temperatures from HAL:',1)[1].split('Current cooling devices from HAL:',1)[0]
            def skin(section):
                values=re.findall(r'Temperature\{mValue=([^,}]+), mType=3, mName=SKIN, mStatus=0\}',section)
                require(len(values)==1,'ambiguous thermal section extent')
                return float(values[0])
            cached_skin.append(skin(cached));hal_skin.append(skin(hal))
    require(set(cached_skin)=={37.9} and min(hal_skin)==30.8 and max(hal_skin)==39.3,'thermal recorded ranges')
    expected_order=['0','1','1','0','0','1']
    summaries=[]
    for tokens in [1024,2048]:
        cohort=[r for r in rows if r['prefill_tokens']==tokens]
        require([r['variant'][-1] for r in cohort]==expected_order,'balanced run order')
        require(len({r['ids_sha256'] for r in cohort})==1,'normal IDS parity')
        require(len({r['model_config_sha256'] for r in cohort})==1,'model config parity')
        summary=dict(tokens=tokens)
        for metric in ['prefill_tps','decode_tps']:
            arms=[[r[metric] for r in sorted(cohort,key=lambda x:x['repetition']) if r['variant'].endswith('-'+arm)] for arm in ['0','1']]
            c,b=arms
            summary[metric]=dict(control=c,candidate=b,median_control=statistics.median(c),median_candidate=statistics.median(b),
                                 median_pct=100*(statistics.median(b)/statistics.median(c)-1),pairs_faster=sum(y>x for x,y in zip(c,b)),
                                 paired_deltas_pct=[100*(y/x-1) for x,y in zip(c,b)])
        for arm,key in [('0','control'),('1','candidate')]:
            summary[key+'_max_peak_kib']=max(r['full_run_sampled_peak_kib'] for r in cohort if r['variant'].endswith('-'+arm))
        require(summary==next(x for x in ref['summary'] if x['tokens']==tokens),'recomputed normal summary differs')
        summaries.append(summary)

    parse=module('recency_trace','author/diagnostic/parse.py')
    diagnostics=[]
    for tokens,kept in [(1024,2405),(2048,2416)]:
        tag=f'continue-prefill-recency-diagnostic{tokens}'
        data=check_run('raw/diagnostic',tag,True)
        require(data['ids_sha256']==next(x['ids_sha256'] for x in rows if x['prefill_tokens']==tokens),'diagnostic IDs')
        logpath=ROOT/'raw/diagnostic'/(tag+'.log')
        trace=parse.parse(logpath.read_text(),tokens,1)
        require(trace['kept_entries']==kept and trace['fast_calls']==38,'actual dispatch count')
        roottrace=read(f'device/root-diagnostic-{tokens}.json')
        require(all(trace[k]==roottrace[k] for k in trace),'root diagnostic readback')
        require(roottrace['log_sha256']==sha(logpath),'root trace SHA')
        diagnostics.append({k:v for k,v in trace.items() if k!='records'})

    native=read('device/root-native-results.json')
    require(native['sha256']==native['remote_sha256']==read('author/checks/native/commands.json')['binary_sha256'],'native actual binary SHA')
    modes=['1','0','unset','true','trace-configured','trace-opened']
    require([r['mode'] for r in native['runs']]==modes,'native modes')
    for r in native['runs']:
        want=f"PASS numeric=49 guards=162 real_lru=72 observers=2 enabled={int(r['mode']=='1')}\n"
        require(r['returncode']==0 and r['output']==want,'native parity')
        require((ROOT/'device'/('root-native-'+r['mode']+'.log')).read_text()==want,'native raw log')
    for folder in ['author/checks/final-san','author/checks/fast','author/checks/generic','review/final-host-san']:
        log=read(folder+'/commands.json')
        require([r['mode'] for r in log['runs']]==modes,'host modes')
        for r in log['runs']:
            enabled=folder!='author/checks/generic' and r['mode']=='1'
            want=f'PASS numeric=49 guards=162 real_lru=72 observers=2 enabled={int(enabled)}'
            require(r['returncode']==0 and r['output']==want,'host correctness')
            require((ROOT/folder/(r['mode']+'.log')).read_text().strip()==want,'host raw log')
    require((ROOT/'review/final-policy-san/run.log').read_text().strip()=='RECENCY_LRU 12 streams 2880 state-transition cases PASS','independent policy')
    mutants=read('review/mutants/results.json')
    require(len(mutants)==6 and all(x['pass'] and x['compile_returncode']==0 and x['returncode']!=0 for x in mutants),'semantic mutants')

    accepted=read('device/root-accepted-control.json')
    identity=read('device/root-normal-runtime-identity.json')
    off=read('device/lfm-prefill-recency-0-artifacts.json')['binaries']
    on=read('device/lfm-prefill-recency-1-artifacts.json')['binaries']
    diag=artifact['binaries']
    accepted_map={Path(k).name:v for k,v in accepted['binaries'].items()}
    require(len(off)==7 and off==on==identity['binaries'] and identity['both_arms_identical'],'same binary A/B')
    require([k for k in off if off[k]!=accepted_map[k]]==['libcausallm.so'],'only normal app changes')
    require([k for k in diag if diag[k]!=off[k]]==['libcausallm.so'],'only diagnostic app changes')
    restored=read('device/root-normal-restored.json')
    require(restored['binaries']==accepted['binaries'],'seven local binaries restored')
    require(restored['phone_verification']['binaries']==accepted_map and
            restored['phone_verification']['verified_after_both_diagnostics'],'phone seven hashes after diagnostic')
    for name,digest in restored['sources'].items():
        require(hashlib.sha256(before[name].encode()).hexdigest()==digest,'source restoration')

    io=module('original_io','queue/repo/docs/measurements/lfm2_tl2_prefill1000_20261016/validation/analyze_prefill_io.py')
    m0=module('original_m0','queue/analyze_existing.py')
    matched=module('original_matched','queue/analyze_matched.py')
    queue=[];streams=[]
    def normalized(value):
        if isinstance(value,dict):return {k:normalized(v) for k,v in value.items() if k not in ('log','path')}
        if isinstance(value,list):return [normalized(x) for x in value]
        return value
    for arm in [0,1]:
        p=ROOT/f'queue/repo/docs/measurements/lfm2_tl2_prefill1000_20261016/continue-router-arm-io1024-{arm}.log'
        data,records=matched.analyze(p,io,m0)
        require(normalized(data)==normalized(read(f'queue/matched-{arm}-summary.json')),'queue replay differs')
        streams.append([(r['batch'],r['expert'],r['off_gu'],r['bytes'],r['calls']) for r in records['READ']])
        queue.append({k:data[k] for k in ['batches','experts','direct_calls','requested_bytes','read_window_ms','service_ms',
                                         'io_ms','copy_ms','other_read_ms','exposed_wait_ms','register_ms','excluded_warmup_calls',
                                         'traced_compute_calls','traced_moe_calls','queue_reconstruction']})
    require(streams[0]==streams[1],'queue payload order differs')
    return dict(decision='not_adopted',reason='No consistent whole-model prefill benefit; 2048 variability is large and causality unresolved.',
                normal_runs=12,diagnostic_runs=2,all_65_ids_exact=True,normal=summaries,diagnostic_counts=diagnostics,
                diagnostic_tps_used_for_performance=False,queue_profile=queue,queue_payload_identical=True,
                memory_scope=ref['memory_scope'],native_environments=6,independent_policy_transitions=2880,
                thermal_observations=dict(normal_runs=12,endpoint_snapshots=24,thermal_status_zero_snapshots=24,
                                          cached_skin_values=sorted(set(cached_skin)),hal_skin_range=[min(hal_skin),max(hal_skin)],
                                          proves_absence_of_dvfs_or_throttling=False,discarded_runs=0,causality='unresolved'),
                independent_mutants=6,local_and_phone_accepted_runtime_restored=True,
                source_and_deployment_patch_replay=True,
                scope='Archive readback only: no tests, model, compilation, SDK or phone rerun.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--write-results',action='store_true',help='Curator only, before freezing manifest')
    p.add_argument('--output',type=Path,help='Optional report outside the archive')
    a=p.parse_args()
    result=validate()
    if a.write_results:
        (ROOT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    else:
        require(read('results.json')==result,'results differ from raw evidence')
        manifest=read('manifest.json')
        # Nested original manifests are evidence and must remain covered.
        actual={str(x.relative_to(ROOT)) for x in ROOT.rglob('*') if x.is_file() and x!=ROOT/'manifest.json'}
        require(actual==set(manifest['files']),'manifest file-set mismatch')
        for name,entry in manifest['files'].items():
            f=ROOT/name
            require(sha(f)==entry['sha256'] and f.stat().st_size==entry['bytes'],'manifest hash '+name)
    report=dict(pass_=True,normal_runs=12,diagnostic_runs=2,queue_logs=2,all_65_ids_exact=True,
                source_and_patch_replay=True,accepted_runtime_restored=True,
                thermal_snapshots_verified=24,decision='not_adopted')
    report['pass']=report.pop('pass_')
    if a.output:
        require(ROOT not in a.output.resolve().parents,'write report outside frozen archive')
        a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    main()
