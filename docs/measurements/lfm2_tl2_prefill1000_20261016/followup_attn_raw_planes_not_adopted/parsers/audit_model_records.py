#!/usr/bin/env python3
"""Audit existing model artifacts and frozen runtime records without execution."""
import argparse
import json
from pathlib import Path
import re
import statistics
import sys
sys.dont_write_bytecode = True
from parse_stages import need, sha

IDS = {1024:'b6c1b801f95caded5399ec685cce0f900a20c3f9df760482dd7b5c9a62a5b92e',
       2048:'2bc7afc3c7d8e4387c8d9c7378d6792a1ef0fb56c2635f73b95e8650ac7fef33'}
ERR = ('fdinfo_read_errors', 'unknown_identity_fds', 'unknown_identity_bytes',
       'inode_size_conflicts', 'ino_without_mount_fds', 'invalid_fields', 'counter_overflow')


def memory_readback(m):
    need(m['child_waited'] is True and m['child_exit_code'] == 0 and m['child_signal'] is None, 'Child wait/success')
    need(m['mode'] == 'supervised-child' and m['requested_period_ms'] == 20, 'Memory scope/period')
    samples = m['samples']; available = [s for s in samples if s['composite_available']]
    need(len(samples) == m['sample_count'] and len(available) == m['composite_available_sample_count'], 'Memory sample counts')
    need(samples[0]['pre_exec'] is True, 'Memory pre-exec coverage')
    for s in available:
        need(s['anon_plus_fdinfo_raw_bytes'] == s['rss_anon_kib'] * 1024 + s['fdinfo_size_bytes_raw'], 'Raw memory arithmetic')
        need(s['anon_plus_fdinfo_dedup_or_raw_fallback_bytes'] == s['rss_anon_kib'] * 1024 + s['fdinfo_size_bytes_dedup_or_raw_fallback'], 'Dedup memory arithmetic')
    for key in ('raw', 'dedup_or_raw_fallback'):
        need(max(s['anon_plus_fdinfo_'+key+'_bytes'] for s in available) == m['peak_anon_plus_fdinfo_'+key+'_bytes'], 'Memory maximum')
    peak = m['peak_dedup_sample']
    need(peak in samples and peak['anon_plus_fdinfo_dedup_or_raw_fallback_bytes'] == m['peak_anon_plus_fdinfo_dedup_or_raw_fallback_bytes'], 'Peak sample membership')
    need(not any(peak.get(k,0) for k in ERR), 'Peak sample identity/error fields')
    unavailable = [i for i,s in enumerate(samples) if not s['composite_available']]
    need(unavailable and unavailable == list(range(unavailable[0],len(samples))) and
         all(samples[i]['status_read'] and samples[i]['fdinfo_directory_read'] and
             not samples[i]['rss_anon_present'] for i in unavailable),
         'Only contiguous terminal RSS-less samples unavailable')
    intervals = [b['begin_ms'] - a['begin_ms'] for a,b in zip(samples,samples[1:])]
    need(len(intervals) == m['sample_interval_ms']['count'], 'Memory interval count')
    for k,v in [('min',min(intervals)),('max',max(intervals)),('mean',sum(intervals)/len(intervals))]:
        need(abs(v-m['sample_interval_ms'][k]) < 0.00001, 'Memory interval '+k)
    return {'pid':m['pid'], 'samples':len(samples), 'available':len(available),
            'peak_kib':m['peak_anon_plus_fdinfo_dedup_or_raw_fallback_bytes']/1024,
            'peak_rss_anon_kib':peak['rss_anon_kib'],
            'peak_fd_size_bytes':peak['fdinfo_size_bytes_dedup_or_raw_fallback'],
            'peak_error_fields':{k:peak.get(k,0) for k in ERR},
            'all_sample_error_totals':{k:sum(s.get(k,0) for s in samples) for k in ERR},
            'flagged_sample_indices':[i for i,s in enumerate(samples) if any(s.get(k,0) for k in ERR)],
            'interval_ms':m['sample_interval_ms'], 'terminal_unavailable_indices':unavailable,
            'dma_buffer_identity_verified':m['dma_buffer_identity_verified']}


def read_run(evidence, job, execution, build, profile):
    t=job['tag']; need(execution['tag'] == t and execution['returncode'] == 0, 'Execution identity/return code')
    files = {suffix:evidence/(t+'.'+suffix) for suffix in ('json','ids','log','memory.json')}
    meta=json.loads(files['json'].read_text()); mem=json.loads(files['memory.json'].read_text())
    tokens=2048 if job['model']=='model2048' else 1024
    need(meta['tag']==t and meta['variant']==job['variant'] and meta['model']==job['model'], 'Job metadata identity')
    need(meta['profile'] is profile and meta['instrumented'] is profile and meta['performance_comparison_eligible'] is not profile, 'Profile eligibility')
    need(meta['valid'] is True and meta['ids_match_reference'] is True and meta['returncode']==0 and meta['concurrent_processes']==[], 'Successful exclusive model run')
    need(meta['prefill_tokens']==tokens and meta['decode_tokens']==64 and meta['moe_cache_experts']==16, 'Run length/cache')
    env=meta['environment']
    for k,v in {'NNTR_NUM_THREADS':'8','NNTR_HTP_PAGE_CACHE':'0','NNTR_LOAD_PREFETCH':'0','NNTR_MOE_CACHE_GROW':'0','NNTR_MOE_PRELOAD_EXPERTS':'128',**{k:str(v) for k,v in job['env'].items()}}.items():
        need(env.get(k)==v, 'Environment '+k)
    for k in ('NNTR_HTP_ATTN_TRACE','NNTR_HTP_PROFILE','NNTR_LAYER_PROFILE'):
        need(profile or k not in env, 'Normal run unexpectedly traced')
    ids=files['ids'].read_text().splitlines()
    need(len(ids)==65 and all(re.fullmatch(r'\d+', x) for x in ids), 'Exactly 65 dumped IDs')
    need(sha(files['ids'])==meta['ids_sha256']==IDS[tokens], 'IDs equal accepted reference')
    need(sha(files['memory.json'])==meta['full_run_memory_report_sha256'], 'Memory report hash')
    mr=memory_readback(mem)
    need(mr['peak_kib']==meta['full_run_sampled_peak_kib'] and mr['samples']==meta['full_run_memory_sample_count'], 'Memory metadata summary')
    need(mem['command'][0].endswith('/'+job['variant']+'/nntr_causallm') and mem['command'][1].endswith('/'+job['model']), 'Supervised binary/model')
    log=files['log'].read_text()
    for label,k in [('prefill','prefill'),('generation','decode')]:
        matches=re.findall(rf'^{label}: (\d+) tokens, ([\d.]+) ms, ([\d.]+) TPS$',log,re.M)
        need(len(matches)==1, 'Exactly one performance summary '+label)
        count,ms,tps=matches[0]
        need(int(count)==meta[k+'_tokens'] and float(ms)==meta[k+'_ms'] and float(tps)==meta[k+'_tps'], 'Raw performance/metadata '+label)
        # std::ostream uses significant digits: e.g. 1003.92 has coarser
        # printed resolution than 983.670. Check the actual text's last digit.
        quantum=10.0**(-len(tps.split('.')[1])) if '.' in tps else 1.0
        need(abs(1000*int(count)/float(ms)-float(tps)) <= quantum/2+1e-9, 'Rounded TPS arithmetic')
    manifest=evidence/(job['variant']+'-artifacts.json'); runtimes=json.loads(manifest.read_text())['binaries']
    need(sha(manifest)==execution['runtime_manifest_sha256'] and runtimes==execution['runtime_sha256_before'], 'Fresh runtime hashes/manifest')
    arm='control' if job['variant'].endswith('-control') else 'candidate'
    need(runtimes['libnntr_hvx_skel.so']==build[arm]['sha256'], 'Normal DSP controlled-build identity')
    return {'tag':t, 'variant':job['variant'], 'arm':arm, 'tokens':tokens,
            'repetition':int(re.search(r'-r(\d+)$',t).group(1)) if re.search(r'-r(\d+)$',t) else None,
            'prefill_tps':meta['prefill_tps'], 'decode_tps':meta['decode_tps'],
            'prefill_ms':meta['prefill_ms'], 'decode_ms':meta['decode_ms'],
            'config_sha256':meta['model_config_sha256'], 'ids_sha256':meta['ids_sha256'],
            'runtime_sha256':runtimes, 'memory':mr,
            'environment':{k:v for k,v in env.items() if k not in ('LD_LIBRARY_PATH','ADSP_LIBRARY_PATH','NNTR_GENERATED_IDS')},
            'input_sha256':{k:sha(v) for k,v in files.items()},
            'runtime_manifest_sha256':sha(manifest), 'performance_comparison_eligible':not profile}


def summarize(runs):
    summaries=[]
    for n in (1024,2048):
        rows=[x for x in runs if x['tokens']==n]; reps=sorted({r['repetition'] for r in rows})
        sides={arm:sorted([r for r in rows if r['arm']==arm],key=lambda r:r['repetition']) for arm in ('control','candidate')}
        need(all([r['repetition'] for r in rr]==reps for rr in sides.values()), 'Complete unique pairs')
        item={'tokens':n,'pairs':len(reps),'repetitions':reps}
        for key in ('prefill_tps','decode_tps'):
            c=[r[key] for r in sides['control']];b=[r[key] for r in sides['candidate']]
            mc=statistics.median(c);mb=statistics.median(b)
            item[key]={'control':c,'candidate':b,'median_control':mc,'median_candidate':mb,
                       'median_pct':100*(mb/mc-1),'pairs_faster':sum(y>x for x,y in zip(c,b)),
                       'paired_deltas_pct':[100*(y/x-1) for x,y in zip(c,b)]}
        for arm in sides:
            item[arm+'_max_peak_kib']=max(r['memory']['peak_kib'] for r in sides[arm])
        summaries.append(item)
    return summaries


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('evidence','results','fixed-build','output'):
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--include-confirmation',action='store_true')
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    build=json.loads((a.fixed_build/'validation.json').read_text())
    allruns=[];groups={}
    for group in ('normal','confirmation') if a.include_confirmation else ('normal',):
        jobs=json.loads((a.results/('root-'+group+'-fixed-jobs.json')).read_text())
        executions=json.loads((a.results/('root-'+group+'-fixed-executions.json')).read_text())
        need(len(jobs)==len(executions)==12,'Exactly twelve '+group+' runs')
        runs=[read_run(a.evidence,j,e,build,False) for j,e in zip(jobs,executions)]
        for n in (1024,2048):
            rows=[r for r in runs if r['tokens']==n]
            need([r['arm'] for r in rows]==(['control','candidate','candidate','control','control','candidate'] if group=='normal' else ['candidate','control','control','candidate','candidate','control']), 'Prespecified pair order')
        groups[group]={'runs':runs,'summary':summarize(runs)};allruns+=runs
        claimed=json.loads((a.results/('root-'+group+'-fixed-summary.json')).read_text())
        for calc,old in zip(groups[group]['summary'],claimed['summary']):
            for key in ('prefill_tps','decode_tps','control_max_peak_kib','candidate_max_peak_kib'):
                need(calc[key]==old[key],'Root '+group+' summary independent recomputation '+key)
    non_dsp={k:v for k,v in allruns[0]['runtime_sha256'].items() if k!='libnntr_hvx_skel.so'}
    need(all({k:v for k,v in r['runtime_sha256'].items() if k!='libnntr_hvx_skel.so'}==non_dsp for r in allruns),'Six common non-DSP binaries')
    need(all(r['environment']==allruns[0]['environment'] for r in allruns),'Normal environment unchanged')
    need(all(len({r['config_sha256'] for r in allruns if r['tokens']==n})==1 for n in (1024,2048)),'Same model config per length')
    result={'status':'PASS_EXISTING_RECORD_READBACK','groups':groups,'all_runs':len(allruns),
            'all_ids_and_peak_checks_pass':True,'common_non_dsp_runtime_sha256':non_dsp,
            'limitations':['Small paired samples with wide observed timing variance; no samples/outliers were removed.',
              'The candidate changes prefill attention, so decode TPS differences do not establish a decode kernel effect.',
              'Peak is sampled process RssAnon + deduplicated fdinfo allocated sizes, not total device physical memory. Peak errors and all-sample errors are separate.',
              'No device, compiler, linker or model was executed by this readback. Recorded fresh device hashes bind each run to the controlled artifact set.']}
    if a.include_confirmation:
        pooled=summarize(allruns);result['pooled']=pooled
        result['prespecified_confirmation_plan_sha256']=sha(a.results/'root-confirmation-plan.json')
        result['prespecified_adoption_metric_pass']=all(x['prefill_tps']['median_pct']>=0 and x['prefill_tps']['pairs_faster']>=4 for x in pooled)
    name='pooled-normal-readback.json' if a.include_confirmation else 'initial-normal-readback.json'
    (a.output/name).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'all_runs':len(allruns),'summary':result.get('pooled',groups['normal']['summary']),
                      'prespecified_adoption_metric_pass':result.get('prespecified_adoption_metric_pass')}))


if __name__=='__main__':
    main()
