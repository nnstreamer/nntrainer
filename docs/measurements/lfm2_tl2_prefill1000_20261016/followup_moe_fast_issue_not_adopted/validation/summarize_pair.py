from pathlib import Path
import argparse, hashlib, json, re, statistics
p=argparse.ArgumentParser()
for name in ('jobs','control','candidate','output'): p.add_argument('--'+name,required=True)
a=p.parse_args();e=Path('/home/jungone-lee/workspace/nntrainer-lfm2-tl2-prefill1000/docs/measurements/lfm2_tl2_prefill1000_20261016');rows=[];configs={}
expected_ids={1024:'b6c1b801f95caded5399ec685cce0f900a20c3f9df760482dd7b5c9a62a5b92e',2048:'2bc7afc3c7d8e4387c8d9c7378d6792a1ef0fb56c2635f73b95e8650ac7fef33'}
for j in json.loads(Path(a.jobs).read_text()):
 tag=j['tag'];d=json.loads((e/(tag+'.json')).read_text());m=json.loads((e/(tag+'.memory.json')).read_text());s=m['peak_dedup_sample']
 assert d['valid'] and d['performance_comparison_eligible'] and not d['instrumented'] and not d['profile'] and d['returncode']==0 and not d['concurrent_processes'],tag
 assert d['moe_cache_experts']==16 and d['decode_tokens']==64
 for k,v in {'NNTR_HTP_PAGE_CACHE':'0','NNTR_MOE_CACHE_GROW':'0','NNTR_MOE_PRELOAD_EXPERTS':'128','NNTR_LFM2_ROUTER_INPUT_HANDOFF':'1'}.items(): assert d['environment'][k]==v,(tag,k)
 for k,v in j['env'].items(): assert d['environment'][k]==str(v),(tag,k)
 assert d['ids_match_reference'] and d['ids_sha256']==expected_ids[d['prefill_tokens']] and hashlib.sha256((e/(tag+'.ids')).read_bytes()).hexdigest()==d['ids_sha256']
 assert m['child_exit_code']==0 and m['child_waited'] and m['requested_period_ms']==20 and s['dedup_complete_for_readable_size_fds']
 assert not any(s[k] for k in ['fdinfo_read_errors','unknown_identity_fds','inode_size_conflicts','invalid_fields','counter_overflow'])
 assert hashlib.sha256((e/(tag+'.memory.json')).read_bytes()).hexdigest()==d['full_run_memory_report_sha256']
 assert d['full_run_sampled_peak_kib']==m['peak_anon_plus_fdinfo_dedup_or_raw_fallback_bytes']/1024
 configs.setdefault(d['prefill_tokens'],set()).add(d['model_config_sha256'])
 row={k:d[k] for k in ['tag','variant','prefill_tokens','prefill_tps','decode_tps','full_run_sampled_peak_kib','ids_sha256']};row['repetition']=int(re.search(r'-r(\d+)$',tag)[1]);rows.append(row)
assert all(len(v)==1 for v in configs.values())
summaries=[]
for n in sorted(configs):
 x=sorted([r for r in rows if r['prefill_tokens']==n and r['variant']==a.control],key=lambda r:r['repetition']);y=sorted([r for r in rows if r['prefill_tokens']==n and r['variant']==a.candidate],key=lambda r:r['repetition']);assert [r['repetition'] for r in x]==[1,2,3]==[r['repetition'] for r in y];s={'tokens':n}
 for k in ['prefill_tps','decode_tps']:
  av=[r[k] for r in x];bv=[r[k] for r in y];am=statistics.median(av);bm=statistics.median(bv);s[k]={'control':av,'candidate':bv,'median_control':am,'median_candidate':bm,'median_pct':100*(bm/am-1),'pairs_faster':sum(b>a for a,b in zip(av,bv)),'paired_deltas_pct':[100*(b/a-1) for a,b in zip(av,bv)]}
 for k,v in [('control',x),('candidate',y)]:s[k+'_max_peak_kib']=max(r['full_run_sampled_peak_kib'] for r in v)
 summaries.append(s)
result={'control':a.control,'candidate':a.candidate,'runs':rows,'summary':summaries,'all_checks_pass':True,'memory_scope':'Sampled whole child lifetime RssAnon + deduplicated fdinfo allocation sizes. Not total device physical memory; unreported DSP/driver/VTCM and between-sample spikes excluded.'}
Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summaries,indent=2))
