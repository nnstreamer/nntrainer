#!/usr/bin/env python3
"""Recompute the rejected Qprep cohort from archived raw bytes, without SDK/phone."""
import hashlib,json,re,statistics
CONTROL='lfm-attn-qprep-group4-control'; CANDIDATE='lfm-attn-qprep-group4'; DIAGNOSTIC='lfm-attn-qprep-group4-diagnostic'
PROD='a07332c9043b6760a42392cae06a237319e6a1e383829edc0bee5d838b435df8'
TESTS='f962d48fabdb73ce2702333f67907b8c5f5ddd86e17c01c38d37048ea66db23f'
TRACE='af263c745b23b9215a5707f9bc60c008ef2c168ea0880d9b201adddef4871a39'
IDS={1024:'b6c1b801f95caded5399ec685cce0f900a20c3f9df760482dd7b5c9a62a5b92e',2048:'2bc7afc3c7d8e4387c8d9c7378d6792a1ef0fb56c2635f73b95e8650ac7fef33'}
ENV={'NNTR_NUM_THREADS':'8','NNTR_HTP_PAGE_CACHE':'0','NNTR_LOAD_PREFETCH':'0','NNTR_MOE_CACHE_GROW':'0','NNTR_MOE_PRELOAD_EXPERTS':'128','NNTR_INIT_TRACE':'1','NNTR_HTP_PREDICT':'2','NNTR_MOE_PREFETCH_RESIDUAL':'0.5','NNTR_HTP_TL2_STABLE_SLOTS':'1','NNTR_LFM2_PREFILL_EARLY':'1','NNTR_LFM2_DECODE_PAIR_READ':'1','NNTR_LFM2_DECODE_DEMAND_HANDOFF':'1','NNTR_LFM2_ROUTER_INPUT_HANDOFF':'1','NNTR_LFM2_ROUTER_ARM':'1'}
def need(x,msg):
 if not x:raise ValueError(msg)
def sha(data):return hashlib.sha256(data).hexdigest()

def audit(get):
 J=lambda p:json.loads(get(p)); T=lambda p:get(p).decode()
 need(sha(get('source/production.patch'))==PROD,'production SHA')
 need(sha(get('source/tests.patch'))==TESTS,'tests SHA')
 need(sha(get('diagnostic/diagnostic.patch'))==TRACE,'diagnostic SHA')
 source=J('source/source-hashes.json'); tests=J('source/tests-hashes.json')
 for r,h in source.items():
  need(sha(get('source/candidate/'+r))==h['candidate_sha256'],'candidate source '+r)
  if h['baseline_sha256']:need(sha(get('source/baseline/'+r))==h['baseline_sha256'],'baseline source '+r)
 for r,h in tests.items():need(sha(get('source/candidate/'+r))==h,'fixture '+r)
 author=J('validation/author/validation.json'); review=J('validation/independent/validation.json')
 need(author['production_sha256']==review['production_patch_sha256']==PROD,'author/review production')
 need(author['tests_sha256']==review['tests_patch_sha256']==TESTS and review['status']=='PASS','author/review tests')
 for root,path in [('validation/author','checks/frozen-strict/attn_qprep_native_check.log'),('validation/independent','quant-san-fixed/attn_qprep_native_check.log')]:
  text=T(root+'/'+path);need('576 cases PASS' in text and '31 PASS' in text,'strict HVX output')
  need('runtime error:' not in text and 'ERROR: AddressSanitizer' not in text,'strict HVX sanitizer')
 need('runtime error: load of misaligned address' in T('validation/author/checks/final/attn_qprep_native_check.log'),'initial recover error evidence')
 need('runtime error: load of misaligned address' in T('validation/independent/quant-san/attn_qprep_native_check.log'),'independent strict failure evidence')
 fixed=J('fixed-build/build/validation.json'); trace=J('fixed-build/diagnostic/validation.json')
 need(fixed['status']=='PASS' and fixed['frozen_objects']==63 and fixed['unchanged_objects']==59,'63/59 fixed objects')
 need(len(fixed['candidate_recompiled_tus'])==4 and fixed['baseline_objects_unchanged'] and fixed['all_other_link_inputs_identical'],'four-TU boundary')
 need(fixed['shared_linker_setting']=='-no-threads' and fixed['not_a_claim_of_original_3b_binary_identity'],'link/identity scope')
 need(trace['status']=='PASS' and trace['macro_off_object_exact'] and trace['repeat_link_identical'] and trace['runtime_imports_identical'],'trace build')
 shared=J('fixed-build/shared-link-audit/validation.json')
 need(shared['status']=='PASS' and len(shared['shared_linker_input_sha256'])==7 and shared['same_path_relink_both_original_artifacts_exact'],'shared linker input supplement')
 artifacts={v:J('artifacts/'+v+'.json') for v in [CONTROL,CANDIDATE,DIAGNOSTIC]}
 runtime={v:x['binaries'] for v,x in artifacts.items()}
 runtime_identity=J('validation/root/root-normal-runtime-identity.json')
 for arm,variant in [('control',CONTROL),('candidate',CANDIDATE),('diagnostic',DIAGNOSTIC)]:
  need(artifacts[variant]['source_patch_sha256']==sha(get('validation/root/'+arm+'-deployment.patch')),'deployment source patch')
  if arm!='diagnostic':need(artifacts[variant]==runtime_identity[arm],'normal runtime identity')
 need(J('validation/root/root-diagnostic-validation.json')['runtime_manifest']==artifacts[DIAGNOSTIC],'diagnostic runtime identity')
 baseline=J('validation/root/root-baseline.json')
 need(baseline['fixed_build_validation_sha256']==sha(get('fixed-build/build/validation.json')) and baseline['source_map']==source,'root fixed-build/source binding')
 need(all(len(x)==7 for x in runtime.values()),'seven runtime files')
 host={k:v for k,v in runtime[CONTROL].items() if k!='libnntr_hvx_skel.so'}
 need(all({k:v for k,v in x.items() if k!='libnntr_hvx_skel.so'}==host for x in runtime.values()),'common six host binaries')
 for arm,variant in [('control',CONTROL),('candidate',CANDIDATE)]:
  need(runtime[variant]['libnntr_hvx_skel.so']==fixed[arm]['sha256'],'controlled runtime')
 need(runtime[DIAGNOSTIC]['libnntr_hvx_skel.so']==trace['diagnostic_so_sha256'],'diagnostic runtime')
 need(runtime[CONTROL]['libnntr_hvx_skel.so']!=runtime[CANDIDATE]['libnntr_hvx_skel.so'],'different DSP candidates')
 jobs=J('validation/root/root-normal-jobs.json');need(len(jobs)==12,'normal12')
 need([j['variant'] for j in jobs]==[CONTROL,CANDIDATE,CANDIDATE,CONTROL,CONTROL,CANDIDATE]*2,'AB/BA/AB order')
 memory_rows=[]
 def run(tag,n,diagnostic=False):
  prefix=('diagnostic/raw/' if diagnostic else 'raw/')+tag
  r=J(prefix+'.json');m=J(prefix+'.memory.json');ids=get(prefix+'.ids');log=T(prefix+'.log')
  need(r['tag']==tag and r['prefill_tokens']==n and r['decode_tokens']==64 and r['moe_cache_experts']==16,'run identity/cache')
  need(r['returncode']==0 and r['valid'] and r['ids_match_reference'] and not r['concurrent_processes'],'completed run')
  need(r['profile']==diagnostic and r['instrumented']==diagnostic and r['performance_comparison_eligible']!=diagnostic,'performance eligibility')
  need(sha(ids)==r['ids_sha256']==IDS[n] and len(ids.decode().splitlines())==65,'65 exact IDs')
  for k,v in ENV.items():need(r['environment'].get(k)==v,'fixed env '+k)
  expected=set(ENV)|{'LD_LIBRARY_PATH','ADSP_LIBRARY_PATH','NNTR_GENERATED_IDS'}|({'NNTR_HTP_ATTN_TRACE','NNTR_HTP_PROFILE','NNTR_OP_TIME','NNTR_LAYER_PROFILE'} if diagnostic else set())
  need(set(r['environment'])==expected,'unexpected or missing environment')
  if diagnostic:
   for k,v in {'NNTR_HTP_ATTN_TRACE':'1','NNTR_HTP_PROFILE':'1','NNTR_LAYER_PROFILE':'1','NNTR_OP_TIME':'prefill'}.items():need(r['environment'][k]==v,'diagnostic env '+k)
  cfg=J('configuration/lfm2-'+str(n)+'/nntr_config.json')
  need(r['model_config_sha256']==sha(get('configuration/lfm2-'+str(n)+'/nntr_config.json')) and cfg['moe_cache_experts']==16 and cfg['num_to_generate']==64 and cfg['init_seq_len']==n and cfg['moe_layer_dtype']=='TL2CX_WH' and cfg['fc_layer_dtype']=='QS4CX_WH_AFFINE','model config/quant')
  peak=m['peak_dedup_sample'];need(m['child_waited'] and m['child_exit_code']==0 and m['child_signal'] is None and m['requested_period_ms']==20 and m['mode']=='supervised-child','whole child monitor')
  need(peak['composite_available'] and peak['dedup_complete_for_readable_size_fds'] and not any(peak[k] for k in ['fdinfo_read_errors','unknown_identity_fds','inode_size_conflicts','invalid_fields','counter_overflow']),'complete peak')
  value=m['peak_anon_plus_fdinfo_dedup_or_raw_fallback_bytes']
  available=[x['anon_plus_fdinfo_dedup_or_raw_fallback_bytes'] for x in m['samples'] if x['composite_available']]
  need(r['full_run_memory_report_sha256']==sha(get(prefix+'.memory.json')) and r['full_run_sampled_peak_kib']*1024==value==peak['anon_plus_fdinfo_dedup_or_raw_fallback_bytes']==max(available),'memory source/max')
  need(r['full_run_memory_sample_count']==m['sample_count']==len(m['samples']) and len(available)==m['composite_available_sample_count'],'memory samples')
  for label,field in [('prefill','prefill'),('generation','decode')]:
   rows=re.findall(label+r': (\d+) tokens, ([\d.]+) ms, ([\d.]+) TPS',log)
   need(len(rows)==1 and tuple(map(float,rows[0]))==(r[field+'_tokens'],r[field+'_ms'],r[field+'_tps']),'raw throughput readback')
  for marker in ['[MoE] preload_experts=128 cache_slots=608','[HTP-PREFETCH] direct slots=32 bytes=64618496 readers=2 pair_read=1','[HTP] demand read pool: readers=4']:
   need(marker in log,'unchanged read/cache marker')
  memory_rows.append({'tag':tag,'peak_kib':r['full_run_sampled_peak_kib'],'sample_count':m['sample_count'],'composite_available_sample_count':m['composite_available_sample_count'],'sample_interval_ms':m['sample_interval_ms'],'sample_scan_ms':m['sample_scan_ms'],'peak_complete':True,'dma_buffer_identity_verified':m['dma_buffer_identity_verified']})
  return r,m
 rows=[]
 for i,j in enumerate(jobs):
  n=1024 if i<6 else 2048;rep=[1,1,2,2,3,3][i%6];side=int(j['variant']==CANDIDATE)
  tag='continue-attn-qprep-group4'+('2048' if n==2048 else '')+f'-{side}-r{rep}'
  need(j['tag']==tag,'normal tag');r,m=run(tag,n)
  need(r['variant']==j['variant'] and r['model']==j['model']==('model' if n==1024 else 'model2048'),'normal model variant')
  row={k:r[k] for k in ['tag','variant','prefill_tokens','prefill_tps','decode_tps','full_run_sampled_peak_kib','ids_sha256']};row['repetition']=rep;rows.append(row)
 computed=[]
 for n in [1024,2048]:
  g={v:sorted((r for r in rows if r['prefill_tokens']==n and r['variant']==v),key=lambda r:r['repetition']) for v in [CONTROL,CANDIDATE]};z={'tokens':n}
  for key in ['prefill_tps','decode_tps']:
   x,y=[[r[key] for r in g[v]] for v in [CONTROL,CANDIDATE]];xm,ym=statistics.median(x),statistics.median(y)
   z[key]={'control':x,'candidate':y,'median_control':xm,'median_candidate':ym,'median_pct':100*(ym/xm-1),'pairs_faster':sum(b>a for a,b in zip(x,y)),'paired_deltas_pct':[100*(b/a-1) for a,b in zip(x,y)]}
  for label,v in [('control',CONTROL),('candidate',CANDIDATE)]:z[label+'_max_peak_kib']=max(r['full_run_sampled_peak_kib'] for r in g[v])
  computed.append(z)
 normal=J('validation/root/root-normal-summary.json')
 need(normal['runs']==rows and normal['summary']==computed and normal['all_checks_pass'],'normal summary independently recomputed')
 # Independent raw-token parser, not the author's parser implementation.
 diagnostics=[];original=J('validation/root/root-diagnostic-summary.json')
 for root in original:
  n=root['model'];tag=root['tag'];r,m=run(tag,n,True);pid=root['pid']
  need(m['pid']==pid and r['variant']==DIAGNOSTIC and root['trace_enabled_binary_verified'] and not root['normal_performance_eligible'],'diagnostic scope/PID')
  raw=get('diagnostic/raw/'+tag+'.live.logcat.txt');need(sha(raw)==root['live_log_sha256'],'live raw hash')
  sequence={};duplicates=0
  for line in raw.decode().splitlines():
   match=re.fullmatch(r'\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+\s+(\d+)\s+\d+\s+\w\s+.*\[nntr-qprep-group4\] seq=(\d+) groups=(\d+) k=(\d+) stride=(\d+) bytes=(\d+) !',line)
   need(match is not None,'malformed/truncated diagnostic line');p,seq,groups,k,stride,bytes_=map(int,match.groups())
   need(p==pid and seq>0 and groups in (1152,896) and k==64 and stride==2048 and bytes_==groups*1024,'diagnostic fields')
   val=(groups,bytes_)
   if seq in sequence:need(sequence[seq]==val,'conflicting duplicate');duplicates+=1
   else:sequence[seq]=val
  count=160 if n==2048 else 0;need(len(sequence)==count,'expected dispatch count')
  need(not sequence or min(sequence)==1 and max(sequence)==count,'sequence gaps')
  need(all(v[0]==(1152 if seq%2 else 896) for seq,v in sequence.items()),'chunk order')
  rec={'model':n,'pid':pid,'raw_tagged_records':len(raw.decode().splitlines()),'unique_calls':count,'transport_duplicates':duplicates,'groups_per_call_counts':{str(g):sum(v[0]==g for v in sequence.values()) for g in [1152,896]},'total_groups':sum(v[0] for v in sequence.values()),'logical_gather_bytes_skipped':sum(v[1] for v in sequence.values())}
  need(all(root[k]==v for k,v in rec.items()),'root diagnostic summary')
  need(J('diagnostic/raw/'+tag+'.parsed.json')==root,'stored root parser output')
  need(root['parser_sha256']==sha(get('diagnostic/parse_trace.py')),'root parser binding');diagnostics.append(rec)
 # Optional phase data is an existing diagnostic-only observation, never a normal timing cohort.
 stages=J('validation/root/root-diagnostic-attention-stages.json')
 stage_raw=get('diagnostic/raw/'+stages['tag']+'.logcat.txt')
 need(sha(stage_raw)==stages['log_sha256'] and stages['status']=='PASS','phase log identity')
 observed=[];wrappers=[]
 for line in stage_raw.decode().splitlines():
  if 'HTP attn stages q: n_q=2048 ' in line:
   fields=dict((k,int(v)) for k,v in re.findall(r'(\w+)=(\d+)',line.split('HTP attn stages q: ',1)[1]))
   observed.append(fields)
  if 'HTP attn trace q:' in line and ' n_q=2048 ' in line:
   fields=dict(re.findall(r'(\w+)=(\S+)',line.split('HTP attn trace q: ',1)[1]));need(fields['err']=='0x0','phase wrapper error');wrappers.append(fields)
 need(len(observed)==len(wrappers)==10 and observed==stages['stage_records'],'ten diagnostic attention stages')
 need({k:sum(r[k] for r in observed) for k in stages['sums_us']}==stages['sums_us'],'diagnostic stage sums')
 need({k:sum(int(r[k]) for r in wrappers) for k in stages['wrapper_sums_us']}==stages['wrapper_sums_us'],'diagnostic wrapper sums')
 restored=J('validation/root/root-normal-restored.json')
 need(restored['status']=='PASS' and restored['tracked_source_clean'] and restored['source_restored_after_diagnostics'],'restoration status')
 need(restored['source_sha256']=={r:h['baseline_sha256'] for r,h in source.items()},'five restored sources')
 accepted=restored['accepted_local_sha256'];need(len(accepted)==7 and accepted==restored['accepted_phone_sha256']==J('artifacts/accepted-restored-runtime.json')['binaries'],'accepted seven identity')
 need(accepted['libnntr_hvx_skel.so']=='3baddb92d14d1d23fe9b6c6a7cacfe3a6cd4708001c110bb8b15e1b0b7276f7c','original accepted DSP')
 need({k:v for k,v in accepted.items() if k!='libnntr_hvx_skel.so'}==host,'accepted common host runtime')
 return {'decision':'not_adopted','reason':'No consistent 2048 prefill gain; median lower and only 1/3 paired wins. Control 983.197 outlier is retained without causal diagnosis. The 1024 and decode changes are not attributed to this 2048-prefill helper.','normal':computed,'normal_runs':12,'diagnostics':diagnostics,'all14_runs_65_ids_exact':True,'strict_whole_memory_checks_pass':True,'memory_sampling':memory_rows,'runtime_binaries':runtime,'original_accepted_seven_restored':True,'fixed_objects':63,'unchanged_common_objects':59,'candidate_recompiled_tus':4,'shared_linker_inputs_supplemented':7,'source_tests_diagnostic_hashes':[PROD,TESTS,TRACE],'raw_readback_independent_of_author_parser':True,'diagnostic_attention_stage_reference_only':{'tag':stages['tag'],'sums_us':stages['sums_us'],'scope':stages['scope']},'new_tests_or_phone_runs':False,'memory_scope':'max sampled child RssAnon + inode-deduplicated fdinfo allocated sizes; not total physical/device memory; unreported DSP/driver/VTCM and between-sample spikes excluded.'}
