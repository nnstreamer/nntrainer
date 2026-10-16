from pathlib import Path
import hashlib,json,subprocess,sys
r=Path('/home/jungone-lee/workspace/nntrainer-lfm2-tl2-prefill1000');e=r/'docs/measurements/lfm2_tl2_prefill1000_20261016';v,tag=sys.argv[1:];env=json.loads(Path('/tmp/nntr-lfm2-fc-fast-issue-20261016/root-normal-jobs.json').read_text())[0]['env'];command=[sys.executable,str(e/'bench.py'),v,tag,'--model','model2048','--memory','--profile']
for k,val in env.items():command+=['--env',k+'='+str(val)]
subprocess.run(command,check=True,stdout=subprocess.PIPE,text=True)
d=json.loads((e/(tag+'.json')).read_text());raw=subprocess.check_output(['adb','-s','R3CY70LV96T','logcat','-d','-v','threadtime','-T',d['device_logcat_start']],text=True);lines=[s for s in raw.splitlines() if 'LFM-AFC2' in s];(e/(tag+'.affine.logcat.txt')).write_text('\n'.join(lines)+'\n')
m=json.loads((e/(tag+'.memory.json')).read_text());p=m['peak_dedup_sample'];assert d['valid'] and d['ids_match_reference'] and d['profile'] and d['instrumented'] and not d['performance_comparison_eligible'];assert not d['concurrent_processes'] and d['moe_cache_experts']==16 and d['prefill_tokens']==2048 and d['decode_tokens']==64
for k,val in {'NNTR_HTP_PAGE_CACHE':'0','NNTR_MOE_CACHE_GROW':'0','NNTR_MOE_PRELOAD_EXPERTS':'128','NNTR_LFM2_ROUTER_INPUT_HANDOFF':'1'}.items():assert d['environment'][k]==val
assert d['ids_sha256']=='2bc7afc3c7d8e4387c8d9c7378d6792a1ef0fb56c2635f73b95e8650ac7fef33';assert hashlib.sha256((e/(tag+'.ids')).read_bytes()).hexdigest()==d['ids_sha256'];assert m['child_exit_code']==0 and p['dedup_complete_for_readable_size_fds'] and not any(p[k] for k in ['fdinfo_read_errors','unknown_identity_fds','inode_size_conflicts']);print(tag,'IDs and memory PASS; captured',len(lines),'diagnostic lines; excluded from TPS',flush=True)
