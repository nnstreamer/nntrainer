#!/usr/bin/env python3
import argparse,hashlib,json,os,shutil,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--sanitize',action='store_true');a=p.parse_args()
D=Path(__file__).parent;O=a.output;O.mkdir(parents=True,exist_ok=True);B=a.repo/'nntrainer/tensor/htp_backend';H=a.repo/'test/htp/host';rel=Path('nntrainer/tensor/htp_backend/hmx/hexkl_mm_u8i4_dma.c');commands=[];hashes={}
wrappers=['hexkl_micro_hmx_acc_clear_int32','hexkl_micro_hmx_mm_u8i4','hexkl_micro_hmx_acc_read_int32','hexkl_dma_ring_push2d','hexkl_dma_ring_drain','hvx_worker_pool_run','hvx_worker_pool_submit','hvx_worker_pool_wait','hvx_quant_params_pack_u8_ah','hvx_quant_pack_u8_ah']
for name,root in [('baseline',a.reference),('candidate',a.source)]:
 W=O/name;(W/'hmx').mkdir(parents=True,exist_ok=True);s=(root/rel).read_text();hashes[name]=hashlib.sha256(s.encode()).hexdigest()
 if name=='candidate':s+='\nint review_direct_guard(const hexkl_weight_u8i4_table*t,const uint8_t*v,uint32_t M,uint32_t K,const uint32_t*h,uint32_t n,const hexkl_mm_opts*o,uint32_t ab,const uint32_t*w,uint32_t wb,uint32_t arena){return lfm2_fc_hmx_issue_ok(t,v,M,K,h,n,o,ab,w,wb,arena);}\n'
 (W/'hmx/kernel.c').write_text(s);shutil.copy2(D/'hexkl_hmx_mm.h',W/'hmx/hexkl_hmx_mm.h');shutil.copy2(root/'nntrainer/tensor/htp_backend/htp_tl2_opts.h',W/'htp_tl2_opts.h')
 c=['gcc','-std=c99','-O2','-g','-Wall','-Wextra','-Werror','-Wno-unused-parameter','-ffunction-sections','-fdata-sections']
 if a.sanitize:c+=['-fsanitize=address,undefined','-fno-sanitize-recover=all']
 if name=='candidate':c+=['-DREVIEW_CANDIDATE=1']
 for inc in [H,H/'stub',H/'standin',B.parent,B/'hmx',B/'hvx']:c+=['-I',str(inc)]
 c += [str(D/'caller_check.c'),str(H/'hvx_scalar_stubs.c'),str(H/'standin/hvx_scalar.c'),str(W/'hmx/kernel.c'),'-Wl,--gc-sections','-lm','-o',str(W/'check')]+['-Wl,--wrap='+s for s in wrappers]
 commands.append(c);r=subprocess.run(c,capture_output=True,text=True);(W/'build.log').write_text(r.stdout+r.stderr)
 if r.returncode:raise RuntimeError(r.stdout+r.stderr)
 env=os.environ.copy();env.update(ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',UBSAN_OPTIONS='halt_on_error=1');r=subprocess.run([str(W/'check')],capture_output=True,text=True,env=env);(W/'run.log').write_text(r.stdout+r.stderr)
 if r.returncode:raise RuntimeError(f'{name} failed: '+r.stdout+r.stderr)
 print(name,r.stdout.splitlines()[-1],flush=True)
assert (O/'baseline/run.log').read_bytes()==(O/'candidate/run.log').read_bytes(),'baseline/candidate event/order mismatch'
(O/'commands.json').write_text(json.dumps(commands,indent=2)+'\n');(O/'result.json').write_text(json.dumps({'pass':True,'source_sha256':hashes,'case_count':46,'whole_caller_control_events_exact':True,'limits':['SDK packet arithmetic mocked; all actual packet addresses and order checked','Direct candidate helper exported in test-only appended wrapper','Host HMX header uses device alignment predicate and records packet addresses','No hardware timing claim']},indent=2)+'\n');print('Whole caller packet/address/control equality PASS')
