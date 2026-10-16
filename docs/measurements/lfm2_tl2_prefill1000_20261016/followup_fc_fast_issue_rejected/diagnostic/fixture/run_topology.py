#!/usr/bin/env python3
from pathlib import Path
import json,subprocess,shutil,os,argparse
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
D=Path(__file__).resolve().parent.parent;B=a.repo/'nntrainer/tensor/htp_backend';H=a.repo/'test/htp/host';rel=Path('nntrainer/tensor/htp_backend/hmx/hexkl_mm_u8i4_dma.c');commands=[]
wraps=['hexkl_micro_hmx_acc_clear_int32','hexkl_micro_hmx_mm_u8i4','hexkl_micro_hmx_acc_read_int32','hexkl_dma_ring_push2d','hexkl_dma_ring_drain','hvx_worker_pool_run','hvx_worker_pool_submit','hvx_worker_pool_wait','hvx_quant_params_pack_u8_ah','hvx_quant_pack_u8_ah']
for name in ['control','candidate']:
 W=a.output/name;(W/'hmx').mkdir(parents=True,exist_ok=True);shutil.copytree(D/'fixture/stub',W/'stub',dirs_exist_ok=True);shutil.copy2(D/name/rel,W/'hmx/kernel.c');shutil.copy2(B/'htp_tl2_opts.h',W/'htp_tl2_opts.h');shutil.copy2(D/'fixture/hexkl_hmx_mm.h',W/'hmx/hexkl_hmx_mm.h')
 c=['gcc','-std=c99','-O1','-g','-Wall','-Wextra','-Wno-unused-parameter','-ffunction-sections','-fdata-sections','-fsanitize=address,undefined','-fno-sanitize-recover=all','-DNNTR_LFM2_AFFINE_FC_TRACE=1','-DREVIEW_CANDIDATE='+str(int(name=='candidate'))]
 for inc in [W/'stub',H/'stub',H/'standin',B.parent,B/'hmx',B/'hvx']: c+=['-I',str(inc)]
 c += [str(D/'fixture/topology.c'),str(H/'hvx_scalar_stubs.c'),str(H/'standin/hvx_scalar.c'),str(W/'hmx/kernel.c'),'-Wl,--gc-sections','-lm','-o',str(W/'check')]+['-Wl,--wrap='+w for w in wraps]
 commands.append(c);p=subprocess.run(c,capture_output=True,text=True);(W/'build.log').write_text(p.stdout+p.stderr);assert p.returncode==0,p.stdout+p.stderr
 p=subprocess.run([str(W/'check')],capture_output=True,text=True,env=dict(os.environ,ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',UBSAN_OPTIONS='halt_on_error=1'));(W/'run.log').write_text(p.stdout+p.stderr);assert p.returncode==0,p.stdout+p.stderr;print(name,p.stdout.splitlines()[-1])
 for rows in (1024,2048):
  p=subprocess.run([str(W/'check'),str(rows)],capture_output=True,text=True,env=dict(os.environ,ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',UBSAN_OPTIONS='halt_on_error=1'));(W/('cohort'+str(rows)+'.log')).write_text(p.stdout+p.stderr);assert p.returncode==0,p.stdout+p.stderr;print(name,p.stdout.splitlines()[-1])
(a.output/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
