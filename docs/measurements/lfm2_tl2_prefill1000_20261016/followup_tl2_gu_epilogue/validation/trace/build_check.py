#!/usr/bin/env python3
import hashlib,json,subprocess
from pathlib import Path
D=Path(__file__).resolve().parent
raw=json.loads((D.parent/'checks/codegen/commands.json').read_text())[0]
common=raw[:raw.index('-fstack-usage')]+['-fstack-usage']
O=D/'v81';O.mkdir(exist_ok=True)
commands=[];results={};nm=Path(common[0]).with_name('hexagon-nm')
rel='nntrainer/tensor/htp_backend/hmx/hexkl_mm_u8i4_moe.c'
for mode in ['baseline','off','on']:
 root=D/('baseline' if mode=='baseline' else 'candidate');flags=['-DNNTR_LFM2_TL2_GU_TRACE='+('1' if mode=='on' else '0')]
 out=O/(mode+'.o');cmd=common+flags+['-c',str(root/rel),'-o',str(out)];commands.append(cmd)
 with (O/(mode+'.log')).open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
 syms=subprocess.check_output([str(nm),'-u',str(out)],text=True);(O/(mode+'.undefined.txt')).write_text(syms)
 if '__atomic' in syms or '__sync' in syms:raise RuntimeError('unexpected atomic imports: '+syms)
 results[mode]={'object_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'undefined_symbols':syms.splitlines()}
 cmd=common+flags+['-S',str(root/rel),'-o',str(O/(mode+'.s'))];commands.append(cmd)
 with (O/(mode+'-asm.log')).open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
assert results['baseline']['object_sha256']==results['off']['object_sha256']
assembly=(O/'on.s').read_text();assert 'memw_locked' in assembly
results['off_object_identical']=True;results['inline_atomic_memw_locked']=True;results['no_atomic_runtime_imports']=True
(O/'commands.json').write_text(json.dumps(commands,indent=2)+'\n');(O/'results.json').write_text(json.dumps(results,indent=2)+'\n')
print('PASS: three full v81 MoE production builds; OFF object byte-identical; inline atomic32 memw_locked; no atomic runtime imports')
