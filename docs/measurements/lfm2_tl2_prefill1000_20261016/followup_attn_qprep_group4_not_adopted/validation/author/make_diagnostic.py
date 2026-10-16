#!/usr/bin/env python3
import difflib,hashlib,json,subprocess
from pathlib import Path
D=Path(__file__).resolve().parent
r='nntrainer/tensor/htp_backend/hvx/hvx_quant_u8.c'
s=(D/'candidate'/r).read_text(); diag=D/'diagnostic';p=diag/'source'/r;p.parent.mkdir(parents=True,exist_ok=True)
macro='NNTR_LFM2_QPREP_GROUP4_TRACE'
s=s.replace('#include "hvx_quant_u8_group4.h"','#include "hvx_quant_u8_group4.h"\n#if defined('+macro+') && '+macro+'\n#include <HAP_farf.h>\n/* Diagnostic only: one successful synchronous helper call per record. */\nstatic uint32_t lfm2_qprep_group4_seq;\n#endif',1)
s=s.replace('  hvx_worker_pool_run(pool, quant_group4_worker, &c, groups);\n  return AEE_SUCCESS;', '''  hvx_worker_pool_run(pool, quant_group4_worker, &c, groups);
#if defined(NNTR_LFM2_QPREP_GROUP4_TRACE) && NNTR_LFM2_QPREP_GROUP4_TRACE
  const uint32_t seq =
    __atomic_add_fetch(&lfm2_qprep_group4_seq, 1u, __ATOMIC_RELAXED);
  FARF(ERROR,
       "[nntr-qprep-group4] seq=%u groups=%u k=%u stride=%u bytes=%u !",
       (unsigned)seq, (unsigned)groups, (unsigned)k, (unsigned)group_stride,
       (unsigned)(groups * 4u * k * sizeof(float)));
#endif
  return AEE_SUCCESS;''',1)
p.write_text(s)
old=(D/'candidate'/r).read_text();patch='diff --git a/'+r+' b/'+r+'\n'+''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='a/'+r,tofile='b/'+r,n=3))
(diag/'diagnostic.patch').write_text(patch)
plans=json.loads((D/'codegen/commands.json').read_text())
plan=next(v for v in plans if v['side']=='candidate' and v['source']==r and v['mode']=='object')
records=[]
for mode in ['off','on']:
 out=diag/'build'/mode;out.mkdir(parents=True,exist_ok=True)
 cmd=plan['command'][:]
 for arg in ['-o','-dependency-file','-MT','-stack-usage-file']:
  while arg in cmd:
   i=cmd.index(arg);del cmd[i:i+2]
 cmd=[str(p) if x==str(D/'candidate'/r) else x for x in cmd]
 if mode=='on':cmd+=['-D'+macro+'=1']
 cmd+=['-o',str(out/'hvx_quant_u8.o'),'-stack-usage-file',str(out/'hvx_quant_u8.su')]
 q=subprocess.run(cmd,cwd=plan['cwd'],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
 (out/'compile.log').write_text(q.stdout);assert q.returncode==0,q.stdout
 (out/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
 nm=Path(cmd[0]).parent/'llvm-nm'
 imports=subprocess.check_output([str(nm),'--undefined-only',str(out/'hvx_quant_u8.o')],text=True)
 (out/'imports.txt').write_text(imports)
 assert 'atomic' not in imports and 'sync_' not in imports
 records.append({'mode':mode,'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'object_sha256':hashlib.sha256((out/'hvx_quant_u8.o').read_bytes()).hexdigest(),'imports':imports})
normal=D/'codegen/candidate/hvx_quant_u8.o'
assert (diag/'build/off/hvx_quant_u8.o').read_bytes()==normal.read_bytes(), 'macro-off object differs'
report={'macro':macro,'production_patch_sha256':hashlib.sha256((D/'production.patch').read_bytes()).hexdigest(),'diagnostic_patch_sha256':hashlib.sha256((diag/'diagnostic.patch').read_bytes()).hexdigest(),'macro_off_object_byte_exact':True,'extra_global_bytes_on':4,'records':records,'scope':'Successful group4 helper calls after synchronous worker drain; not proof of later HMX/RPC success; require normal IDs and completed run separately.'}
(diag/'build-validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
