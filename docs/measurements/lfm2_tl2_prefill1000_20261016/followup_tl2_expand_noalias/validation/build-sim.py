from pathlib import Path
import subprocess, json, os
A=Path(__file__).resolve().parent
T=Path('/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.4.0.1/tools/HEXAGON_Tools/19.0.04/Tools')
out=A/'checks/sim';out.mkdir(parents=True,exist_ok=True)
exports=('hvx_tl2_weight_bytes','hvx_tl2_table_bytes','hvx_tl2_k_splits','hvx_tl2_pack','hvx_tl2_unpack','hvx_tl2_expand_wh96','hvx_tl2_prepare','hvx_gemv_u8tl2_panel')
flags=['-mv81','-mhvx','-mhvx-length=128B','-O3','-Wall','-Werror','-I',str(A/'candidate/nntrainer/tensor/htp_backend/hvx')]
commands=[]
def run(cmd,label,env=None):
 commands.append(cmd);(out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
 with (out/(label+'.log')).open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env)
 if r.returncode:raise SystemExit(f'{label} {r.returncode}: {out/(label+".log")}')
for kind in ['baseline','candidate']:
 run([str(T/'bin/hexagon-clang'),*flags,*(['-D'+n+'=reference_'+n for n in exports] if kind=='baseline' else []),'-c',str(A/kind/'nntrainer/tensor/htp_backend/hvx/hvx_gemv_u8tl2.c'),'-o',str(out/(kind+'.o'))],'build-'+kind)
run([str(T/'bin/hexagon-clang'),*flags,str(A/'bench.c'),str(out/'baseline.o'),str(out/'candidate.o'),'-lhexagon','-lm','-o',str(out/'check')],'link')
env=os.environ.copy();env['LD_LIBRARY_PATH']='/tmp/nntr-lfm2-decode-opt-20261016/dn-dequant/compat/lib/x86_64-linux-gnu'+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
run([str(T/'bin/hexagon-sim'),'-mv81','--timing',str(out/'check')],'run',env)
print((out/'run.log').read_text())
