#!/usr/bin/env python3
import hashlib,json,pathlib,subprocess
T=pathlib.Path(__file__).resolve().parent; W=T/'worktree'; sdk=pathlib.Path('/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.4.0.1'); ht=sdk/'tools/HEXAGON_Tools/19.0.04/Tools/bin'; b=W/'nntrainer/tensor/htp_backend'; base=pathlib.Path('/home/jungone-lee/workspace/nntrainer-lfm2-tl2-prefill1000')
args=[str(ht/'hexagon-clang'),'-mv81','-mhvx','-mhvx-length=128B','-mhmx','-G0','-O3','-fPIC','-Wall','-Werror','-fstack-usage']
for inc in [W/'test/htp',base/'test/htp/generated',b.parent,b,b/'hvx',b/'hmx',pathlib.Path('/home/jungone-lee/workspace/nntrainer/Qualcomm/hexkl-1.0-beta.2/hexkl_addon/include'),sdk/'rtos/qurt/computev81/include/qurt',sdk/'rtos/qurt/computev81/include/posix']:args+=['-I',str(inc)]
for inc in [sdk/'incs',sdk/'incs/stddef',sdk/'ipc/fastrpc/incs']:args+=['-isystem',str(inc)]
manifest={}
for flag in [0,1]:
 out=T/f'nntr_hvx_mm_u8i4-trace{flag}.o'
 cmd=args+[f'-DNNTR_LFM2_MOE_STAGE_TRACE={flag}','-c',str(W/'test/htp/nntr_hvx_mm_u8i4.c'),'-o',str(out)]
 subprocess.run(cmd,check=True); manifest[str(flag)]={'command':cmd,'sha256':hashlib.sha256(out.read_bytes()).hexdigest()}
 print(f'PASS: v81 full TU -O3 -Wall -Werror diagnostic={flag}')
(T/'compile.json').write_text(json.dumps(manifest,indent=2)+'\n')
