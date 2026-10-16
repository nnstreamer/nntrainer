#!/usr/bin/env python3
import json,subprocess
from pathlib import Path
D=Path(__file__).resolve().parent
A=Path('/tmp/nntr-lfm2-tl2-expand-noalias-20261016')
O=D/'host';O.mkdir(exist_ok=True)
cmds=json.loads((A/'checks/hoisted-asan/commands.json').read_text())
raw=next(c for c in cmds if '-c' in c and any(x.endswith('/candidate/nntrainer/tensor/htp_backend/hvx/hvx_gemv_u8tl2.c') for x in c))
common=raw[:raw.index('-c')]
common=[str(D/'candidate'/Path(x).relative_to(A/'candidate')) if x.startswith(str(A/'candidate')) else x for x in common]
commands=[]
def run(cmd,label):
    commands.append(cmd)
    with (O/(label+'.log')).open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
    (O/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
run(common+['-DNNTR_LFM2_TL2_EXPAND_TRACE=1','-c',str(D/'candidate/nntrainer/tensor/htp_backend/hvx/hvx_gemv_u8tl2.c'),'-o',str(O/'actual.o')],'actual-build')
lib=Path('/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.4.0.1/tools/HEXAGON_Tools/19.0.04/Tools/libnative/lib/libnative.a')
run(['g++','-std=c++17','-O1','-g','-Wall','-Wextra','-Werror','-fsanitize=address,undefined','-fno-sanitize-recover=all','-fno-omit-frame-pointer','-no-pie','-DNNTR_LFM2_TL2_EXPAND_TRACE=1','-I',str(D/'candidate/nntrainer/tensor/htp_backend/hvx'),str(D/'counter_check.cpp'),str(O/'actual.o'),str(lib),'-pthread','-Wl,--gc-sections','-lm','-o',str(O/'counter-check')],'counter-build')
run([str(O/'counter-check')],'counter-run')
print((O/'counter-run.log').read_text(),end='')
