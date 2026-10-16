#!/usr/bin/env python3
import importlib.util,json,subprocess
from pathlib import Path
D=Path(__file__).resolve().parent; A=D.parent; O=D/'host'; O.mkdir(exist_ok=True)
script=A/'frozen/candidate/test/htp/host/run_tl2_gu_epilogue_checks.py'
spec=importlib.util.spec_from_file_location('fixture',script);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
s=(D/'candidate'/module.MOE).read_text()
start=s.index('#if defined(NNTR_LFM2_TL2_GU_TRACE)');end=s.index('#endif',start)+len('#endif')
(O/'counter.inc').write_text(s[start:end]+'\n')
(O/'candidate-unit.inc').write_text(module.extract(s,'candidate'))
base=A/'host-san'
raw=json.loads((base/'commands.json').read_text());flags=raw[0][1:raw[0].index('-c')]
flags=[str(O) if x==str(base) else x for x in flags]
flags += ['-DNNTR_LFM2_TL2_GU_TRACE=1','-pthread']
commands=[['gcc',*flags,'-c',str(D/'counter_check.c'),'-o',str(O/'counter-check.o')],['g++','-fsanitize=address,undefined','-no-pie','-pthread','-Wl,--gc-sections',str(O/'counter-check.o'),*[str(base/(n+'.o')) for n in ('hvx_dequant_i32','hvx_quant_u8','hvx_gemv_u8tl2','hvx_rmsnorm_rows_f32')],'/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.4.0.1/tools/HEXAGON_Tools/19.0.04/Tools/libnative/lib/libnative.a','-lm','-o',str(O/'counter-check')],[str(O/'counter-check')]]
(O/'counter-commands.json').write_text(json.dumps(commands,indent=2)+'\n')
for label,cmd in zip(('counter-build','counter-link','counter-run'),commands):
 with (O/(label+'.log')).open('w') as log: subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
print((O/'counter-run.log').read_text(),end='')
