#!/usr/bin/env python3
import argparse,json,os,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
stub=a.output/'stub';stub.mkdir(exist_ok=True)
(stub/'HAP_perf.h').write_text('#ifndef HAP_PERF_H\n#define HAP_PERF_H\n#include <stdint.h>\nstatic uint64_t lfm_test_clock;\nstatic uint64_t lfm_test_step = 1;\nstatic inline uint64_t HAP_perf_get_qtimer_count(void) {return lfm_test_clock += lfm_test_step;}\nstatic inline uint64_t HAP_perf_qtimer_count_to_us(uint64_t t) {return t;}\n#endif\n')
(stub/'HAP_farf.h').write_text('#ifndef HAP_FARF_H\n#define HAP_FARF_H\nvoid lfm_test_log(const char *, ...);\n#define FARF(level, ...) lfm_test_log(__VA_ARGS__)\n#endif\n')
h=a.repo/'test/htp/host';b=a.repo/'nntrainer/tensor/htp_backend'
cmd=['gcc','-std=c99','-O1','-g','-Wall','-Wextra','-Werror','-Wno-unused-parameter','-ffunction-sections','-fdata-sections','-fsanitize=address,undefined','-fno-omit-frame-pointer','-DNNTR_LFM2_AFFINE_FC_TRACE=1','-DNNTR_TEST_SOURCE="'+str(a.source)+'"']
for inc in [stub,h/'stub',h/'standin',b.parent,b/'hmx',b/'hvx']:cmd+=['-I',str(inc)]
cmd += [str(Path(__file__).with_name('guard_check.c')),'-Wl,--gc-sections','-lm','-o',str(a.output/'guard')]
(a.output/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
for tag,argv in [('compile',cmd),('run',[str(a.output/'guard')])]:
 r=subprocess.run(argv,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=dict(os.environ,ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1'));(a.output/(tag+'.log')).write_text(r.stdout)
 if r.returncode:raise SystemExit(r.stdout)
print('actual-helper25 guard cases + actual full-width printf bounds ASan/UBSan PASS')
