#!/usr/bin/env python3
"""Compare the existing scalar whole-FC fixture with trace off/on; no device."""
import argparse
import json
import os
import subprocess
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--repo', required=True, type=Path)
p.add_argument('--source', required=True, type=Path)
p.add_argument('--reference', required=True, type=Path)
p.add_argument('--output', required=True, type=Path)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
stub = a.output / 'stub';stub.mkdir(exist_ok=True)
# Preserve argument evaluation/format checking; the original fixture is outside
# the trace guard. Do not discard FARF arguments and accidentally hide warnings.
(stub / 'HAP_farf.h').write_text('#ifndef HAP_FARF_H\n#define HAP_FARF_H\n#include <stdio.h>\n#define FARF(level, ...) do { printf(__VA_ARGS__); putchar(10); } while (0)\n#endif\n')
h = a.repo / 'test/htp/host';b = a.repo / 'nntrainer/tensor/htp_backend'
# The upstream fixture intentionally leaves its three main arrays alive until
# process exit. Free only those test-owned arrays in this throwaway copy so LSan
# can remain enabled; do not change production or any assertion/arithmetic.
fixture = a.output / 'fc_layer_host_check.c'
original = (h / 'fc_layer_host_check.c').read_text()
assert original.count('  return fail;\n}') == 1
fixture.write_text(original.replace('  return fail;\n}', '  free(x);\n  free(got);\n  free(want);\n  return fail;\n}'))
commands = []
env = dict(os.environ, ASAN_OPTIONS='detect_leaks=1:halt_on_error=1', UBSAN_OPTIONS='halt_on_error=1:print_stacktrace=1')
for name, src, flags in [('before', a.reference, []), ('enabled', a.source, ['-DNNTR_LFM2_AFFINE_FC_TRACE=1'])]:
    cmd = ['gcc', '-std=c99', '-O1', '-g', '-Wall', '-Wextra', '-Wno-unused-parameter',
           '-fsanitize=address,undefined', '-fno-omit-frame-pointer', *flags]
    for inc in [stub, h, h/'stub', h/'standin', b.parent, b/'hmx', b/'hvx']:
        cmd += ['-I', str(inc)]
    cmd += [str(fixture), str(h/'hvx_scalar_stubs.c'), str(h/'standin/hvx_scalar.c'), str(src), '-lm', '-o', str(a.output/name)]
    commands.append(cmd)
    for label, argv in [('compile', cmd), ('run', [str(a.output/name)])]:
        r = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        (a.output/(name+'-'+label+'.log')).write_text(r.stdout)
        if r.returncode:
            (a.output/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
            raise SystemExit(name+' '+label+' failed: '+str(r.returncode)+'\n'+r.stdout[-3000:])
assert (a.output/'before-run.log').read_bytes() == (a.output/'enabled-run.log').read_bytes(), 'whole-FC fixture output differs'
(a.output/'commands.json').write_text(json.dumps(commands, indent=2)+'\n')
print('whole-FC scalar ASan/UBSan before/on PASS; stdout byte-identical; trace guard excludes these K64 cases')
