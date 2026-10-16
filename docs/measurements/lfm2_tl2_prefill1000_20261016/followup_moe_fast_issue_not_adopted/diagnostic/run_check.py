#!/usr/bin/env python3
"""Compile actual counter locations and validate emitted counts/output bytes."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--checks', type=Path, required=True, help='Frozen whole-caller output')
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
here = Path(__file__).parent
spec = importlib.util.spec_from_file_location('builder', here / 'make.py')
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)
spec = importlib.util.spec_from_file_location('counter_parser', here / 'parse.py')
parser = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parser)
a.output.mkdir(parents=True, exist_ok=True)
(a.output / 'hmx').mkdir(exist_ok=True)
(a.output / 'include').mkdir(exist_ok=True)
code = build.instrument((a.checks / 'candidate/hmx/kernel.c').read_text())
source = a.output / 'hmx/kernel.c'
source.write_text(code)
shutil.copy2(a.checks / 'candidate/hmx/hexkl_hmx_mm.h', source.parent)
shutil.copy2(a.checks / 'candidate/htp_tl2_opts.h', a.output)
(a.output / 'include/HAP_farf.h').write_text('''#ifndef HAP_FARF_H
#define HAP_FARF_H
#include <stdio.h>
#define FARF(level, ...) do { printf(__VA_ARGS__); putchar('\\n'); } while (0)
#endif
''')
# Production qtimer on device; a monotonic stand-in gives unique host timestamps.
(a.output / 'include/HAP_perf.h').write_text('''#ifndef HAP_PERF_H
#define HAP_PERF_H
#include <stdint.h>
static uint64_t ticks;
static inline uint64_t HAP_perf_get_qtimer_count(void) { return ++ticks; }
static inline uint64_t HAP_perf_qtimer_count_to_us(uint64_t c) { return c; }
static inline uint64_t HAP_perf_get_pcycles(void) { return ++ticks; }
#endif
''')
commands = json.loads((a.checks / 'commands.json').read_text())
original = str(a.checks / 'candidate/hmx/kernel.c')
cc = next(c for c in commands if '-c' in c and original in c)
obj = a.output / 'kernel.o'
cc = [str(source) if x == original else str(obj)
      if x == str(a.checks / 'candidate/kernel.o') else x for x in cc]
cc[1:1] = ['-I', str(a.output / 'include'), '-DNNTR_LFM2_MOE_ISSUE_TRACE=1']
link = next(c for c in commands if str(a.checks / 'candidate/check') in c and '-o' in c)
binary = a.output / 'check'
link = [str(obj) if x == str(a.checks / 'candidate/kernel.o') else str(binary)
        if x == str(a.checks / 'candidate/check') else x for x in link]
(a.output / 'commands.json').write_text(json.dumps([cc, link], indent=2) + '\n')
env = os.environ.copy()
env.update(ASAN_OPTIONS='detect_leaks=1:halt_on_error=1', UBSAN_OPTIONS='halt_on_error=1')
for label, command in (('compile', cc), ('link', link),
                       ('run', [str(binary), str(a.output / 'output.bin')])):
    with (a.output / (label + '.log')).open('w') as log:
        rc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
    if rc:
        raise RuntimeError(f'{label} failed {rc}: {a.output}')
log = (a.output / 'run.log').read_text()
records = [r for line in log.splitlines() if (r := parser.packet(line)) is not None]
assert len(records) == 6, records
expected = [(2, 64, 96, 64, 1), (65, 64, 384, 256, 1),
            (2, 1, 96, 64, 1), (2, 4, 192, 128, 1), (2, 60, 192, 128, 1)]
for index, (row, (M, E, gu, dn, selected)) in enumerate(zip(records, expected), 1):
    assert (row['seq'], row['M'], row['E'], row['selected'], row['gu'], row['dn'],
            row['fast_gu'], row['fast_dn'], row['rc']) == (index, M, E, selected, gu, dn, gu, dn, 0), row
assert records[5]['seq'] == 6 and records[5]['rc'] != 0 and records[5]['gu'] == 37
assert records[5]['gu'] == records[5]['fast_gu'] and not records[5]['dn']
assert (a.output / 'output.bin').read_bytes() == (a.checks / 'candidate/output.bin').read_bytes()
valid_lines = [line for line in log.splitlines() if parser.packet(line) and parser.packet(line)['seq'] <= 5]
valid = parser.parse(valid_lines, 2, 4, expect_load_warmup=False)
try:
    parser.parse(log.splitlines(), 2, 4, expect_load_warmup=False)
except ValueError:
    pass
else:
    raise AssertionError('Error records accepted')
fmt = re.search(r'FARF\(ERROR,\s*("\[LFM-MISSUE\].*?!")', code).group(1)
maximum = a.output / 'format.c'
maximum.write_text('''#include <assert.h>
#include <stdint.h>
#include <limits.h>
#include <stdio.h>
int main(void) {
  char line[1024];
  int n = snprintf(line, sizeof line, ''' + fmt + ''',
    UINT32_MAX, (unsigned long long)UINT64_MAX, UINT32_MAX, UINT32_MAX,
    INT_MIN, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, INT_MIN);
  assert(n > 0 && n + 64 + 1 <= 256);
  printf("payload=%d prefix=64 nul=1 total=%d\\n%s\\n", n, n + 65, line);
}
''')
cmd = ['gcc', '-std=c11', '-Wall', '-Wextra', '-Werror', '-fsanitize=address,undefined',
       '-fno-sanitize-recover=all', str(maximum), '-o', str(a.output / 'format')]
subprocess.run(cmd, check=True)
with (a.output / 'format.log').open('w') as out:
    subprocess.run([str(a.output / 'format')], stdout=out, env=env, check=True)
result = {'pass': True, 'actual_whole_calls': 11, 'actual_scope_records': 6,
          'successful_records': 5, 'error_records_rejected': 1, 'bytes_exact': 688128,
          'counter_records': records, 'parser_actual_subset': valid,
          'format': (a.output / 'format.log').read_text()}
(a.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
print('MOE_ISSUE_TRACE actual counts 6 records / 11 calls, bytes, errors, FARF length PASS')
