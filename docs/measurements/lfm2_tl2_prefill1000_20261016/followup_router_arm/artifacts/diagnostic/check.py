#!/usr/bin/env python3
"""Actual diagnostic helper/caller counts, including joined error fallback.

First run the production host checker with --sanitize, then supply its output.
The actual arithmetic kernel and HVX objects are unchanged and reused here.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import parse as parser

p = argparse.ArgumentParser()
p.add_argument('--checked-output', type=Path, required=True)
p.add_argument('--production-tests', type=Path, required=True)
p.add_argument('--source', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
base, out = a.checked_output.resolve(), a.output.resolve()
out.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location('extractor', a.production_tests / 'run_lfm2_router_arm_native.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
source_path = a.source / 'Applications/CausalLM/models/lfm2_moe/lfm2_moe_layer.cpp'
source = source_path.read_text()
parts = module.extracts(source)
first = source.index('#if defined(NNTR_LFM2_ROUTER_ARM_TRACE)')
last = source.index('#if defined(__aarch64__) && defined(ENABLE_HEXKL)\n/** Use the existing', first)
parts['helper.inc'] = source[first:last] + parts['helper.inc']
for name, value in parts.items():
    (out / name).write_text(value)
for name in ('arm_neon.h', 'lfm2_router_arm_data.h'):
    shutil.copy2(base / name, out / name)
fixture = (a.production_tests / 'lfm2_router_arm_check.cpp').read_text()
fixture = fixture.replace('int main() {', 'int unused_production_suite() {')
fixture = fixture.replace('  AcceleratedRoute route;\n', '  const bool m0_profile = true;\n  AcceleratedRoute route;\n')
fixture += r'''
int main() {
  try {
    unsigned count = 0;
    for (unsigned rows : {1024u, 2048u})
      for (unsigned kind = 0; kind < 6; ++kind) {
        Data d(rows);
        const unsigned mode = kind == 1 ? 36 : kind == 2 ? 37 :
                              kind == 3 || kind == 5 ? 38 : 0;
        d.reset(mode);
        nntrainer::ThreadManager::lanes = 8;
        nntrainer::ThreadManager::order = 2;
        if (kind == 4)
          d.context.weights[127].type = nntrainer::Tdatatype::QS4CX_WH;
        nntrainer::htp.fail = kind == 5;
        bool failed = false;
        try {
          const bool used = d.caller();
          check(used == (kind == 0 && enabled()), "diagnostic route changed");
        } catch (const std::runtime_error &) {
          failed = true;
        }
        check(failed == (kind == 5), "diagnostic exception changed");
        check(nntrainer::htp.calls == unsigned(!(kind == 0 && enabled())),
              "diagnostic RPC count changed");
        ++count;
      }
    std::printf("ROUTER_ARM_TRACE %u actual caller cases PASS\n", count);
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "ROUTER_ARM_TRACE FAIL %s\n", e.what());
    return 1;
  }
}
'''
(out / 'lfm2_router_arm_check.cpp').write_text(fixture)
link = json.loads((base / 'commands.json').read_text())['commands'][-1]
cmd = [v if v.endswith('.o') else v.replace(str(base), str(out)) for v in link]
cmd.insert(1, '-DNNTR_LFM2_ROUTER_ARM_TRACE=1')
(out / 'command.json').write_text(json.dumps({'argv': cmd, 'source_sha256': hashlib.sha256(source.encode()).hexdigest()}, indent=2) + '\n')
with (out / 'build.log').open('w') as log:
    subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
results = []
for enabled in (0, 1):
    env = os.environ.copy()
    env.pop('NNTR_HTP_ROUTER_U8I8', None)
    env.update(NNTR_LFM2_ROUTER_ARM=str(enabled), ASAN_OPTIONS='detect_leaks=1:halt_on_error=1', UBSAN_OPTIONS='halt_on_error=1')
    r = subprocess.run([str(out / 'check')], env=env, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out / f'run-{enabled}.log').write_text(r.stdout)
    assert r.returncode == 0, r.stdout
    records = parser.packets(r.stdout)
    assert len(records) == 12
    for i, row in enumerate(records):
        kind = i % 6
        rows = 1024 if i < 6 else 2048
        tasks = rows // 4 if enabled and kind not in (2, 4) else 0
        used = int(enabled and kind == 0)
        expected = dict(zip(parser.KEYS, (i+1, rows, 2048, 64, 1, used,
                       int(kind != 5 and not used), tasks, tasks,
                       int(enabled and kind in (1, 3, 5)), int(kind != 5))))
        assert row == expected, (row, expected)
    results.append({'enabled': enabled, 'actual_calls': 12, 'pass': True})
(out / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
print('Actual diagnostic caller: 12 ON + 12 OFF ASan/UBSan PASS')
