#!/usr/bin/env python3
"""Prepare/build a diagnostic-only reset-hit counter on a frozen object pair."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--build-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--fixture', type=Path, required=True)
    a = ap.parse_args()
    B, D = a.build_root.resolve(), a.output.resolve()
    if D.exists():
        raise ValueError('Refuse existing diagnostic output')
    D.mkdir(parents=True)
    V = json.loads((B / 'validation.json').read_text())
    rel = 'nntrainer/tensor/htp_backend/hmx/hexkl_dma_ring.c'
    source = B / 'source' / rel
    before = source.read_text()
    assert sha(source) == V['production_source_sha256'][rel]
    header = '''
#if defined(NNTR_LFM2_M1_RING_RESET_TRACE)
#include <HAP_farf.h>
static uint32_t g_ring_reset_trace_seq;
#define RING_RESET_TRACE(used)                                                  \\
  FARF(ERROR, "nntr-ring-reset q=%u used=%u !",                                  \\
       __atomic_add_fetch(&g_ring_reset_trace_seq, 1u, __ATOMIC_RELAXED), (used))
#else
#define RING_RESET_TRACE(used) ((void)0)
#endif
'''
    after = before.replace('#include <string.h>\n', '#include <string.h>\n' + header)
    anchor = '      g_tail->desc_type != 9u || !hexkl_dma_ring_is_done(last)) {\n    return 0;'
    assert after.count(anchor) == 1
    after = after.replace(anchor, anchor.replace('    return 0;', '    RING_RESET_TRACE(0u);\n    return 0;'))
    anchor = '  g_tail = NULL;\n  g_started = 0;\n  return 1;'
    assert after.count(anchor) == 1
    after = after.replace(anchor, anchor.replace('  return 1;', '  RING_RESET_TRACE(1u);\n  return 1;'))
    patch = ''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile='a/' + rel, tofile='b/' + rel))
    (D / 'diagnostic.patch').write_text(patch)
    (D / 'ring-diagnostic.c').write_text(after)
    records = json.loads((B / 'compilations.json').read_text())
    record = next(r for name, r in records.items() if name.startswith('candidate-') and r['source'].endswith('hexkl_dma_ring.c'))
    normal_obj = Path(record['object'])
    commands = []
    work = B / 'source/test/htp'

    def run(command, label):
        result = subprocess.run(command, cwd=work, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (D / (label + '.log')).write_bytes(result.stdout)
        commands.append({'label': label, 'cwd': str(work), 'argv': [str(x) for x in command], 'returncode': result.returncode})
        save(D / 'commands.json', commands)
        if result.returncode:
            raise ValueError(label + ' failed')
        return result.stdout.decode(errors='replace')

    try:
        source.write_text(after)
        for enabled in (0, 1):
            cmd = record['cc1'].copy()
            obj = D / ('ring-' + str(enabled) + '.o')
            for flag, val in (('-o', obj), ('-dependency-file', obj.with_suffix('.d')), ('-MT', obj)):
                cmd[cmd.index(flag) + 1] = str(val)
            if enabled:
                cmd += ['-DNNTR_LFM2_M1_RING_RESET_TRACE=1']
            run(cmd, 'compile-' + str(enabled))
        assert sha(D / 'ring-0.o') == sha(normal_obj), 'Diagnostic OFF object differs'
        cmd = [str(D / 'ring-1.o') if x == str(normal_obj) else x for x in V['candidate']['link_command']]
        cmd[cmd.index('-o') + 1] = str(D / 'libnntr_hvx_skel.so')
        run(cmd, 'link-1')
        first = sha(D / 'libnntr_hvx_skel.so')
        run(cmd, 'link-2')
        assert sha(D / 'libnntr_hvx_skel.so') == first
    finally:
        source.write_text(before)
    assert sha(source) == V['production_source_sha256'][rel]
    for obj, expected in json.loads((B / 'frozen-object-hashes.json').read_text()).items():
        assert sha(Path(obj)) == expected
    assert sha(normal_obj) == record['object_sha256']
    compiler = Path(record['cc1'][0]).parent
    symbols = run([str(compiler / 'hexagon-nm'), '-u', str(D / 'ring-1.o')], 'object-imports')
    assert not any(word in symbols for word in ('__atomic', '__sync'))
    elf = run([str(compiler / 'hexagon-readelf'), '-h', str(D / 'libnntr_hvx_skel.so')], 'elf')
    assert 'Flags:                             0x81' in elf
    dyn = run([str(compiler / 'hexagon-readelf'), '--dyn-syms', str(D / 'libnntr_hvx_skel.so')], 'imports')
    imports = []
    for line in dyn.splitlines():
        f = line.split()
        if len(f) >= 8 and f[6] == 'UND':
            imports.append({'symbol': f[7], 'binding': f[4]})
    assert imports == V['candidate']['imports'], 'Diagnostic runtime imports changed'

    # Actual ring TU and existing delayed-DMA fixture, with a host FARF sink.
    H = D / 'host';H.mkdir()
    for name in ('hexagon_protos.h', 'hexagon_types.h', 'hmx_hexagon_protos.h'):
        (H / name).write_text('/* Hardware primitives supplied by the existing fixture. */\n')
    (H / 'HAP_farf.h').write_text('#define FARF(level, format, seq, used) fixture_trace(format, seq, used)\n')
    (H / 'hexkl_dma_ring.h').write_bytes(source.with_suffix('.h').read_bytes())
    (H / 'ring-under-test.c').write_text(after)
    fixture = a.fixture.read_text()
    sink = '''
static uint32_t observed_seq, observed_hit;
static void fixture_trace(const char *format, uint32_t seq, uint32_t used) {
  char line[96];
  int count = snprintf(line, sizeof(line), format, seq, used);
  assert(count > 0 && count < 64);
  assert(seq == observed_seq + 1u && used <= 1u);
  observed_seq = seq;
  observed_hit = used;
  puts(line);
}
'''
    fixture = fixture.replace('#define Q6_dccleaninva_A(p)', sink + '\n#define Q6_dccleaninva_A(p)')
    fixture = fixture.replace('  assert(hexkl_dma_ring_try_reset() == wanted);',
                              '  const uint32_t old_seq = observed_seq;\n  assert(hexkl_dma_ring_try_reset() == wanted);\n  assert(observed_seq == old_seq + 1u && observed_hit == (uint32_t)wanted);')
    fixture = fixture.replace('  assert(hexkl_dma_ring_try_reset() == 1);',
                              '  const uint32_t old_seq = observed_seq;\n  assert(hexkl_dma_ring_try_reset() == 1);\n  assert(observed_seq == old_seq + 1u && observed_hit == 1u);')
    (H / 'fixture.c').write_text(fixture)
    cmd = ['gcc', '-std=gnu11', '-O1', '-g', '-Wall', '-Wextra', '-Werror', '-fsanitize=address,undefined',
           '-fno-sanitize-recover=undefined', '-fno-omit-frame-pointer', '-fno-pie', '-no-pie',
           '-DNNTR_LFM2_M1_RING_RESET_TRACE=1', '-I', str(H), str(H / 'fixture.c'), '-o', str(H / 'check')]
    run(cmd, 'host-compile')
    output = run([str(H / 'check')], 'host-run')
    assert 'actual ring: 175 lifecycle/reset checks,' in output and output.rstrip().endswith('PASS')
    save(D / 'validation.json', {'diagnostic_patch_sha256': sha(D / 'diagnostic.patch'),
                                'production_patch_sha256': V['source_patch_sha256'],
                                'diagnostic_source_sha256': sha(D / 'ring-diagnostic.c'),
                                'diagnostic_so_sha256': first,
                                'diagnostic_off_object_exact': True,
                                'candidate_normal_object_sha256': sha(normal_obj),
                                'diagnostic_object_sha256': sha(D / 'ring-1.o'),
                                'runtime_imports_identical': True, 'atomic_library_imports': False,
                                'stable_link_repeat': True, 'arch': 'v81',
                                'normal_source_and_objects_unchanged': True,
                                'host_actual_ring_cases': 175, 'host_sanitizers': ['address', 'undefined'],
                                'metadata_bytes': 4,
                                'scope': 'One diagnostic log per actual try_reset return; used=1 proves descriptor clear skipped. Counts attempts, not successful later MoE RPCs; no device or timing proof.'})
    print('Diagnostic build and actual-ring host trace checks PASS')


if __name__ == '__main__':
    main()
