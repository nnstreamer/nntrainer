#!/usr/bin/env python3
"""Produce an isolated actual MoE leaf dispatch counter (no timing claims)."""
import argparse
import difflib
from pathlib import Path


def instrument(s):
    marker = '#include "hvx_scale_add_f32.h"\n'
    assert s.count(marker) == 1
    s = s.replace(marker, marker + '''
#ifndef NNTR_LFM2_MOE_ISSUE_TRACE
#define NNTR_LFM2_MOE_ISSUE_TRACE 0
#endif
#if NNTR_LFM2_MOE_ISSUE_TRACE
#include <HAP_farf.h>
static _Atomic uint32_t issue_trace_next = 0;
#endif
''', 1)
    pos = s.index('int hexkl_mm_u8i4_moe_layer_run(')
    brace = s.index('{', pos)
    s = s[:brace + 1] + '''
#if NNTR_LFM2_MOE_ISSUE_TRACE
  int issue_trace_scope = 0;
  uint32_t issue_trace_seq = 0;
  uint64_t issue_trace_id = 0;
  uint32_t issue_total_gu = 0, issue_total_dn = 0;
  uint32_t issue_fast_gu = 0, issue_fast_dn = 0;
  int issue_selected = -1;
#endif
''' + s[brace + 1:]
    pos = s.index('  const int use_m1 =')
    end = s.index(';', pos) + 1
    s = s[:end] + '''
#if NNTR_LFM2_MOE_ISSUE_TRACE
  issue_trace_scope = M > 1u && K == 2048u && inter == 1536u &&
                      N_out == 2048u && n_experts > 0u && n_experts <= 64u &&
                      n_with_rows && n_tl2 == n_with_rows && !n_affine &&
                      act == HVX_GLU_SILU;
  if (issue_trace_scope) {
    issue_trace_seq = atomic_fetch_add_explicit(&issue_trace_next, 1u,
                                                memory_order_relaxed) + 1u;
    issue_trace_id = hexkl_probe_now_ticks();
  }
#endif
''' + s[end:]
    pos = s.index('  const int lfm_hmx_leaf =')
    end = s.index(';', pos) + 1
    s = s[:end] + '''
#if NNTR_LFM2_MOE_ISSUE_TRACE
  issue_selected = lfm_hmx_leaf;
#endif
''' + s[end:]
    for phase, ptr in (('gu', 'L.act_off'), ('dn', 'L.mid_off + pp * mid_bytes')):
        marker = '          if (hmx_inline) {\n            const uint8_t *a = vtcm_base + ' + ptr + ';'
        assert s.count(marker) == 1
        s = s.replace(marker, '#if NNTR_LFM2_MOE_ISSUE_TRACE\n          ++issue_total_' + phase + ';\n#endif\n' + marker, 1)
    pos = 0
    for phase in ('gu', 'dn'):
        marker = '            if (lfm_hmx_leaf) {'
        pos = s.index(marker, pos) + len(marker)
        extra = '\n#if NNTR_LFM2_MOE_ISSUE_TRACE\n              ++issue_fast_' + phase + ';\n#endif'
        s = s[:pos] + extra + s[pos:]
        pos += len(extra)
    marker = '  /* Nothing to free: the scratch stays with the session. */\n  return rc;'
    assert s.count(marker) == 1
    s = s.replace(marker, '''#if NNTR_LFM2_MOE_ISSUE_TRACE
  if (issue_trace_scope) {
    FARF(ERROR,
         "[LFM-MISSUE] q=%u s=%llx d=%u,%u b=%d t=%u,%u f=%u,%u r=%d !",
         issue_trace_seq, (unsigned long long)issue_trace_id, M, n_experts,
         issue_selected,
         issue_total_gu, issue_total_dn, issue_fast_gu, issue_fast_dn, rc);
  }
#endif
''' + marker, 1)
    return s


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    rel = Path('nntrainer/tensor/htp_backend/hmx/hexkl_mm_u8i4_moe.c')
    original = (a.source / rel).read_text()
    changed = instrument(original)
    target = a.output / 'candidate' / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(changed)
    (a.output / 'diagnostic.patch').write_text(''.join(difflib.unified_diff(
        original.splitlines(True), changed.splitlines(True),
        fromfile='a/' + str(rel), tofile='b/' + str(rel))))
