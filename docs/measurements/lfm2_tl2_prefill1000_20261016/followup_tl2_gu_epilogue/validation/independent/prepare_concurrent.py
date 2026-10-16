#!/usr/bin/env python3
"""Build an independent pthread-lane extension of the frozen GU fixture.

Only the fixture is extended; actual candidate/reference extraction is unchanged.
This tests byte parity with simultaneous lanes, not QuRT, DMA or timing.
"""
from pathlib import Path
import hashlib
import shutil

AUTHOR = Path('/tmp/nntr-lfm2-tl2-gu-epilogue-20261016')
OUT = Path(__file__).resolve().parent / 'concurrent-source'
CHECK = Path('test/htp/host/tl2_gu_epilogue_check.c')
RUN = CHECK.with_name('run_tl2_gu_epilogue_checks.py')
MOE = Path('nntrainer/tensor/htp_backend/hmx/hexkl_mm_u8i4_moe.c')
FROZEN = {
    MOE: 'a19d1ee954f294721d09d2eaf46286d0ee5f3682150938c22169b98f9e7013ea',
    CHECK: 'c8a9257989c468495846f2a906f1db3ed3d63a3cac84b1dd077a1a5733517ebf',
    RUN: 'c8895f0282be9f92bf8f3751ad1b030b65a1c9dc6a2963049cea059dfa655f54',
}
for path, expected in FROZEN.items():
    source = AUTHOR / 'candidate' / path
    assert hashlib.sha256(source.read_bytes()).hexdigest() == expected, path
    (OUT / path).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, OUT / path)

check = (OUT / CHECK).read_text()
check = check.replace('#include <stdint.h>',
                      '#include <stdint.h>\n#include <pthread.h>\n#include <stdatomic.h>')
check = check.replace('static unsigned reference_calls, candidate_calls, direct_calls;',
                      'static _Atomic unsigned reference_calls, candidate_calls, direct_calls;')
check = check.replace('int main(void) {', 'static int author_main(void) {', 1)
check += r'''

typedef struct {
  hvx_worker_pool_func fn;
  void *ctx;
  unsigned lanes, id;
  pthread_barrier_t *barrier;
} concurrent_lane;

static void *invoke_lane(void *raw) {
  concurrent_lane *lane = raw;
  int rc = pthread_barrier_wait(lane->barrier);
  assert(rc == 0 || rc == PTHREAD_BARRIER_SERIAL_THREAD);
  lane->fn(lane->lanes, lane->id, lane->ctx);
  return NULL;
}

static void invoke_parallel(hvx_worker_pool_func fn, void *ctx, unsigned lanes) {
  pthread_t threads[8];
  concurrent_lane jobs[8];
  pthread_barrier_t barrier;
  assert(lanes >= 1u && lanes <= 8u);
  assert(pthread_barrier_init(&barrier, NULL, lanes) == 0);
  for (unsigned i = 0; i < lanes; ++i) {
    jobs[i] = (concurrent_lane){fn, ctx, lanes, i, &barrier};
    assert(pthread_create(&threads[i], NULL, invoke_lane, &jobs[i]) == 0);
  }
  for (unsigned i = 0; i < lanes; ++i)
    assert(pthread_join(threads[i], NULL) == 0);
  assert(pthread_barrier_destroy(&barrier) == 0);
}

static void compare_parallel(unsigned lanes, int requant) {
  poison_padding(reference_ctx.inter, reference_ctx.tl2_splits, 1);
  invoke_parallel(requant ? reference_moe_m1_requant_worker
                           : reference_moe_tl2_gate_epilogue_worker,
                  &reference_ctx, lanes);
  invoke_parallel(requant ? candidate_moe_m1_requant_worker
                           : candidate_moe_tl2_gate_epilogue_worker,
                  &candidate_ctx, lanes);
  poison_padding(reference_ctx.inter, reference_ctx.tl2_splits, 0);
  same_outputs();
}

static void parallel_pipeline(unsigned lanes, unsigned native_count, int qs2,
                              int mag7, unsigned splits, int gemma) {
  unsigned inter = gemma ? 704u : INTER;
  init((native_count + lanes) % 6u, inter, splits, 1u, 1u);
  const unsigned active = native_count + (unsigned)qs2;
  if (qs2) {
    gu[native_count].tl2_bytes = dn[native_count].tl2_bytes = NULL;
    for (unsigned j = 0; j < inter; ++j)
      expected[native_count].gate[GUARD + j] =
        actual[native_count].gate[GUARD + j] =
          (float)((int)(j % 31u) - 15) / 8.0f;
  }
  if (mag7) {
    unsigned e = native_count > 1u ? 1u : 0u;
    gu[e].pal[0] = gu[e].pal[1] = -7;
    gu[e].pal[3] = 7;
  }
  if (gemma) {
    reference_ctx.k_tiles = candidate_ctx.k_tiles = 88u;
    reference_ctx.N_out = candidate_ctx.N_out = 2816u;
    reference_ctx.act = candidate_ctx.act = HVX_GLU_GELU_TANH;
  }
  reference_ctx.n_active = candidate_ctx.n_active = active;
  reference_ctx.tl2_fused = candidate_ctx.tl2_fused = native_count >= 4u;
  if (native_count < 4u) {
    for (unsigned e = 0; e < native_count; ++e) {
      reference_ctx.u_lo = candidate_ctx.u_lo = e * (inter / 32u);
      reference_ctx.u_hi = candidate_ctx.u_hi = (e + 1u) * (inter / 32u);
      compare_parallel(lanes, 0);
    }
  }
  compare_parallel(lanes, 1);
  const unsigned eligible = !gemma && splits == 1u
                              ? native_count - (unsigned)mag7 : 0u;
  assert(reference_calls == native_count * (inter / 32u));
  assert(candidate_calls == (native_count - eligible) * (inter / 32u));
  assert(direct_calls == eligible * (native_count >= 4u ? 1u : lanes));
}

int main(void) {
  assert(author_main() == 0);
  unsigned parallel_cases = 0;
  for (unsigned lanes = 1; lanes <= 8u; ++lanes) {
    for (unsigned native_count = 1; native_count <= 4u; ++native_count)
      for (unsigned qs2 = 0; qs2 <= 1u; ++qs2)
        for (unsigned mag7 = 0; mag7 <= 1u; ++mag7) {
          parallel_pipeline(lanes, native_count, qs2, mag7, 1u, 0);
          ++parallel_cases;
        }
    for (unsigned native_count = 3u; native_count <= 4u; ++native_count) {
      for (unsigned split = 2; split <= 4u; split *= 2u) {
        parallel_pipeline(lanes, native_count, 1, 0, split, 0);
        ++parallel_cases;
      }
      parallel_pipeline(lanes, native_count, 0, 0, 4u, 1);
      ++parallel_cases;
    }
  }
  assert(parallel_cases == 176u);
  printf("Independent simultaneous pthread-lane actual-math pipeline: "
         "%u cases PASS; no DMA or device-timing claim\n", parallel_cases);
  return 0;
}
'''
(OUT / CHECK).write_text(check)
run = (OUT / RUN).read_text()
run = run.replace("flags = ['-std=gnu11',", "flags = ['-std=gnu11', '-pthread',")
run = run.replace("link = [cxx]", "link = [cxx, '-pthread']")
(OUT / RUN).write_text(run)
print(OUT)
