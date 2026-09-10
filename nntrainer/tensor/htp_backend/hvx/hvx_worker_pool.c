// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   hvx_worker_pool.c
 * @date   06 Aug 2026
 * @brief  Fixed-size QuRT thread pool for splitting HVX-bound work by index
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * The seqn/barrier/futex protocol below is not novel -- it is
 * llama.cpp ggml-hexagon's htp/work-queue.c pattern (same QuRT primitives,
 * same target), trimmed to a single-job-in-flight fork/join: this pool has
 * no run_async and no multi-slot queue, since every caller here blocks
 * until its job is done anyway. That is a smaller, already-proven surface
 * for what is otherwise this codebase's first genuinely concurrent code
 * path.
 */

#include "hvx_worker_pool.h"

#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>

#include <HAP_perf.h>
#include <qurt.h>

/** @brief Hexagon SMT pause hint for the calling thread's spin-wait below --
 *         same instruction ggml-hexagon's work-queue.c uses while a job is
 *         in flight. */
static inline void hvx_worker_pool_pause(void) {
  asm volatile(" pause(#255)\n");
}

/**
 * @brief How long a worker spins on seqn before it goes back to the futex.
 *
 * The gaps this has to cover are the ones INSIDE one layer call. A gate_up
 * call dispatches the input quant, then runs ~350 us of HMX and dequant
 * with no dispatch at all, then SwiGLU, then the output requant -- five
 * dispatches, and a worker that sleeps between them pays a futex wake each
 * time. That wake is the same order of magnitude as the 4-22 us of HVX work
 * a dispatch actually carries, which is what put five dispatches per call at
 * roughly a fifth of the call's DSP time (doc 44 section 7.4).
 *
 * 400 us covers that middle stretch. It does not separate cleanly from the
 * gap BETWEEN calls (~270 us of FastRPC transport), so while a model is
 * running these threads mostly spin rather than sleep -- which is the
 * intent. The DSP session is dedicated for the duration, and pause(#255)
 * releases the hardware thread for its own duration, so a spinner is not
 * taking issue slots from the thread doing real work. Once inference stops
 * the spin expires once per worker and they are all back on the futex.
 *
 * ponytail: one constant, and nothing adapts it. Ceiling: a workload whose
 * dispatch rhythm differs from this one wants a different number and there
 * is nothing here that would notice. Upgrade path, in the order the profile
 * would justify them -- (1) count sleepers so hvx_worker_pool_run can skip
 * qurt_futex_wake when none are asleep, which is race-free because
 * futex_wait rechecks seqn against prev_seqn itself before sleeping; (2)
 * grow and shrink the spin from how often it expires with a job arriving
 * immediately afterwards.
 */
#define HVX_WORKER_POOL_SPIN_US 400u

/** @brief Pause iterations between two qtimer reads while spinning. The
 *         timer read is the expensive part of the loop; this amortizes it
 *         without letting the spin overshoot its budget meaningfully. */
#define HVX_WORKER_POOL_SPIN_PAUSES 64u

/** @brief Per-worker stack, generous for the quant kernels this pool
 *         currently runs (a handful of HVX vector locals, no recursion). */
#define HVX_WORKER_POOL_STACK_SIZE (16u * 1024u)

typedef struct hvx_worker_pool_s hvx_worker_pool;

typedef struct {
  hvx_worker_pool *pool;
  uint32_t id; /**< 1..n_workers; 0 is reserved for the calling thread */
} hvx_worker_ctx;

struct hvx_worker_pool_s {
  _Atomic uint32_t seqn;    /**< bumped once per hvx_worker_pool_run call */
  _Atomic uint32_t barrier; /**< participating workers still running */
  _Atomic int killed;
  hvx_worker_pool_func func;
  void *ctx;
  uint32_t n_threads; /**< participants (main + workers) for the current run */
  uint32_t n_workers;
  qurt_thread_t *tids;
  hvx_worker_ctx *worker_ctx;
  unsigned char *stack_blob;
};

/**
 * @brief Spins on seqn for up to HVX_WORKER_POOL_SPIN_US.
 *
 * @return the new seqn if a job was published while spinning, otherwise
 *         @a prev_seqn -- meaning the caller should sleep on the futex.
 */
static uint32_t hvx_worker_pool_spin(const hvx_worker_pool *pool,
                                     uint32_t prev_seqn) {
  const uint64_t t0 = HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count());
  for (;;) {
    for (uint32_t i = 0; i < HVX_WORKER_POOL_SPIN_PAUSES; ++i) {
      hvx_worker_pool_pause();
      const uint32_t seqn =
        atomic_load_explicit(&pool->seqn, memory_order_acquire);
      if (seqn != prev_seqn) {
        return seqn;
      }
    }
    // Checked here rather than only at the loop top so that teardown cannot
    // be delayed by a full spin budget. hvx_worker_pool_destroy does bump
    // seqn after setting this, so the read above would catch it too; this
    // just makes the loop's termination obvious without relying on that.
    if (atomic_load_explicit(&pool->killed, memory_order_relaxed)) {
      return prev_seqn;
    }
    const uint64_t now =
      HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count());
    if (now - t0 >= HVX_WORKER_POOL_SPIN_US) {
      return prev_seqn;
    }
  }
}

static void hvx_worker_pool_thread_entry(void *arg) {
  hvx_worker_ctx *me = (hvx_worker_ctx *)arg;
  hvx_worker_pool *pool = me->pool;
  uint32_t prev_seqn = 0;

  for (;;) {
    if (atomic_load_explicit(&pool->killed, memory_order_relaxed)) {
      qurt_thread_exit(0);
    }

    uint32_t seqn = atomic_load_explicit(&pool->seqn, memory_order_acquire);
    if (seqn == prev_seqn) {
      seqn = hvx_worker_pool_spin(pool, prev_seqn);
    }
    if (seqn == prev_seqn) {
      // The spin expired with no job. futex_wait rechecks seqn against
      // prev_seqn atomically before it sleeps, so a job published between
      // the spin's last read and this call is not lost -- the same
      // guarantee the pre-spin version of this loop relied on.
      qurt_futex_wait(&pool->seqn, (int)prev_seqn);
      continue;
    }
    prev_seqn = seqn;

    if (me->id < pool->n_threads) {
      pool->func(pool->n_threads, me->id, pool->ctx);
      atomic_fetch_sub_explicit(&pool->barrier, 1, memory_order_release);
    }
    // me->id >= pool->n_threads: this run didn't need this worker; loop
    // back and wait for the next one.
  }
}

hvx_worker_pool *hvx_worker_pool_create(uint32_t n_workers) {
  hvx_worker_pool *pool = (hvx_worker_pool *)calloc(1, sizeof(*pool));
  if (!pool) {
    return NULL;
  }
  atomic_init(&pool->seqn, 0);
  atomic_init(&pool->barrier, 0);
  atomic_init(&pool->killed, 0);
  pool->n_workers = n_workers;

  if (n_workers == 0) {
    return pool;
  }

  pool->tids = (qurt_thread_t *)calloc(n_workers, sizeof(qurt_thread_t));
  pool->worker_ctx =
    (hvx_worker_ctx *)calloc(n_workers, sizeof(hvx_worker_ctx));
  pool->stack_blob =
    (unsigned char *)malloc((size_t)n_workers * HVX_WORKER_POOL_STACK_SIZE);
  if (!pool->tids || !pool->worker_ctx || !pool->stack_blob) {
    hvx_worker_pool_destroy(pool);
    return NULL;
  }

  qurt_thread_attr_t attr;
  qurt_thread_attr_init(&attr);
  qurt_thread_attr_set_stack_size(&attr, HVX_WORKER_POOL_STACK_SIZE);

  // Match the creating thread's priority, same as ggml-hexagon's
  // work_queue_init -- these workers only ever run while the FastRPC
  // thread that owns this session is waiting on them, so there is no
  // reason for them to run at a different priority.
  int prio = qurt_thread_get_priority(qurt_thread_get_id());
  if (prio < 1) {
    prio = 1;
  }
  qurt_thread_attr_set_priority(&attr, (unsigned short)prio);

  for (uint32_t i = 0; i < n_workers; ++i) {
    pool->worker_ctx[i].pool = pool;
    pool->worker_ctx[i].id = i + 1;
    qurt_thread_attr_set_stack_addr(
      &attr, pool->stack_blob + (size_t)i * HVX_WORKER_POOL_STACK_SIZE);

    char name[16];
    snprintf(name, sizeof(name), "hvxpool:%u", (unsigned)i);
    qurt_thread_attr_set_name(&attr, name);

    if (qurt_thread_create(&pool->tids[i], &attr, hvx_worker_pool_thread_entry,
                           &pool->worker_ctx[i]) != 0) {
      // Only i threads actually started; limit teardown's join loop to
      // those.
      pool->n_workers = i;
      hvx_worker_pool_destroy(pool);
      return NULL;
    }
  }

  return pool;
}

void hvx_worker_pool_destroy(hvx_worker_pool *pool) {
  if (!pool) {
    return;
  }
  if (pool->n_workers > 0 && pool->tids) {
    atomic_store_explicit(&pool->killed, 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&pool->seqn, 1, memory_order_release);
    qurt_futex_wake(&pool->seqn, (int)pool->n_workers);
    for (uint32_t i = 0; i < pool->n_workers; ++i) {
      int status;
      qurt_thread_join(pool->tids[i], &status);
    }
  }
  free(pool->tids);
  free(pool->worker_ctx);
  free(pool->stack_blob);
  free(pool);
}

void hvx_worker_pool_run(hvx_worker_pool *pool, hvx_worker_pool_func func,
                         void *ctx, uint32_t n_units) {
  if (!pool || pool->n_workers == 0 || n_units <= 1) {
    func(n_units == 0 ? 1u : n_units, 0, ctx);
    return;
  }

  uint32_t n = n_units;
  if (n > pool->n_workers + 1u) {
    n = pool->n_workers + 1u;
  }

  pool->func = func;
  pool->ctx = ctx;
  pool->n_threads = n;
  atomic_store_explicit(&pool->barrier, n - 1u, memory_order_relaxed);

  // Publish the job, then wake every worker -- not just the n-1 that will
  // participate. qurt_futex_wake(addr, k) wakes k ARBITRARY sleepers on
  // addr, not k specific ones by id: waking only n-1 risked waking
  // non-participants while the actual participants stayed asleep forever
  // (found on-device: the very first call deadlocked here). A
  // non-participant that wakes just rechecks its id against n_threads,
  // finds it doesn't apply, and goes back to sleep -- harmless.
  atomic_fetch_add_explicit(&pool->seqn, 1, memory_order_release);
  qurt_futex_wake(&pool->seqn, (int)pool->n_workers);

  func(n, 0, ctx); // the calling thread is worker 0

  while (atomic_load_explicit(&pool->barrier, memory_order_relaxed) > 0) {
    hvx_worker_pool_pause();
  }
  // Pairs with each worker's release store to barrier: makes every
  // worker's writes to ctx visible to the calling thread from here on.
  atomic_thread_fence(memory_order_acquire);
}
