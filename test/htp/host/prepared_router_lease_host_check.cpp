// SPDX-License-Identifier: Apache-2.0
/** @file prepared_router_lease_host_check.cpp
 * @brief Host checks for the scoped router-to-MoE staging lease.
 * @author NNTrainer contributors
 * @bug See the repository issue tracker for known issues.
 */
#include <cassert>
#include <cstdio>
#include <htp_activation_reuse.h>
#include <thread>

namespace nntrainer {
/** Observable transport facade for the actual layer's scoped lifetime guard. */
class ComputeOps {
public:
  HtpActivationReuse lease;
  unsigned begins = 0, ends = 0;
  uint64_t begin_moe_input_reuse(const float *input, unsigned rows,
                                 unsigned cols) {
    ++begins;
    return lease.begin(input, rows, cols);
  }
  void end_moe_input_reuse(uint64_t token) {
    ++ends;
    lease.end(token);
  }
};
} // namespace nntrainer

// Extracted from lfm2_moe_layer.cpp by the runner, without copying its logic.
#include "prepared_scope_under_test.inc"

int main() {
  nntrainer::HtpActivationReuse lease;
  float input[8]{}, other[8]{}, staging[8]{};
  lease.remember(input, 2, 4, staging);
  assert(!lease.consume(input, 2, 4)); // no caller lifetime promise
  auto token = lease.begin(input, 2, 4);
  assert(lease.token(input, 2, 4) == token);
  assert(!lease.token(other, 2, 4));
  assert(!lease.token(input, 4, 2));
  std::thread query_foreign([&] { assert(!lease.token(input, 2, 4)); });
  query_foreign.join();
  lease.remember(input, 2, 4, staging);
  assert(lease.consume(input, 2, 4) == staging);
  assert(!lease.consume(input, 2, 4)); // consume once, even split FFN calls

  lease.remember(input, 2, 4, staging);
  assert(!lease.consume(other, 2, 4));
  assert(!lease.consume(input, 2, 4)); // mismatch also retires saved staging
  lease.remember(input, 2, 4, staging);
  assert(!lease.consume(input, 4, 2));  // equal bytes, different shape
  lease.remember(other, 2, 4, staging); // router's normalized input differs
  assert(!lease.consume(input, 2, 4));

  lease.remember(input, 2, 4, staging);
  lease.invalidate(); // intervening activation upload, possibly same buffer
  assert(!lease.consume(input, 2, 4));
  lease.remember(input, 2, 4, staging);
  lease.end(token);
  assert(!lease.consume(input, 2, 4));
  lease.remember(input, 2, 4, staging);
  assert(!lease.consume(input, 2, 4)); // ended scope cannot revive staging

  token = lease.begin(input, 2, 4);
  lease.remember(input, 2, 4, staging);
  const auto nested = lease.begin(input, 2, 4);
  assert(nested != token);
  assert(!lease.consume(input, 2, 4)); // nested scope invalidates outer copy
  lease.remember(input, 2, 4, staging);
  lease.end(token); // stale destructor cannot end the new scope
  assert(lease.consume(input, 2, 4) == staging);
  assert(lease.token(input, 2, 4) == nested);
  lease.end(nested);
  assert(!lease.token(input, 2, 4));
  assert(!lease.consume(input, 2, 4));

  token = lease.begin(input, 2, 4);
  lease.remember(input, 2, 4, staging);
  std::thread foreign_consumer([&] { assert(!lease.consume(input, 2, 4)); });
  foreign_consumer.join();
  assert(!lease.consume(input, 2, 4));
  std::thread foreign_router([&] { lease.remember(input, 2, 4, staging); });
  foreign_router.join();
  assert(!lease.consume(input, 2, 4));
  lease.remember(input, 2, 4, staging);
  std::thread intervening_upload([&] { lease.invalidate(); });
  intervening_upload.join();
  assert(!lease.consume(input, 2, 4));
  lease.end(token);
  token = lease.begin(input, 2, 4);
  lease.remember(input, 2, 4, staging);
  lease.invalidate();
  assert(lease.token(input, 2, 4) == token); // prepared Q8 owns its storage
  lease.cancel(); // graph abort/exception ends both forms of reuse
  assert(!lease.token(input, 2, 4));
  assert(!lease.consume(input, 2, 4));
  lease.end(token);
  const auto fresh = lease.begin(input, 2, 4);
  assert(fresh != token);
  lease.end(token);
  assert(lease.token(input, 2, 4) == fresh);
  lease.end(fresh);

  nntrainer::ComputeOps ops;
  {
    MoeInputReuseScope scope(&ops, input, 2, 4);
    assert(ops.lease.token(input, 2, 4));
    ops.lease.remember(input, 2, 4, staging);
  }
  assert(ops.begins == 1 && ops.ends == 1);
  assert(!ops.lease.token(input, 2, 4));
  assert(!ops.lease.consume(input, 2, 4));
  {
    MoeInputReuseScope scope(&ops, input, 2, 4);
    scope.reset();
    scope.reset();
    assert(ops.ends == 2);
  }
  assert(ops.ends == 2); // reset and destructor consume the token once
  try {
    MoeInputReuseScope scope(&ops, input, 2, 4);
    throw 7;
  } catch (int value) {
    assert(value == 7);
  }
  assert(ops.begins == 3 && ops.ends == 3);
  assert(!ops.lease.token(input, 2, 4));
  {
    MoeInputReuseScope outer(&ops, input, 2, 4);
    MoeInputReuseScope inner(&ops, other, 2, 4);
    outer.reset();
    assert(ops.lease.token(other, 2, 4));
  }
  assert(ops.begins == 5 && ops.ends == 5);
  assert(!ops.lease.token(other, 2, 4));
  {
    MoeInputReuseScope disabled(nullptr, input, 2, 4);
    disabled.reset();
  }
  std::puts("activation reuse lifetime, single consumption, shape, nesting, "
            "thread, actual scope destructor and exception checks passed");
}
