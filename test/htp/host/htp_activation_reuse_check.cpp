// SPDX-License-Identifier: Apache-2.0
/** @file htp_activation_reuse_check.cpp
 * @brief Host checks for the scoped router-to-MoE staging lease.
 */
#include <cassert>
#include <cstdio>
#include <htp_activation_reuse.h>
#include <thread>

int main() {
  nntrainer::HtpActivationReuse lease;
  float input[8]{}, other[8]{}, staging[8]{};
  lease.remember(input, 2, 4, staging);
  assert(!lease.consume(input, 2, 4)); // no caller lifetime promise
  auto token = lease.begin(input, 2, 4);
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
  lease.end(nested);
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
  std::puts("activation reuse lifetime, single consumption, shape, nesting, "
            "and thread checks passed");
}
