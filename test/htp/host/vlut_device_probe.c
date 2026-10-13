// SPDX-License-Identifier: Apache-2.0
/**
 * @file vlut_device_probe.c
 * @brief Reports repeated VLUT cycle counts from one CDSP session.
 * @author NNTrainer contributors
 * @bug See the repository issue tracker for known issues.
 */
#include "nntr_hvx.h"
#include <AEEStdErr.h>
#include <assert.h>
#include <inttypes.h>
#include <remote.h>
#include <stdio.h>

int main(void) {
  struct remote_rpc_control_unsigned_module u = {CDSP_DOMAIN_ID, 1};
  assert(remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &u,
                                sizeof(u)) == AEE_SUCCESS);
  remote_handle64 session;
  assert(nntr_hvx_open(nntr_hvx_URI "&_dom=cdsp", &session) == AEE_SUCCESS);
  const uint32 n = 200000u;
  for (uint32 r = 0; r < 10u; ++r)
    for (uint32 k = 0; k < 4u; ++k) {
      const uint32 mode = r & 1u ? 3u - k : k;
      uint32 stats[5] = {0};
      assert(nntr_hvx_vlut_probe(session, mode, n, stats, 5) == AEE_SUCCESS);
      const uint64_t cycles = ((uint64_t)stats[1] << 32) | stats[0];
      const uint64_t us = ((uint64_t)stats[3] << 32) | stats[2];
      printf("VLUT rep=%u mode=%u n=%u cycles=%" PRIu64 " us=%" PRIu64
             " cycles_per_op=%.6f checksum=%u\n",
             r, mode, n, cycles, us, (double)cycles / (8.0 * n), stats[4]);
    }
  assert(nntr_hvx_close(session) == AEE_SUCCESS);
  return 0;
}
