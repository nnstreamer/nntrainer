// SPDX-License-Identifier: Apache-2.0
/** @file nntr_hvx_expert_update.h
 * @brief Internal queue operation; no separate FastRPC method is needed.
 * @author NNTrainer contributors
 * @bug See the repository issue tracker for known issues.
 */
#ifndef NNTR_HVX_EXPERT_UPDATE_H
#define NNTR_HVX_EXPERT_UPDATE_H
#include <htp_backend/htp_dspq_wire.h>
#include <remote.h>

int nntr_hvx_expert_update_batch(remote_handle64 handle, uint32_t K,
                                 uint32_t inter, uint32_t N_out,
                                 const htp_dspq_expert_update *updates,
                                 uint32_t count);
#endif
