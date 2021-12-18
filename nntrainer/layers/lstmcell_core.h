// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 hyeonseok lee <hs89.lee@samsung.com>
 *
 * @file   lstmcell_core.h
 * @date   25 November 2021
 * @brief  These are lstm core functions.
 * @see	   https://github.com/nnstreamer/nntrainer
 * @author hyeonseok lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __LSTMCELLCORE_H__
#define __LSTMCELLCORE_H__
#ifdef __cplusplus

#include <acti_func.h>

namespace nntrainer {

void lstmcell_forwarding(const unsigned int unit, const unsigned int batch_size,
                         const bool disable_bias, const bool integrate_bias,
                         ActiFunc &acti_func, ActiFunc &recurrent_acti_func,
                         const Tensor &input, const Tensor &prev_hidden_state,
                         const Tensor &prev_cell_state, Tensor &hidden_state,
                         Tensor &cell_state, const Tensor &weight_ih,
                         const Tensor &weight_hh, const Tensor &bias_h,
                         const Tensor &bias_ih, const Tensor &bias_hh,
                         Tensor &ifgo);
void lstmcell_calcDerivative(const Tensor &d_ifgo, const Tensor &weight_ih,
                             Tensor &outgoing_derivative);
void lstmcell_calcGradient(
  const unsigned int unit, const unsigned int batch_size,
  const bool disable_bias, const bool integrate_bias, ActiFunc &acti_func,
  ActiFunc &recurrent_acti_func, const Tensor &input,
  const Tensor &prev_hidden_state, Tensor &d_prev_hidden_state,
  const Tensor &prev_cell_state, Tensor &d_prev_cell_state,
  Tensor &d_hidden_state, const Tensor &cell_state, Tensor &d_cell_state,
  Tensor &d_weight_ih, const Tensor &weight_hh, Tensor &d_weight_hh,
  Tensor &d_bias_h, Tensor &d_bias_ih, Tensor &d_bias_hh, const Tensor &ifgo,
  Tensor &d_ifgo);

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __LSTMCELLCORE_H__ */
