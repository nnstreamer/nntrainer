// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 hyeonseok lee <hs89.lee@samsung.com>
 *
 * @file   lstmcell_core.cpp
 * @date   25 November 2021
 * @brief  These are lstm core functions.
 * @see    https://github.com/nnstreamer/nntrainer
 * @author hyeonseok lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <lstmcell_core.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>

#ifdef USE_BLAS
#include <cblas.h>
#endif

namespace nntrainer {

void lstmcell_forwarding(const unsigned int batch_size, const unsigned int unit,
                         const bool disable_bias, const bool integrate_bias,
                         ActiFunc &acti_func, ActiFunc &recurrent_acti_func,
                         const Tensor &input, const Tensor &prev_hidden_state,
                         const Tensor &prev_cell_state, Tensor &hidden_state,
                         Tensor &cell_state, const Tensor &weight_ih,
                         const Tensor &weight_hh, const Tensor &bias_h,
                         const Tensor &bias_ih, const Tensor &bias_hh,
                         Tensor &ifgo) {
  input.dot(weight_ih, ifgo);
  prev_hidden_state.dot(weight_hh, ifgo, false, false, 1.0);
  if (!disable_bias) {
    if (integrate_bias) {
      ifgo.add_i(bias_h);
    } else {
      ifgo.add_i(bias_ih);
      ifgo.add_i(bias_hh);
    }
  }

  Tensor input_forget_gate =
    ifgo.getSharedDataTensor({batch_size, 1, 1, unit * 2}, 0, false);
  Tensor input_gate =
    ifgo.getSharedDataTensor({batch_size, 1, 1, unit}, 0, false);
  Tensor forget_gate =
    ifgo.getSharedDataTensor({batch_size, 1, 1, unit}, unit, false);
  Tensor memory_cell =
    ifgo.getSharedDataTensor({batch_size, 1, 1, unit}, unit * 2, false);
  Tensor output_gate =
    ifgo.getSharedDataTensor({batch_size, 1, 1, unit}, unit * 3, false);

  recurrent_acti_func.run_fn(input_forget_gate, input_forget_gate);
  recurrent_acti_func.run_fn(output_gate, output_gate);
  acti_func.run_fn(memory_cell, memory_cell);

  prev_cell_state.multiply_strided(forget_gate, cell_state);
  memory_cell.multiply_strided(input_gate, cell_state, 1.0f);

  acti_func.run_fn(cell_state, hidden_state);
  hidden_state.multiply_i_strided(output_gate);
}

void lstmcell_calcDerivative(Tensor &outgoing_derivative,
                             const Tensor &weight_ih, const Tensor &d_ifgo,
                             const float alpha) {
  d_ifgo.dot(weight_ih, outgoing_derivative, false, true, alpha);
}

void lstmcell_calcGradient(
  const unsigned int batch_size, const unsigned int unit,
  const bool disable_bias, const bool integrate_bias, ActiFunc &acti_func,
  ActiFunc &recurrent_acti_func, const Tensor &input,
  const Tensor &prev_hidden_state, Tensor &d_prev_hidden_state,
  const Tensor &prev_cell_state, Tensor &d_prev_cell_state,
  const Tensor &d_hidden_state, const Tensor &cell_state,
  const Tensor &d_cell_state, Tensor &d_weight_ih, const Tensor &weight_hh,
  Tensor &d_weight_hh, Tensor &d_bias_h, Tensor &d_bias_ih, Tensor &d_bias_hh,
  const Tensor &ifgo, Tensor &d_ifgo) {
  Tensor input_forget_gate =
    ifgo.getSharedDataTensor({batch_size, 1, 1, unit * 2}, 0, false);
  Tensor input_gate =
    ifgo.getSharedDataTensor({batch_size, 1, 1, unit}, 0, false);
  Tensor forget_gate =
    ifgo.getSharedDataTensor({batch_size, 1, 1, unit}, unit, false);
  Tensor memory_cell =
    ifgo.getSharedDataTensor({batch_size, 1, 1, unit}, unit * 2, false);
  Tensor output_gate =
    ifgo.getSharedDataTensor({batch_size, 1, 1, unit}, unit * 3, false);

  Tensor d_input_forget_gate =
    d_ifgo.getSharedDataTensor({batch_size, 1, 1, unit * 2}, 0, false);
  Tensor d_input_gate =
    d_ifgo.getSharedDataTensor({batch_size, 1, 1, unit}, 0, false);
  Tensor d_forget_gate =
    d_ifgo.getSharedDataTensor({batch_size, 1, 1, unit}, unit, false);
  Tensor d_memory_cell =
    d_ifgo.getSharedDataTensor({batch_size, 1, 1, unit}, unit * 2, false);
  Tensor d_output_gate =
    d_ifgo.getSharedDataTensor({batch_size, 1, 1, unit}, unit * 3, false);

  Tensor activated_cell_state;
  acti_func.run_fn(cell_state, activated_cell_state);
  d_hidden_state.multiply_strided(activated_cell_state, d_output_gate);
  acti_func.run_prime_fn(activated_cell_state, d_prev_cell_state,
                         d_hidden_state);
  d_prev_cell_state.multiply_i_strided(output_gate);
  d_prev_cell_state.add_i(d_cell_state);

  d_prev_cell_state.multiply_strided(input_gate, d_memory_cell);
  d_prev_cell_state.multiply_strided(memory_cell, d_input_gate);

  d_prev_cell_state.multiply_strided(prev_cell_state, d_forget_gate);
  d_prev_cell_state.multiply_i_strided(forget_gate);

  recurrent_acti_func.run_prime_fn(output_gate, d_output_gate, d_output_gate);
  recurrent_acti_func.run_prime_fn(input_forget_gate, d_input_forget_gate,
                                   d_input_forget_gate);
  acti_func.run_prime_fn(memory_cell, d_memory_cell, d_memory_cell);

  if (!disable_bias) {
    if (integrate_bias) {
      d_ifgo.sum(0, d_bias_h, 1.0f, 1.0f);
    } else {
      d_ifgo.sum(0, d_bias_ih, 1.0f, 1.0f);
      d_ifgo.sum(0, d_bias_hh, 1.0f, 1.0f);
    }
  }

  if (input.batch() != 1) {
    input.dot(d_ifgo, d_weight_ih, true, false, 1.0f);
  } else {
    for (unsigned int i = 0; i < d_weight_ih.height(); ++i) {
      unsigned int out_width = d_weight_ih.width();
      float in_ih = input.getValue(i);

      float *d_weight_ih_address = d_weight_ih.getAddress(i * out_width);

      float *d_ifgo_address = d_ifgo.getData();
#ifdef USE_BLAS
      cblas_saxpy(out_width, in_ih, d_ifgo_address, 1, d_weight_ih_address, 1);
#else
      for (unsigned int j = 0; j < out_width; ++j) {
        d_weight_ih_address[j] += d_ifgo_address[j] * in_ih;
      }
#endif
    }
  }

  if (prev_hidden_state.batch() != 1) {
    prev_hidden_state.dot(d_ifgo, d_weight_hh, true, false, 1.0f);
  } else {
    for (unsigned int i = 0; i < d_weight_hh.height(); ++i) {
      unsigned int out_width = d_weight_hh.width();
      float in_hh = prev_hidden_state.getValue(i);

      float *d_weight_hh_address = d_weight_hh.getAddress(i * out_width);

      float *d_ifgo_address = d_ifgo.getData();

#ifdef USE_CBLAS
      cblas_saxpy(out_width, in_hh, d_ifgo_address, 1, d_weight_hh_address, 1);
#else
      for (unsigned int j = 0; j < out_width; ++j) {
        d_weight_hh_address[j] += d_ifgo_address[j] * in_hh;
      }
#endif
    }
  }

  d_ifgo.dot(weight_hh, d_prev_hidden_state, false, true);
}

} // namespace nntrainer
