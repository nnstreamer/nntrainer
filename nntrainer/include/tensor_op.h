// SPDX-License-Identifier: Apache-2.0-only
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file	tensor_op.h
 * @date	14 August 2020
 * @brief	This is Tensor Operations Class of Neural Network
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __TENSOR_OP_H__
#define __TENSOR_OP_H__
#ifdef __cplusplus

#include <tensor.h>
#include <vector>

namespace nntrainer {

struct TensorParam {
  Tensor var;
  Tensor grad;
  std::string name;
  bool trainable;

  TensorParam(Tensor v, Tensor g, std::string n, bool train = false) :
    var(v),
    grad(g),
    name(n),
    trainable(train) {}
};

/**
 * @class   TensorOp
 * @brief   Tensor Operation
 */
class TensorOp {
public:
  virtual ~TensorOp() {}

  void setInputTensors(std::vector<TensorParam> in) { inputs_op = in; }

  void setOutputTensors(std::vector<TensorParam> out) { outputs_op = out; }

  /**
   * @brief     compute the tensor operation
   */
  virtual void computeOp() {} /// = 0;

  /**
   * @brief     compute the gradient for tensor operation
   */
  virtual void computeGrad() {} /// = 0;

  /**
   * @brief     apply the gradients to the weight
   */
  virtual void applyGrad(int iteration) {} /// = 0;

  /**
   * @brief     initialize the tensor operation and tensors inside it
   */
  void initialize();

protected:
  std::vector<TensorParam> inputs_op;
  std::vector<TensorParam> outputs_op;
};
} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __TENSOR_OP_H__ */
