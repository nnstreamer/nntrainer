// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   layer_context.h
 * @date   10 June 2021
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is the layer context for each layer
 */

#ifndef __LAYER_CONTEXT_H__
#define __LAYER_CONTEXT_H__

#include <memory>
#include <vector>

#include <common_properties.h>
#include <tensor.h>
#include <tensor_dim.h>
#include <var_grad.h>
#include <weight.h>

namespace nntrainer {

/**
 * @class   Layer Context class for all layers
 * @brief   Class for Layer context
 *
 * @details This provides for the layer initialization. This context will not
 * contain any structures which allow allocation of memory or support to
 * allocate any new memory, but rather only support storing specifications based
 * on which memory will be allocated later.
 */
class InitLayerContext {
public:
  /**
   * @brief define the lifespan of the given tensor to reduce peak memory
   *
   */
  enum TensorLifespan {
    FORWARD_FUNC_LIFESPAN,  /**< tensor must not be reset before during the
                               forward function call, eg. temporary tensors
                               needed during forward operations */
    BACKWARD_FUNC_LIFESPAN, /**< tensor must not be reset before during the
                               backward function call, eg. temporary tensors
                               needed during backward operations */
    ITERATION_LIFESPAN,     /**< tensor must not be reset until the owning layer
                               finishes its execution in the current iteration,
                               eg. hidden memory/cells of RNN */
    EPOCH_LIFESPAN, /**< tensor must not be reset before the epoch ends */
    MAX_LIFESPAN,   /**< tensor must not be reset until the end of the model
                       execution, eg. layer weights */
  };

  /**
   * @brief Construct a new Init Layer Context object
   *
   */
  InitLayerContext() = default;

  /**
   * @brief Construct a new Init Layer Context object
   *
   * @param dim Input dimensions for the layer
   */
  InitLayerContext(const std::vector<TensorDim> &dim) : input_dim(dim) {}

  /**
   * @brief Get the Input Dimensions object
   *
   * @return const std::vector<TensorDim>& Input dimensions
   */
  const std::vector<TensorDim> &getInputDimensions() const { return input_dim; }

  /**
   * @brief Get the Output Dimensions object
   *
   * @return std::vector<TensorDim>& Output dimensions
   */
  const std::vector<TensorDim> &getOutputDimensions() const {
    return output_dim;
  }

  /**
   * @brief Set the Output Dimensions object
   *
   * @param out_dim the output dimension to set to
   */
  void setOutputDimensions(const std::vector<TensorDim> &out_dim) {
    output_dim = out_dim;
  }

  /**
   * @brief Request a new weight for the layer
   *
   * @param dim dimension of the weight
   * @param init initializer for the weight
   * @param reg regularizer for the weight
   * @param reg_const regularization constant for the weight
   * @param name name of the weight
   * @param trainable if the weight is trainable (require gradient or not)
   * @return unsigned int index of the weight for its getter
   */
  unsigned int requestWeight(const TensorDim &dim, const WeightInitializer init,
                             const WeightRegularizer reg, const float reg_const,
                             std::string name, bool trainable = true) {
    weights_spec.emplace_back(dim, init, reg, reg_const, trainable, name);
    return weights_spec.size() - 1;
  }

  /**
   * @brief Request a new tensor for the layer
   *
   * @param dim dimension of the tensor
   * @param trainable if the tensor is trainable (require gradient or not)
   * @param name name of the tensor
   * @param lifespan lifespan of the tensor
   * @return unsigned int index of the tensor for its getter
   */
  unsigned int requestTensor(const TensorDim &dim, const std::string &name,
                             bool trainable = false,
                             TensorLifespan lifespan = ITERATION_LIFESPAN) {
    tensors_spec.emplace_back(dim, trainable, name);
    return tensors_spec.size() - 1;
  }

  /**
   * @brief Specification of the tensors
   *
   */
  typedef Var_Grad::Spec TensorSpec;

  /**
   * @brief Get the current weights spec
   *
   * @return The current weights spec
   */
  const std::vector<Weight::Spec> &getWeightsSpec() const {
    return weights_spec;
  }

  /**
   * @brief Get the current tensors spec
   *
   * @return The current tensors spec
   */
  const std::vector<TensorSpec> &getTensorsSpec() const { return tensors_spec; }

private:
  std::vector<TensorDim> input_dim;  /**< Input dimensions for the layer */
  std::vector<TensorDim> output_dim; /**< Output dimensions for the layer */

  std::vector<Weight::Spec> weights_spec; /**< Specification for the weights */
  std::vector<TensorSpec>
    tensors_spec; /**< Specification for the var_grad (trainable/non-trainable
                     variables) */
};

/**
 * @class   Layer Context class for all layers
 * @brief   Class for Layer context
 *
 * @details This provides for the layer executiong. This context will contain
 * structures with memory allocated or support to allocate any new memory, but
 * rather only support storing specifications based on which memory will be
 * allocated later.
 */
class RunLayerContext {
public:
  /**
   * @brief Construct a new Run Layer Context object
   *
   */
  RunLayerContext() = default;

  /**
   * @brief Construct a new Run Layer Context object
   * @todo  Include properties like name/trainable later
   *
   * @param w weights of the layer
   * @param in inputs of the layer
   * @param out outputs of the layer
   * @param t extra tensors of the layer
   */
  RunLayerContext(const std::vector<Weight *> &w,
                  const std::vector<Var_Grad *> &in,
                  const std::vector<Var_Grad *> &out,
                  const std::vector<Var_Grad *> &t) :
    trainable(false),
    weights(w),
    inputs(in),
    outputs(out),
    tensors(t) {}

  /**
   * @brief Get the Weight tensor object
   *
   * @param idx Identifier of the weight
   * @return Tensor& Reference to the weight tensor
   */
  Tensor &getWeight(unsigned int idx) const {
    return weights[idx]->getVariableRef();
  }

  /**
   * @brief Get the Weight Gradient tensor object
   *
   * @param idx Identifier of the weight
   * @return Tensor& Reference to the weight grad tensor
   */
  Tensor &getWeightGrad(unsigned int idx) {
    if (!weights[idx]->getTrainable())
      throw std::invalid_argument(
        "Requesting gradient for a non-trainable weight.");
    if (!trainable)
      throw std::invalid_argument(
        "Requesting gradient for a non-trainable layer.");
    return weights[idx]->getGradientRef();
  }

  /**
   * @brief Get the Output tensor object
   *
   * @param idx Identifier of the output
   * @return Tensor& Reference to the output tensor
   */
  Tensor &getOutput(unsigned int idx) { return outputs[idx]->getVariableRef(); }

  /**
   * @brief Get the Output Grad tensor object
   *
   * @param idx Identifier of the output
   * @return Tensor& Reference to the output grad tensor
   */
  Tensor &getOutputGrad(unsigned int idx) {
    if (!outputs[idx]->getTrainable())
      throw std::invalid_argument(
        "Requesting gradient for a non-trainable tensor.");
    return outputs[idx]->getGradientRef();
  }

  /**
   * @brief Get the Input tensor object
   *
   * @param idx Identifier of the input
   * @return Tensor& Reference to the input grad tensor
   */
  Tensor &getInput(unsigned int idx) { return inputs[idx]->getVariableRef(); }

  /**
   * @brief Get the Input Grad tensor object
   *
   * @param idx Identifier of the input
   * @return Tensor& Reference to the input grad tensor
   */
  Tensor &getInputGrad(unsigned int idx) {
    if (!inputs[idx]->getTrainable())
      throw std::invalid_argument(
        "Requesting gradient for a non-trainable tensor.");
    return inputs[idx]->getGradientRef();
  }

  /**
   * @brief Get the Tensor object
   *
   * @param idx Identifier of the tensor
   * @return Tensor& Reference to the tensor
   */
  Tensor &getTensor(unsigned int idx) { return tensors[idx]->getVariableRef(); }

  /**
   * @brief Get the Tensor Grad object
   *
   * @param idx Identifier of the tensor
   * @return Tensor& Reference to the tensor grad tensor
   */
  Tensor &getTensorGrad(unsigned int idx) {
    if (!tensors[idx]->getTrainable())
      throw std::invalid_argument(
        "Requesting gradient for a non-trainable tensor.");
    return tensors[idx]->getGradientRef();
  }

  /**
   * @brief Get the number of Outputs tensor objects
   *
   * @return unsigned int number of output tensors
   */
  unsigned int getNumOutputs() { return outputs.size(); }

  /**
   * @brief Get the number of inputs tensor objects
   *
   * @return unsigned int number of input tensors
   */
  unsigned int getNumInputs() { return inputs.size(); }

  /**
   * @brief Get the number of weights tensor objects
   *
   * @return unsigned int number of weight tensors
   */
  unsigned int getNumWeights() const { return weights.size(); }

  /**
   * @brief Get the if the layer is trainable
   *
   * @return bool true if trainable, else false
   */
  bool getTrainable() { return trainable; }

private:
  std::tuple<props::Name> props; /**< props of the layer */
  bool trainable;                /**< if the layer is trainable */

  std::vector<Weight *> weights;   /**< weights of the layer */
  std::vector<Var_Grad *> inputs;  /**< inputs of the layer */
  std::vector<Var_Grad *> outputs; /**< outputs of the layer */
  std::vector<Var_Grad *> tensors; /**< tensors of the layer */
};

} // namespace nntrainer
#endif // __LAYER_CONTEXT_H__
