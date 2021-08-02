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
  EPOCH_LIFESPAN,         /**< tensor must not be reset before the epoch ends */
  MAX_LIFESPAN, /**< tensor must not be reset until the end of the model
                  execution, eg. layer weights */
};

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
   * @brief Construct a new Init Layer Context object
   *
   */
  InitLayerContext() : InitLayerContext({}, 1) {}

  /**
   * @brief Construct a new Init Layer Context object
   *
   * @param dim Input dimensions for the layer
   */
  InitLayerContext(const std::vector<TensorDim> &dim, unsigned int num_out) :
    input_dim(dim),
    num_outputs(num_out) {}

  /**
   * @brief Get the number of inputs for the layer
   *
   * @return unsigned int number of inputs
   */
  unsigned int getNumInputs() const { return input_dim.size(); }

  /**
   * @brief Get the number of inputs for the layer
   *
   * @return unsigned int number of inputs
   */
  unsigned int getNumOutputs() const { return num_outputs; }

  /**
   * @brief Get the Input Dimensions object
   *
   * @return const std::vector<TensorDim>& Input dimensions
   */
  const std::vector<TensorDim> &getInputDimensions() const { return input_dim; }

  /**
   * @brief Set the Dim Flag to retrieve effective dimension
   *
   * @param dim_flag_ dimension bit to calculate, rightmost is width
   */
  void setEffDimFlagInputDimension(unsigned int idx,
                                   const std::bitset<MAXDIM> &dim_flag_) {
    input_dim[idx].setEffDimFlag(dim_flag_);
  }

  /**
   * @brief Set the dynamic Dim Flag to retrieve dynamic dimension (that can
   * change during running)
   *
   * @param dim_flag_ dimension bit to calculate, rightmost is width
   */
  void setDynDimFlagInputDimension(unsigned int idx,
                                   const std::bitset<MAXDIM> &dim_flag_) {
    input_dim[idx].setDynDimFlag(dim_flag_);
  }

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
    if (out_dim.size() != num_outputs)
      throw std::invalid_argument("Mismatch number of outputs");
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
   *
   * @todo Consider providing a guarantee that the returned indices will always
   * start from 0 and will always be incremental.
   */
  unsigned int requestWeight(const TensorDim &dim, const WeightInitializer init,
                             const WeightRegularizer reg, const float reg_const,
                             std::string name, bool trainable = true) {
    weights_spec.emplace_back(dim, init, reg, reg_const, trainable, name);
    return weights_spec.size() - 1;
  }

  /**
   * @brief Request a new weight for the layer
   *
   * @param spec tensor spec
   * @return unsigned int index of the weight for its getter
   *
   * @todo Consider providing a guarantee that the returned indices will always
   * start from 0 and will always be incremental.
   */
  unsigned int requestWeight(const Weight::Spec &spec) {
    weights_spec.emplace_back(spec);
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
   *
   * @todo Consider providing a guarantee that the returned indices will always
   * start from 0 and will always be incremental.
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
   * @brief Request a new tensor for the layer
   *
   * @param spec tensor spec
   * @return unsigned int index of the tensor for its getter
   *
   * @todo Consider providing a guarantee that the returned indices will always
   * start from 0 and will always be incremental.
   */
  unsigned int requestTensor(const TensorSpec &spec) {
    tensors_spec.emplace_back(spec);
    return tensors_spec.size() - 1;
  }

  /**
   * @brief Get the current weights spec
   *
   * @return The current weights spec
   */
  const std::vector<Weight::Spec> &getWeightsSpec() const {
    return weights_spec;
  }

  /**
   * @brief Get the number of requested weights
   *
   * @return The current number of requested weights
   */
  unsigned int getNumWeights() const { return weights_spec.size(); }

  /**
   * @brief Get the current tensors spec
   *
   * @return The current tensors spec
   */
  const std::vector<TensorSpec> &getTensorsSpec() const { return tensors_spec; }

  /**
   * @brief Get the number of requested tensors objects
   *
   * @return unsigned int number of requested tensors
   */
  unsigned int getNumTensors() const { return tensors_spec.size(); }

  /**
   * @brief Set the batch for the init context
   *
   * @param batch Updated batch size
   */
  void setBatch(unsigned int batch) {
    for (auto &dim : input_dim)
      dim.batch(batch);
    for (auto &dim : output_dim)
      dim.batch(batch);
  }

  /**
   * @brief Update the dimensions for a requested tensor
   *
   * @param idx index of the tensor (identifier)
   * @param batch Updated batch size
   */
  void updateTensorSpec(unsigned int idx, unsigned int batch) {
    std::get<0>(tensors_spec[idx]).batch(batch);
  }

  /**
   * @brief Validate the context
   *
   * @return true if validated, else false
   * @note this must be called before passing a context to a layer for finalize
   */
  bool validate() {
    if (input_dim.empty()) {
      return false;
    }

    for (auto const &dim : input_dim) {
      if (dim.getDataLen() == 0) {
        return false;
      }
    }

    return true;
  }

private:
  std::vector<TensorDim> input_dim;  /**< Input dimensions for the layer */
  std::vector<TensorDim> output_dim; /**< Output dimensions for the layer */

  std::vector<Weight::Spec> weights_spec; /**< Specification for the weights */
  std::vector<TensorSpec>
    tensors_spec; /**< Specification for the var_grad (trainable/non-trainable
                     variables) */

  unsigned int num_outputs; /**< number of outputs for the layer */
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
  RunLayerContext() : loss(0.0) {}

  /**
   * @brief Construct a new Run Layer Context object
   *
   */
  RunLayerContext(const std::string &name) : RunLayerContext() {
    std::get<props::Name>(props).set(name);
  }

  /**
   * @brief Construct a new Run Layer Context object
   * @todo  Include properties like name/trainable later
   *
   * @param w weights of the layer
   * @param in inputs of the layer
   * @param out outputs of the layer
   * @param t extra tensors of the layer
   */
  RunLayerContext(const std::string &name, float l,
                  const std::vector<Weight *> &w,
                  const std::vector<Var_Grad *> &in,
                  const std::vector<Var_Grad *> &out,
                  const std::vector<Var_Grad *> &t) :
    loss(l),
    weights(w),
    inputs(in),
    outputs(out),
    tensors(t) {
    std::get<props::Name>(props).set(name);
  }

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
  Tensor &getWeightGrad(unsigned int idx) const {
    if (!weights[idx]->hasGradient())
      throw std::invalid_argument(
        "Requesting gradient for a non-trainable weight.");
    return weights[idx]->getGradientRef();
  }

  /**
   * @brief Get the Weight name
   *
   * @param idx Identifier of the weight
   * @return name of the weight
   */
  const std::string &getWeightName(unsigned int idx) const {
    return weights[idx]->getName();
  }

  /**
   * @brief check if the weight has gradient
   *
   * @param idx Identifier of the weight
   * @return true if weight has gradient, else false
   */
  bool weightHasGradient(unsigned int idx) const {
    return weights[idx]->hasGradient();
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
    if (!outputs[idx]->hasGradient())
      throw std::invalid_argument(
        "Requesting gradient for a non-trainable tensor.");
    return getOutputGradUnsafe(idx);
  }

  /**
   * @brief Get the Output Grad tensor object
   *
   * @param idx Identifier of the output
   * @return Tensor& Reference to the output grad tensor
   *
   * @note recommended to NOT use this function as a layer developer but rather
   * use getOutputGrad().
   */
  Tensor &getOutputGradUnsafe(unsigned int idx) {
    return outputs[idx]->getGradientRef();
  }

  /**
   * @brief Get the incoming Derivative tensor object
   *
   * @param idx Identifier of the output
   * @return Tensor& Reference to the output derivative tensor
   */
  Tensor &getIncomingDerivative(unsigned int idx) { return getOutputGrad(idx); }

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
    if (!inputs[idx]->hasGradient())
      throw std::invalid_argument(
        "Requesting gradient for a non-trainable tensor.");
    return inputs[idx]->getGradientRef();
  }

  /**
   * @brief Get the outgoing Derivative tensor object
   *
   * @param idx Identifier of the input
   * @return Tensor& Reference to the input derivative tensor
   */
  Tensor &getOutgoingDerivative(unsigned int idx) { return getInputGrad(idx); }

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
    if (!tensors[idx]->hasGradient())
      throw std::invalid_argument(
        "Requesting gradient for a non-trainable tensor.");
    return tensors[idx]->getGradientRef();
  }

  /**
   * @brief check if the tensor has gradient
   *
   * @param idx Identifier of the tensor
   * @return true if tensor has gradient, else false
   */
  bool tensorHasGradient(unsigned int idx) const {
    return tensors[idx]->hasGradient();
  }

  /**
   * @brief Get the tensor name
   *
   * @param idx Identifier of the tensor
   * @return name of the tensor
   */
  const std::string &getTensorName(unsigned int idx) const {
    return tensors[idx]->getName();
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
   * @brief Get the number of requested tensors objects
   *
   * @return unsigned int number of requested tensors
   */
  unsigned int getNumTensors() const { return tensors.size(); }

  /**
   * @brief Set the batch for the run context
   *
   * @param batch Update batch size
   */
  void setBatch(unsigned int batch) {
    for (auto &vg : inputs)
      vg->setBatchSize(batch);
    for (auto &vg : outputs)
      vg->setBatchSize(batch);
  }

  /**
   * @brief Update the dimensions for a requested tensor
   *
   * @param idx index of the tensor (identifier)
   * @param batch Updated batch size
   */
  void updateTensor(unsigned int idx, unsigned int batch) {
    tensors[idx]->setBatchSize(batch);
  }

  /**
   * @brief   Get weight object for the weights
   *
   * @param idx index of the weight (identifier)
   * @return weight object
   */
  Weight &getWeightObject(unsigned int idx) { return *weights[idx]; }

  /**
   * @brief   check if the label is available
   *
   * @param idx Identifier of the input
   * @return true if label is available else false
   */
  bool isLabelAvailable(unsigned int idx) const {
    return !outputs[idx]->getGradientRef().uninitialized();
  }

  /**
   * @brief   Get label tensor
   *
   * @param idx Identifier of the input
   * @return Tensor& Reference to the label tensor
   */
  Tensor &getLabel(unsigned int idx) {
    if (isLabelAvailable(idx))
      return outputs[idx]->getGradientRef();
    else
      throw std::invalid_argument("Request tensor which does not exist");
  }

  /**
   * @brief   update loss by the layer
   *
   * @param val updated loss value
   * @note loss value is only used for loss layers. For non-loss layers, setting
   * this value will have no change on the behavior of the model.
   */
  void setLoss(float val) { loss = val; }

  /**
   * @brief   update loss by the layer
   *
   * @return loss of the layer
   * @note does not includes the regularization loss.
   */
  float getLoss() const { return loss; }

  /**
   * @brief   get regularization loss of the layer
   *
   * @return regularization loss of the layer
   */
  float getRegularizationLoss() const {
    float loss_ = 0;
    for (unsigned int idx = 0; idx < getNumWeights(); idx++) {
      loss_ += getWeightRegularizationLoss(idx);
    }
    return loss_;
  }

  /**
   * @brief   get name by the layer
   *
   * @return name of the layer
   */
  const std::string &getName() const { return std::get<props::Name>(props); }

  /**
   * @brief   check if run context is set and is ready to use
   *
   * @return true if ready, else false
   */
  bool readyToUse() const {
    /**
     * assumption:
     * 1. there must be atleast 1 input
     * 2. the setter set everything at once
     */
    if (inputs.empty())
      return false;
    return !inputs[0]->getVariable().uninitialized();
  }

private:
  std::tuple<props::Name> props; /**< props of the layer */
  float loss;                    /**< loss of the layer */

  std::vector<Weight *> weights;   /**< weights of the layer */
  std::vector<Var_Grad *> inputs;  /**< inputs of the layer */
  std::vector<Var_Grad *> outputs; /**< outputs of the layer */
  std::vector<Var_Grad *> tensors; /**< tensors of the layer */

  /**
   * @brief Get regularization loss for the weight
   *
   * @param idx Identifier of the weight
   * @return float Value of the loss
   */
  float getWeightRegularizationLoss(unsigned int idx) const {
    if (weights[idx]->hasGradient())
      return weights[idx]->getRegularizationLoss();

    return 0;
  }
};

} // namespace nntrainer
#endif // __LAYER_CONTEXT_H__
