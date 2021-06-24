// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   plugged_layer.h
 * @date   27 January 2021
 * @brief  This file contains a wrapper for a plugged layer, INTERNAL USE ONLY
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __PLUGGED_LAYER_H__
#define __PLUGGED_LAYER_H__

#include <layer.h>

#include <layer_internal.h>
#include <manager.h>
#include <nntrainer_error.h>

namespace nntrainer {
namespace internal {

/**
 * @brief PluggedLayer to wrap a layer from shared object file
 *
 */
class PluggedLayer : public nntrainer::LayerV1 {
public:
  /**
   * @brief Construct a new Plugged Layer object
   *
   * @param pluggable LayerPluggable structure from the symbol
   */
  PluggedLayer(const nntrainer::LayerV1Pluggable *pluggable) :
    /// @todo we won't need dynamic pointer cast here after api is fully
    /// implemented
    layerImpl(pluggable->createfunc()),
    destroy_func(pluggable->destroyfunc) {
    NNTR_THROW_IF(layerImpl == nullptr, std::invalid_argument)
      << "either create_func_ failed or cannot dynamic cast to layer_internal";
  }

  /**
   * @brief Destroy the Plugged Layer object
   *
   */
  ~PluggedLayer() override { destroy_func(layerImpl); }

  /**
   * @brief Move Contruct Plugged Layer object
   *
   * @param rhs layer to move
   */
  PluggedLayer(PluggedLayer &&rhs) noexcept = default;

  /**
   * @brief Move assign Plugged Layer Object
   *
   * @param rhs layer to move
   * @return PluggedLayer& *this
   */
  PluggedLayer &operator=(PluggedLayer &&rhs) = default;

  /**
   * @copydoc Layer::initialize(Manager &manager)
   */
  int initialize(Manager &manager) override {
    return layerImpl->initialize(manager);
  }

  /**
   * @copydoc Layer::forwarding(bool training)
   */
  void forwarding(bool training = true) override {
    layerImpl->forwarding(training);
  }

  /**
   * @copydoc Layer::calcDerivative()
   */
  void calcDerivative() override { layerImpl->calcDerivative(); }

  /**
   * @copydoc Layer::calcGradient()
   */
  void calcGradient() override { layerImpl->calcGradient(); }

  /**
   * @copydoc Layer::applyGradient(unsigned int, std::shared_ptr<Optimizer>)
   */
  void applyGradient(unsigned int iteration,
                     std::shared_ptr<Optimizer> optimizer) override {
    layerImpl->applyGradient(iteration, std::move(optimizer));
  }

  /**
   * @copydoc Layer::read(std::ifstream &file)
   */
  void read(std::ifstream &file) override { layerImpl->read(file); }

  /**
   * @copydoc Layer::save(std::ofstream &file)
   */
  void save(std::ofstream &file) override { layerImpl->save(file); }

  /**
   * @copydoc Layer::setProperty(std::vector<std::string> values)
   */
  int setProperty(std::vector<std::string> values) override {
    return layerImpl->setProperty(std::move(values));
  }

  /**
   * @copydoc Layer::checkValidation()
   */
  int checkValidation() override { return layerImpl->checkValidation(); }

  /**
   * @copydoc Layer::getOutputDimension()
   */
  std::vector<TensorDim> getOutputDimension() override {
    return layerImpl->getOutputDimension();
  }

  /**
   * @copydoc Layer::getInputDimension()
   */
  std::vector<TensorDim> getInputDimension() override {
    return layerImpl->getInputDimension();
  }

  /**
   * @copydoc Layer::getLoss()
   */
  float getLoss() override { return layerImpl->getLoss(); }

  /**
   * @copydoc Layer::copy(std::shared_ptr<Layer> l)
   */
  void copy(std::shared_ptr<LayerV1> l) override { layerImpl->copy(l); }

  /**
   * @copydoc bool supportBackwarding() const
   */
  bool supportBackwarding() const override {
    return layerImpl->supportBackwarding();
  };

  /**
   * @copydoc Layer::getWeights()
   */
  std::vector<Weight> getWeights() override { return layerImpl->getWeights(); }

  /**
   * @copydoc Layer::getType()
   */
  virtual const std::string getType() const override {
    return layerImpl->getType();
  }

  /**
   * @copydoc Layer::printPreset(std::ostream &out, PrintPreset preset)
   */
  void printPreset(std::ostream &out,
                   PrintPreset preset = PrintPreset::PRINT_SUMMARY) override {
    return layerImpl->printPreset(out, preset);
  }

  /**
   * @copydoc Layer::weightAt(const unsigned int position)
   */
  Weight &weightAt(const unsigned int position) override {
    return layerImpl->weightAt(position);
  }

  /**
   * @copydoc Layer::getNumWeights()
   */
  unsigned int getNumWeights() override { return layerImpl->getNumWeights(); }

  /**
   * @copydoc Layer::setBatch(unsigned int batch)
   */
  void setBatch(unsigned int batch) override {
    return layerImpl->setBatch(batch);
  }

  /**
   * @copydoc Layer::getOutputs()
   */
  std::vector<Tensor> getOutputs() override { return layerImpl->getOutputs(); }

  /**
   * @copydoc Layer::getDerivatives()
   */
  std::vector<Tensor> getDerivatives() override {
    return layerImpl->getDerivatives();
  }

  /**
   * @copydoc Layer::getWeightsRef()
   */
  std::vector<Weight> &getWeightsRef() override {
    return layerImpl->getWeightsRef();
  }

  /**
   * @copydoc Layer::setInputBuffers(std::vector<std::shared_ptr<VarGrad>>
   * inputs)
   */
  void setInputBuffers(std::vector<std::shared_ptr<Var_Grad>> inputs) override {
    return layerImpl->setInputBuffers(std::move(inputs));
  }

  /**
   * @copydoc Layer::setOutputBuffers(std::vector<std::shared_ptr<Var_Grad>>
   * outputs)
   */
  void
  setOutputBuffers(std::vector<std::shared_ptr<Var_Grad>> outputs) override {
    return layerImpl->setOutputBuffers(std::move(outputs));
  }

#ifdef ENABLE_TEST
  unsigned int getNumInputs() override { return layerImpl->getNumInputs(); }
  unsigned int getNumOutputs() override { return layerImpl->getNumOutputs(); }
#endif

private:
  /// @todo: migrate to ml::train::Layer
  // ml::train::Layer *layerImpl;
  nntrainer::LayerV1 *layerImpl;
  nntrainer::DestroyLayerV1Func destroy_func;
};
} // namespace internal
} // namespace nntrainer

#endif // __PLUGGED_LAYER_H__
