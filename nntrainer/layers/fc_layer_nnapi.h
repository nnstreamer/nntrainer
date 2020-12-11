// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file fc_layer_nnapi.h
 * @date 10 December 2020
 * @brief This is NNAPI implementation of Fully Connected Layer
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */

#ifndef __FC_LAYER_NNAPI_H__
#define __FC_LAYER_NNAPI_H__

#ifndef __ANDROID__
#error NNAPI IMPLMENTATION REQUIRES ANDROID
#endif

#ifdef __cplusplus

#include <fc_layer.h>
#include <layer_internal.h>
#include <tensor.h>

#include <android/NeuralNetworks.h>
namespace nntrainer {

/**
 * @class   FullyConnecedLayer APIimplmentation
 * @brief   fully connected layer
 * @note    This is kind of fully connected layer, although it is using seperate
 * specifier for faster development.
 */
class FullyConnectedLayer_NNAPI : public FullyConnectedLayer {
public:
  /**
   * @brief     Constructor of Fully Connected Layer NNAPI
   */
  template <typename... Args>
  FullyConnectedLayer_NNAPI(unsigned int unit_ = 0, Args... args) :
    FullyConnectedLayer(unit_, args...),
    nnapi_model(nullptr),
    nnapi_compilation(nullptr),
    nnapi_execution(nullptr) {}

  /**
   * @brief     Destructor of Fully Connected Layer NNAPI
   */
  ~FullyConnectedLayer_NNAPI();

  /**
   *  @brief  Move constructor of layer
   *  @param[in] FullyConnected &&
   */
  FullyConnectedLayer_NNAPI(FullyConnectedLayer_NNAPI &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @param[in] rhs FullyConnectedLayer_NNAPI to be moved.
   */
  FullyConnectedLayer_NNAPI &
  operator=(FullyConnectedLayer_NNAPI &&rhs) = default;

  /**
   * @copydoc Layer::initialize(Manager &manager)
   */
  int initialize(Manager &manager) override;

  /**
   * @brief copydoc Layer::compile(Manager &manager);
   */
  int postInitialize(Manager &manager) override;

  /**
   * @copydoc Layer::forwarding(sharedConstTensors in)
   */
  void forwarding(sharedConstTensors in) override;

  /** using fc's cpu implementation for now */

  /**
   * @copydoc Layer::calcDerivative(sharedConstTensors in)
   */
  // void calcDerivative(sharedConstTensors in) override;

  /**
   * @copydoc Layer::calcGradient(sharedConstTensors in)
   */
  // void calcGradient(sharedConstTensors in) override;

  /**
   * @copydoc Layer::getType()
   * @todo this needs to be same type as fullyconnectedlayer when enabling
   * backend
   */
  const std::string getType() const override {
    return FullyConnectedLayer_NNAPI::type;
  }

  /// @todo: deprecate this
  static const std::string type;

private:
  ANeuralNetworksMemory *nnapi_weight_mem;
  ANeuralNetworksModel *nnapi_model;
  ANeuralNetworksCompilation *nnapi_compilation;
  ANeuralNetworksExecution *nnapi_execution;
};
} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __FC_LAYER_NNAPI_H__ */
