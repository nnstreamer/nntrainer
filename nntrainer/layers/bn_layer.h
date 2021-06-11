/**
 * Copyright (C) 2020 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @file	bn_layer.h
 * @date	14 May 2020
 * @brief	This is Batch Normalization Layer Class of Neural Network
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __BN_LAYER_H__
#define __BN_LAYER_H__
#ifdef __cplusplus

#include <array>
#include <functional>
#include <vector>

#include <layer_internal.h>
#include <tensor.h>

namespace nntrainer {

/**
 * @class   BatchNormalizationLayer
 * @brief   Batch Noramlization Layer
 */
class BatchNormalizationLayer : public LayerV1 {
public:
  /**
   * @brief     Constructor of Batch Noramlization Layer
   */
  template <typename... Args>
  BatchNormalizationLayer(
    int axis = -1, float momentum = 0.99, float epsilon = 0.001,
    WeightInitializer moving_mean_initializer = WeightInitializer::WEIGHT_ZEROS,
    WeightInitializer moving_variance_initializer =
      WeightInitializer::WEIGHT_ZEROS,
    WeightInitializer gamma_initializer = WeightInitializer::WEIGHT_ONES,
    WeightInitializer beta_initializer = WeightInitializer::WEIGHT_ONES,
    Args... args) :
    LayerV1(args...),
    epsilon(epsilon),
    momentum(momentum),
    axis(axis),
    initializers{moving_variance_initializer, moving_variance_initializer,
                 gamma_initializer, beta_initializer} {}

  /**
   * @brief     Destructor of BatchNormalizationLayer
   */
  ~BatchNormalizationLayer() {}

  /**
   *  @brief  Move constructor of Pooling 2D Layer.
   *  @param[in] BatchNormalization &&
   */
  BatchNormalizationLayer(BatchNormalizationLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs BatchNormalizationLayer to be moved.
   */
  BatchNormalizationLayer &operator=(BatchNormalizationLayer &&rhs) = default;

  /**
   * @copydoc Layer::forwarding(bool training)
   */
  void forwarding(bool training = true) override;

  /**
   * @copydoc Layer::calcDerivative()
   */
  void calcDerivative() override;

  /**
   * @copydoc Layer::calcGradient()
   */
  void calcGradient() override;

  /**
   * @brief     copy layer
   * @param[in] l layer to copy
   */
  void copy(std::shared_ptr<LayerV1> l) override;

  /**
   * @brief     initialize layer
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int initialize(Manager &manager) override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override {
    return BatchNormalizationLayer::type;
  }

  /**
   * @copydoc Layer::supportInPlace()
   */
  bool supportInPlace() const override { return true; }

  using LayerV1::setProperty;

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  void setProperty(const PropertyType type,
                   const std::string &value = "") override;

  inline static const std::string type = "batch_normalization";

private:
  Tensor cvar; /**< training variance saved in bn_layer::forwarding and used in
                    bn_layer::calcDerivative */
  Tensor invstd; /**<  inversed training std for backward pass */

  Tensor deviation; /**< (input - current_average) */

  float epsilon;  /**< epsilon */
  float momentum; /**< momentum */
  int axis;       /**< Target axis, axis inferred at initialize when -1 */

  std::vector<unsigned int> axes_to_reduce;      /**< target axes to reduce */
  std::array<WeightInitializer, 4> initializers; /**< weight initializers */
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __BN_LAYER_H__ */
