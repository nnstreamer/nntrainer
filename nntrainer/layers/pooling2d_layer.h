// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   pooling2d_layer.h
 * @date   12 June 2020
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is 2 Dimensional Pooling Layer Class for Neural Network
 *
 */

#ifndef __POOLING2D_LAYER_H__
#define __POOLING2D_LAYER_H__
#ifdef __cplusplus

#include <tuple>
#include <vector>

#include <base_properties.h>
#include <common_properties.h>
#include <layer_devel.h>

namespace nntrainer {

constexpr const unsigned int POOLING2D_DIM = 2;

/**
 * @class   Pooling 2D Layer
 * @brief   Pooling 2D Layer
 */
class Pooling2DLayer : public Layer {
public:
  /**
   * @brief PaddingType Class
   * @todo support keras type of padding
   */
  enum class PaddingType {
    full = 0,
    same = 1,
    valid = 2,
    unknown = 3,
  };

  /**
   * @brief     Constructor of Pooling 2D Layer
   */
  NNTR_API Pooling2DLayer(
    const std::array<unsigned int, POOLING2D_DIM * 2> &padding_ = {0, 0, 0, 0});

  /**
   * @brief     Destructor of Pooling 2D Layer
   */
  NNTR_API ~Pooling2DLayer() = default;

  /**
   *  @brief  Move constructor of Pooling 2D Layer.
   *  @param[in] Pooling2D &&
   */
  NNTR_API Pooling2DLayer(Pooling2DLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs Pooling2DLayer to be moved.
   */
  NNTR_API Pooling2DLayer &operator=(Pooling2DLayer &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  NNTR_API void finalize(InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  NNTR_API void forwarding(RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  NNTR_API void calcDerivative(RunLayerContext &context) override;

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  NNTR_API void exportTo(Exporter &exporter,
                         const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::getType()
   */
  NNTR_API const std::string getType() const override {
    return Pooling2DLayer::type;
  };

  /**
   * @copydoc Layer::supportBackwarding()
   */
  NNTR_API bool supportBackwarding() const override { return true; }

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  NNTR_API void setProperty(const std::vector<std::string> &values) override;

  static constexpr const char *type = "pooling2d";

  /**
   * @copydoc Layer::setBatch(RunLayerContext &context, unsigned int batch)
   */
  NNTR_API void setBatch(RunLayerContext &context, unsigned int batch) override;

private:
  std::array<unsigned int, POOLING2D_DIM * 2> padding;
  std::tuple<props::PoolingType, std::vector<props::PoolSize>,
             std::array<props::Stride, POOLING2D_DIM>, props::Padding2D>
    pooling2d_props;

  unsigned int pool_helper_idx; /**< helper tensor idx */
  std::vector<unsigned int>
    pool_helper_size; /**< helper size for each elements in the case of
                         global_max pooling */

  /**
   * @brief     calculation convolution
   * @param[in] in input tensor (batch sliced)
   * @param[in] training check if training, if training this will memorize index
   * @param[in] output output tensor (batch sliced)
   * @param[in] pool_helper helper tensor (batch sliced)
   * @param[in] batch_idx idx of the batch
   */
  NNTR_API void pooling2d(Tensor &in, bool training, Tensor &output,
                          Tensor &pool_helper, int batch_idx);
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __POOLING_LAYER_H__ */
