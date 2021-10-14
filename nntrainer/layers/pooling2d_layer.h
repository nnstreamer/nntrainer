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
  Pooling2DLayer(const std::array<unsigned int, POOLING2D_DIM * 2> &padding_ = {
                   0, 0, 0, 0});

  /**
   * @brief     Destructor of Pooling 2D Layer
   */
  ~Pooling2DLayer() = default;

  /**
   *  @brief  Move constructor of Pooling 2D Layer.
   *  @param[in] Pooling2D &&
   */
  Pooling2DLayer(Pooling2DLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs Pooling2DLayer to be moved.
   */
  Pooling2DLayer &operator=(Pooling2DLayer &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(RunLayerContext &context) override;

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ExportMethods method)
   */
  void exportTo(Exporter &exporter, const ExportMethods &method) const override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return Pooling2DLayer::type; };

  /**
   * @copydoc Layer::supportBackwarding()
   */
  bool supportBackwarding() const { return true; }

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  inline static const std::string type = "pooling2d";

  /**
   * @copydoc Layer::setBatch(RunLayerContext &context, unsigned int batch)
   */
  void setBatch(RunLayerContext &context, unsigned int batch) override {
    context.updateTensor(pool_helper_idx, batch);
    props::PoolingTypeInfo::Enum pooling_type =
      std::get<props::PoolingType>(pooling2d_props).get();
    if (pooling_type == props::PoolingTypeInfo::Enum::global_max)
      pool_helper_size.resize(batch * context.getInput(0).channel());
  }

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
  void pooling2d(Tensor &in, bool training, Tensor &output, Tensor &pool_helper,
                 int batch_idx);
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __POOLING_LAYER_H__ */
