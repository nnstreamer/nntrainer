// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   loss_layer.h
 * @date   12 June 2020
 * @brief  This is Loss Layer Class of Neural Network
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __LOSS_LAYER_H__
#define __LOSS_LAYER_H__
#ifdef __cplusplus

#include <layer_devel.h>

#include <tensor.h>

namespace nntrainer {

/**
 * @class   LossLayer
 * @brief   loss layer
 */
class LossLayer : public Layer {
public:
  /**
   * @brief     Destructor of Loss Layer
   */
  virtual ~LossLayer() = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  virtual void finalize(InitLayerContext &context) override;

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  virtual void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::supportBackwarding()
   */
  virtual bool supportBackwarding() const override { return true; }

  /**
   * @copydoc Layer::requireLabel()
   */
  bool requireLabel() const override { return true; }

  /**
   * @brief set the Tensor Type for the layer
   * @param     Tensor Type : NCHW or NHWC
   */
  void setTensorType(std::array<const std::string, 2> t_type) {
    if (t_type[0].compare("NCHW") == 0 || t_type[0].compare("nchw") == 0) {
      tensor_format = ml::train::TensorDim::Format::NCHW;
    } else {
      tensor_format = ml::train::TensorDim::Format::NHWC;
    }

    nntrainer::props::TensorDataType type_;

    from_string(t_type[1], type_);

    tensor_dtype = type_;
  }

private:
  ml::train::TensorDim::Format tensor_format;
  ml::train::TensorDim::DataType tensor_dtype;

protected:
  /**
   * @brief     update loss
   * @param     context Run context to update loss in
   * @param     l Tensor data to calculate
   */
  void updateLoss(RunLayerContext &context, const Tensor &l);

  Tensor
    l; /**< loss tensor to store intermediate value to calculate loss value */
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __LOSS_LAYER_H__ */
