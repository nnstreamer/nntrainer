// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   tflite_layer.h
 * @date   3 November 2020
 * @brief  This is class to encapsulate tflite as a layer of Neural Network
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __TENSORFLOW_LITE_H__
#define __TENSORFLOW_LITE_H__
#ifdef __cplusplus

#include <layer_internal.h>
#include <tensor.h>

#include <tensorflow/contrib/lite/interpreter.h>
#include <tensorflow/contrib/lite/kernels/register.h>
#include <tensorflow/contrib/lite/model.h>

namespace nntrainer {

/**
 * @class   TfLiteLayer
 * @brief   Tensorflow Lite layer
 */
class TfLiteLayer : public LayerV1 {
public:
  /**
   * @brief     Constructor of NNStreamer Layer
   */
  TfLiteLayer(std::string model = "") :
    LayerV1(),
    modelfile(model),
    interpreter(nullptr),
    model(nullptr) {
    trainable = false;
  }

  /**
   * @brief     Destructor of NNStreamer Layer
   */
  ~TfLiteLayer() = default;

  /**
   * @copydoc Layer::forwarding(bool training)
   */
  void forwarding(bool training = true) override;

  /**
   * @copydoc Layer::calcDerivative()
   */
  void calcDerivative() override;

  /**
   * @copydoc Layer::copy(std::shared_ptr<layer> l)
   */
  void copy(std::shared_ptr<LayerV1> l) override;

  /**
   * @copydoc Layer::initialize()
   */
  int initialize(Manager &manager) override;

  /**
   * @copydoc Layer::setTrainable(bool train)
   */
  void setTrainable(bool train) override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return TfLiteLayer::type; };

  using LayerV1::setProperty;

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  void setProperty(const PropertyType type,
                   const std::string &value = "") override;

  static const std::string type;

private:
  std::string modelfile;
  std::unique_ptr<tflite::Interpreter> interpreter;
  std::unique_ptr<tflite::FlatBufferModel> model;

  /**
   * @brief Set the Dimensions object
   *
   * @param tensor_idx_list tensor index list
   * @param dim dimension
   * @param is_output check if output
   */
  void setDimensions(const std::vector<int> &tensor_idx_list,
                     std::vector<TensorDim> &dim, bool is_output);
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __TENSORFLOW_LITE_H__ */
