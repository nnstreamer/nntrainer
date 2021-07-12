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

#include <layer_devel.h>

#include <tensorflow/contrib/lite/interpreter.h>
#include <tensorflow/contrib/lite/kernels/register.h>
#include <tensorflow/contrib/lite/model.h>

namespace nntrainer {

/**
 * @class   TfLiteLayer
 * @brief   Tensorflow Lite layer
 */
class TfLiteLayer : public Layer {
public:
  /**
   * @brief     Constructor of NNStreamer Layer
   */
  TfLiteLayer(std::string model = "") :
    Layer(),
    modelfile(model),
    interpreter(nullptr),
    model(nullptr) {}

  /**
   * @brief     Destructor of NNStreamer Layer
   */
  ~TfLiteLayer() = default;

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
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return TfLiteLayer::type; };

  /**
   * @copydoc Layer::supportBackwarding()
   */
  bool supportBackwarding() const { return false; }

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  void setProperty(const std::vector<std::string> &values) override;

  inline static const std::string type = "backbone_tflite";

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

  /**
   * @brief setProperty by type and value separated
   * @param[in] type property type to be passed
   * @param[in] value value to be passed
   * @exception exception::not_supported     when property type is not valid for
   * the particular layer
   * @exception std::invalid_argument invalid argument
   */
  void setProperty(const std::string &type_str, const std::string &value);
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __TENSORFLOW_LITE_H__ */
