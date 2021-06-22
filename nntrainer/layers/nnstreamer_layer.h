// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   nnstreamer_layer.h
 * @date   26 October 2020
 * @brief  This is class to encapsulate nnstreamer as a layer of Neural Network
 * @see	   https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __NNSTREAMER_LAYER_H__
#define __NNSTREAMER_LAYER_H__
#ifdef __cplusplus

#include <layer_internal.h>
#include <nnstreamer-single.h>
#include <nnstreamer.h>
#include <tensor.h>

namespace nntrainer {

/**
 * @class   NNStreamerLayer
 * @brief   nnstreamer layer
 */
class NNStreamerLayer : public LayerV1 {
public:
  /**
   * @brief     Constructor of NNStreamer Layer
   */
  NNStreamerLayer(std::string model = "") :
    LayerV1(),
    modelfile(model),
    single(nullptr),
    in_res(nullptr),
    out_res(nullptr),
    in_data_cont(nullptr),
    out_data_cont(nullptr),
    in_data(nullptr),
    out_data(nullptr) {}

  /**
   * @brief     Destructor of NNStreamer Layer
   */
  ~NNStreamerLayer();

  /**
   * @copydoc Layer::forwarding(bool training)
   */
  void forwarding(bool training = true) override;

  /**
   * @copydoc Layer::calcDerivative()
   */
  void calcDerivative();

  /**
   * @copydoc Layer::copy(std::shared_ptr<layer> l)
   */
  void copy(std::shared_ptr<LayerV1> l);

  /**
   * @copydoc Layer::initialize()
   */
  int initialize(Manager &manager);

  /**
   * @copydoc bool supportBackwarding() const
   */
  bool supportBackwarding() const override { return false; };

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const { return NNStreamerLayer::type; };

  using LayerV1::setProperty;

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  void setProperty(const PropertyType type, const std::string &value = "");

  inline static const std::string type = "backbone_nnstreamer";

private:
  std::string modelfile;
  ml_single_h single;
  ml_tensors_info_h in_res, out_res;
  ml_tensors_data_h in_data_cont, out_data_cont;
  void *in_data, *out_data;

  /**
   * @brief     finalize the layer with the given status
   * @param[in] status status to return
   * @retval return status received as argument
   */
  int finalizeError(int status);

  /**
   * @brief    convert nnstreamer's tensor_info to nntrainer's tensor_dim
   * @param[in] out_res nnstreamer's tensor_info
   * @param[out] dim nntrainer's tensor_dim
   * @retval 0 on success, -errno on failure
   */
  static int nnst_info_to_tensor_dim(ml_tensors_info_h &out_res,
                                     TensorDim &dim);
};
} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __NNSTREAMER_LAYER_H__ */
