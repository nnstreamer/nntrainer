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

#include <layer_devel.h>
#include <nnstreamer-single.h>
#include <nnstreamer.h>

namespace nntrainer {

/**
 * @class   NNStreamerLayer
 * @brief   nnstreamer layer
 */
class NNStreamerLayer : public Layer {
public:
  /**
   * @brief     Constructor of NNStreamer Layer
   */
  NNStreamerLayer(std::string model = "") :
    Layer(),
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
  void exportTo(Exporter &exporter,
                const ExportMethods &method) const override {}

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return NNStreamerLayer::type; };

  /**
   * @copydoc Layer::supportBackwarding()
   */
  bool supportBackwarding() const { return false; }

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  void setProperty(const std::vector<std::string> &values) override;

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
   */
  void finalizeError(int status);

  /**
   * @brief     release the layer resources
   */
  void release() noexcept;

  /**
   * @brief    convert nnstreamer's tensor_info to nntrainer's tensor_dim
   * @param[in] out_res nnstreamer's tensor_info
   * @param[out] dim nntrainer's tensor_dim
   * @retval 0 on success, -errno on failure
   */
  static int nnst_info_to_tensor_dim(ml_tensors_info_h &out_res,
                                     TensorDim &dim);

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
#endif /* __NNSTREAMER_LAYER_H__ */
