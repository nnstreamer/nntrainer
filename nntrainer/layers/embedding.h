// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   embedding.h
 * @date   04 March 2021
 * @brief  This is Embedding Layer Class of Neural Network
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __EMBEDDING_H__
#define __EMBEDDING_H__
#ifdef __cplusplus

#include <layer_impl.h>

namespace nntrainer {

/**
 * @class   EmbeddingLayer
 * @brief   EmbeddingLayer
 */
class EmbeddingLayer : public LayerImpl {
public:
  /**
   * @brief     Constructor of Embedding Layer
   */
  EmbeddingLayer(unsigned int in_dim_ = 0, unsigned int out_dim_ = 0,
                 unsigned int in_length_ = 0) :
    LayerImpl(),
    in_dim(in_dim_),
    out_dim(out_dim_),
    in_length(in_length_),
    weight_idx(0) {}

  /**
   * @brief     Destructor of Embedding Layer
   */
  ~EmbeddingLayer() = default;

  /**
   *  @brief  Move constructor.
   *  @param[in] EmbeddingLayer &&
   */
  EmbeddingLayer(EmbeddingLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs EmbeddingLayer to be moved.
   */
  EmbeddingLayer &operator=(EmbeddingLayer &&rhs) = default;

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
   * @copydoc Layer::calcGradient(RunLayerContext &context)
   */
  void calcGradient(RunLayerContext &context) override;

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ExportMethods method)
   */
  void exportTo(Exporter &exporter,
                const ExportMethods &method) const override {
    LayerImpl::exportTo(exporter, method);
  }

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return EmbeddingLayer::type; };

  /**
   * @copydoc Layer::supportBackwarding()
   */
  bool supportBackwarding() const { return true; }

  using Layer::setProperty;

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  void setProperty(const std::vector<std::string> &values) override;

  inline static const std::string type = "embedding";

private:
  unsigned int in_dim;
  unsigned int out_dim;
  unsigned int in_length;
  unsigned int weight_idx;

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
#endif /* __EMBEDDING_H__ */
