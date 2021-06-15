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

#include <layer_internal.h>
#include <tensor.h>

namespace nntrainer {

/**
 * @class   EmbeddingLayer
 * @brief   EmbeddingLayer
 */
class EmbeddingLayer : public LayerV1 {
public:
  /**
   * @brief     Constructor of Embedding Layer
   */
  template <typename... Args>
  EmbeddingLayer(unsigned int in_dim_ = 0, unsigned int out_dim_ = 0,
                 unsigned int in_length_ = 0, Args... args) :
    LayerV1(args...),
    in_dim(in_dim_),
    out_dim(out_dim_),
    in_length(in_length_) {}

  /**
   * @brief     Destructor of Embedding Layer
   */
  ~EmbeddingLayer(){};

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
  const std::string getType() const override { return EmbeddingLayer::type; };

  using LayerV1::setProperty;

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  void setProperty(const PropertyType type,
                   const std::string &value = "") override;

  static const std::string type;

private:
  unsigned int in_dim;
  unsigned int out_dim;
  unsigned int in_length;
};
} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __EMBEDDING_H__ */
