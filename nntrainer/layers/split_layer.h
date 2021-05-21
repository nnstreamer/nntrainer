// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   split_layer.h
 * @date   21 May 2021
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Split Layer Class for Neural Network
 *
 */

#ifndef __SPLIT_LAYER_H__
#define __SPLIT_LAYER_H__
#ifdef __cplusplus

#include <layer_internal.h>
#include <tensor.h>

namespace nntrainer {

/**
 * @class   Split Layer
 * @brief   Split Layer
 */
class SplitLayer : public Layer {
public:
  /**
   * @brief     Constructor of Split Layer
   */
  template <typename... Args>
  SplitLayer(unsigned int num_output_, unsigned int split_dim = 1,
             Args... args) :
    Layer(args...),
    split_dimension(split_dim) {
    setNumOutputs(num_output_);
  }

  /**
   * @brief     Destructor of Split Layer
   */
  ~SplitLayer() = default;

  /**
   *  @brief  Move constructor of SplitLayer.
   *  @param[in] SplitLayer &&
   */
  SplitLayer(SplitLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs SplitLayer to be moved.
   */
  SplitLayer &operator=(SplitLayer &&rhs) = default;

  /**
   * @brief     initialize layer
   * @param[in] last last layer
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int initialize(Manager &manager) override;

  /**
   * @brief     Read Weight & Bias Data from file
   * @param[in] file input stream file
   */
  void read(std::ifstream &file) override{};

  /**
   * @brief     Save Weight & Bias Data to file
   * @param[in] file output stream file
   */
  void save(std::ofstream &file) override{};

  /**
   * @copydoc Layer::forwarding(bool training)
   */
  void forwarding(bool training = true) override;

  /**
   * @copydoc Layer::calcDerivative()
   */
  void calcDerivative() override;

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  void setProperty(const PropertyType type,
                   const std::string &value = "") override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return SplitLayer::type; };

  static const std::string type;

private:
  unsigned int split_dimension; /** dimension along which to split the input */
  TensorDim input_reshape_helper;  /** helper dimension to reshape input */
  TensorDim output_reshape_helper; /** helper dimension to reshape outputs */
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __SPLIT_LAYER_H__ */
