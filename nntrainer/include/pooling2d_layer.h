// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file	pooling2d_layer.h
 * @date	12 June 2020
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 * @brief	This is 2 Dimensional Pooling Layer Class for Neural Network
 *
 */

#ifndef __POOLING2D_LAYER_H__
#define __POOLING2D_LAYER_H__
#ifdef __cplusplus

#include <layer_internal.h>
#include <tensor.h>
#include <vector>

#define POOLING2D_DIM 2

namespace nntrainer {

/**
 * @class   Pooling 2D Layer
 * @brief   Pooling 2D Layer
 */
class Pooling2DLayer : public Layer {
public:
  enum class PoolingType {
    max = 0,
    average = 1,
    global_max = 2,
    global_average = 3,
    unknown = 4,
  };

  /**
   * @brief     Constructor of Pooling 2D Layer
   */
  template <typename... Args>
  Pooling2DLayer(
    PoolingType pooling_type_ = PoolingType::average,
    const std::array<unsigned int, POOLING2D_DIM> &pool_size_ = {0, 0},
    const std::array<unsigned int, POOLING2D_DIM> &stride_ = {1, 1},
    const std::array<unsigned int, POOLING2D_DIM> &padding_ = {0, 0},
    Args... args) :
    Layer(LayerType::LAYER_POOLING2D, args...),
    pool_size(pool_size_),
    stride(stride_),
    padding(padding_),
    pooling_type(pooling_type_) {}

  /**
   * @brief     Destructor of Pooling 2D Layer
   */
  ~Pooling2DLayer() {}

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
   * @brief     initialize layer
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int initialize();

  /**
   * @brief     Read Weight & Bias Data from file
   * @param[in] file input stream file
   */
  void read(std::ifstream &file){};

  /**
   * @brief     Save Weight & Bias Data to file
   * @param[in] file output stream file
   */
  void save(std::ofstream &file){};

  /**
   * @copydoc Layer::forwarding(sharedConstTensors in)
   */
  sharedConstTensors forwarding(sharedConstTensors in);

  /**
   * @copydoc Layer::backwarding(sharedConstTensors in, int iteration)
   */
  sharedConstTensors backwarding(sharedConstTensors in, int iteration);

  /**
   * @copydoc Layer::setBatch(unsigned int batch)
   */
  void setBatch(unsigned int batch);

  /**
   * @brief     copy layer
   * @param[in] l layer to copy
   */
  void copy(std::shared_ptr<Layer> l);

  /* TO DO : support keras type of padding */
  enum class PaddingType {
    full = 0,
    same = 1,
    valid = 2,
    unknown = 3,
  };

  /**
   * @brief     get the base name for the layer
   * @retval    base name of the layer
   */
  std::string getBaseName() { return "Pooling2D"; };

  using Layer::setProperty;

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  void setProperty(const PropertyType type, const std::string &value = "");

private:
  std::array<unsigned int, POOLING2D_DIM> pool_size;
  std::array<unsigned int, POOLING2D_DIM> stride;
  std::array<unsigned int, POOLING2D_DIM> padding;
  std::vector<unsigned int> max_idx;
  std::vector<std::vector<unsigned int>> max_idx_global;
  PoolingType pooling_type;

  /**
   * @brief     calculation convolution
   * @param[in] batch batch index
   * @param[in] in input tensor
   * @retval Tensor outoput tensor
   */
  Tensor pooling2d(unsigned int batch, Tensor &in);

  /**
   * @brief     set Pooling Type
   * @param[in] t pooling type
   */
  void setPoolingType(PoolingType t) { pooling_type = t; };

  /**
   * @brief     set Parameter Size
   * @param[in] * size : size arrary
   * @param[in] type : Property type
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int setSize(int *size, PropertyType type);
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __POOLING_LAYER_H__ */
