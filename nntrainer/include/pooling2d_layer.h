/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * SPDX-License-Identifier: Apache-2.0-only
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

#include <fstream>
#include <iostream>
#include <layer.h>
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
  Pooling2DLayer() :
    pooling_size{0, 0},
    stride{1, 1},
    padding{0, 0},
    pooling_type(PoolingType::average) {
    setType(LAYER_POOLING2D);
  };

  /**
   * @brief     Destructor of Pooling 2D Layer
   */
  ~Pooling2DLayer(){};

  /**
   * @brief     initialize layer
   * @param[in] last last layer
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int initialize(bool last);

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
   * @brief     forward propagation with input
   * @param[in] in Input Tensor from upper layer
   * @param[out] status Error Status of this function
   * @retval     return Pooling Result
   */
  Tensor forwarding(Tensor in, int &status);

  /**
   * @brief     foward propagation : return Pooling
   * @param[in] in input Tensor from lower layer.
   * @param[in] output dummy variable
   * @param[out] status status
   * @retval    return Pooling Result
   */
  Tensor forwarding(Tensor in, Tensor output, int &status);

  /**
   * @brief     back propagation
   *            Calculate Delivatives
   * @param[in] input Input Tensor from lower layer
   * @param[in] iteration Number of Epoch
   * @retval    dJdB x W Tensor
   */
  Tensor backwarding(Tensor in, int iteration);

  /**
   * @brief     copy layer
   * @param[in] l layer to copy
   */
  void copy(std::shared_ptr<Layer> l);

  /**
   * @brief     set Property of layer
   * @param[in] values values of property
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int setProperty(std::vector<std::string> values);

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

private:
  unsigned int pooling_size[POOLING2D_DIM];
  unsigned int stride[POOLING2D_DIM];
  unsigned int padding[POOLING2D_DIM];
  std::vector<unsigned int> max_idx;
  PoolingType pooling_type;

  /**
   * @brief     calculation convolution
   * @param[in] batch batch index
   * @param[in] in input tensor
   * @param[out] status output of status
   * @retval Tensor outoput tensor
   */
  Tensor pooling2d(unsigned int batch, Tensor in, int &status);
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __POOLING_LAYER_H__ */
