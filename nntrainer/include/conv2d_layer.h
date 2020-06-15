/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * SPDX-License-Identifier: Apache-2.0-only
 *
 * @file	conv2d_layer.h
 * @date	01 June 2020
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 * @brief	This is Convolution Layer Class for Neural Network
 *
 */

#ifndef __CONV2D_LAYER_H_
#define __CONV2D_LAYER_H_
#ifdef __cplusplus

#include <fstream>
#include <iostream>
#include <layer.h>
#include <optimizer.h>
#include <tensor.h>
#include <vector>

#define CONV2D_DIM 2

namespace nntrainer {

/**
 * @class   Convolution 2D Layer
 * @brief   Convolution 2D Layer
 */
class Conv2DLayer : public Layer {
public:
  /**
   * @brief     Property Enumeration
   *            0. input shape : string
   *            1. bias zero : bool
   *            2. normalization : bool
   *            3. standardization : bool
   *            4. activation : string (type)
   *            6. weight_decay : string (type)
   *            7. weight_decay_lambda : float
   *            8. weight_ini : string (type)
   *            9. filter_size : int
   *            10. kernel_size : ( n , m )
   *            11. stride : ( n, m )
   *            12, padding : valid | same
   *
   */

  enum class PropertyType {
    input_shape = 0,
    bias_zero = 1,
    normalization = 2,
    standardization = 3,
    activation = 4,
    weight_decay = 6,
    weight_decay_lambda = 7,
    weight_ini = 9,
    filter = 10,
    kernel_size = 11,
    stride = 12,
    padding = 13,
  };

  /**
   * @brief     Constructor of Conv 2D Layer
   */
  Conv2DLayer() {
    stride[0] = 1;
    stride[1] = 1;
    padding[0] = 0;
    padding[1] = 0;
    kernel_size[0] = 0;
    kernel_size[1] = 0;
    normalization = false;
    standardization = false;
    setType(LAYER_CONV2D);
  };

  /**
   * @brief     Destructor of Conv 2D Layer
   */
  ~Conv2DLayer(){};

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
  void read(std::ifstream &file);

  /**
   * @brief     Save Weight & Bias Data to file
   * @param[in] file output stream file
   */
  void save(std::ofstream &file);

  /**
   * @brief     forward propagation with input
   * @param[in] in Input Tensor from upper layer
   * @param[out] status Error Status of this function
   * @retval    Activation(W x input + B)
   */
  Tensor forwarding(Tensor in, int &status);

  /**
   * @brief     foward propagation : return Input Tensor
   *            It return Input as it is.
   * @param[in] in input Tensor from lower layer.
   * @param[in] output label Tensor.
   * @retval    return Input Tensor
   */
  Tensor forwarding(Tensor in, Tensor output, int &status);

  /**
   * @brief     back propagation
   *            Calculate dJdB & dJdW & Update W & B
   * @param[in] input Input Tensor from lower layer
   * @param[in] iteration Number of Epoch for ADAM
   * @retval    dJdB x W Tensor
   */
  Tensor backwarding(Tensor in, int iteration);

  /**
   * @brief     copy layer
   * @param[in] l layer to copy
   */
  void copy(std::shared_ptr<Layer> l);

  /**
   * @brief     set Parameter Size
   * @param[in] * size : size arrary
   * @param[in] type : Property type
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int setSize(int *size, PropertyType type);

  /**
   * @brief     set Parameter Size
   * @param[in] f filter size
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int setFilter(int f);

  /**
   * @brief     set normalization
   * @param[in] enable boolean
   */
  void setNormalization(bool enable) { this->normalization = enable; };

  /**
   * @brief     set standardization
   * @param[in] enable boolean
   */
  void setStandardization(bool enable) { this->standardization = enable; };

  /**
   * @brief     set Property of layer
   * @param[in] values values of property
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int setProperty(std::vector<std::string> values);

  /**
   * @brief     calculation convolution
   * @param[in] in input tensor
   * @param[in] kernel convolution kernel
   * @param[in] stride stride value : x, y direction
   * @param[out] status output of status
   * @retval Tensor outoput tensor
   */
  Tensor conv2d(Tensor in, Tensor kernel, unsigned int const *stride,
                int &status);

  /* TO DO : support keras type of padding */
  /* enum class PaddingType { */
  /*   full = 0, */
  /*   same = 1, */
  /*   valid = 2, */
  /*   unknown = 3, */
  /* }; */

private:
  unsigned int filter_size;
  unsigned int kernel_size[CONV2D_DIM];
  unsigned int stride[CONV2D_DIM];
  unsigned int padding[CONV2D_DIM];
  std::vector<Tensor> filters;
  std::vector<float> bias;
  bool normalization;
  bool standardization;
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __CONV2D_LAYER_H__ */
