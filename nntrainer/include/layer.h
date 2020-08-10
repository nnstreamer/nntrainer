/**
 * Copyright (C) 2019 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @file	layer.h
 * @date	04 December 2019
 * @brief	This is Layer classes of Neural Network
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */
#ifndef __LAYERS_H__
#define __LAYERS_H__
#ifdef __cplusplus

#include <fstream>
#include <iostream>
#include <memory>
#include <optimizer.h>
#include <set>
#include <tensor.h>
#include <tensor_dim.h>
#include <vector>

namespace nntrainer {

/**
 * @brief     Enumeration of cost(loss) function type
 *            0. MSE ( Mean Squared Error )
 *            1. ENTROPY ( Cross Entropy )
 *            2. ENTROPY_SIGMOID (Cross Entropy amalgamated with sigmoid for
 * stability)
 *            3. ENTROPY_SOFTMAX (Cross Entropy amalgamated with softmax for
 * stability)
 *            4. Unknown
 */
typedef enum {
  COST_MSE,
  COST_ENTROPY,
  COST_ENTROPY_SIGMOID,
  COST_ENTROPY_SOFTMAX,
  COST_UNKNOWN
} CostType;

/**
 * @brief     Enumeration of activation function type
 *            0. tanh
 *            1. sigmoid
 *            2. relu
 *            3. softmax
 *            4. none
 *            5. Unknown
 */
typedef enum {
  ACT_TANH,
  ACT_SIGMOID,
  ACT_RELU,
  ACT_SOFTMAX,
  ACT_NONE,
  ACT_UNKNOWN
} ActiType;

/**
 * @brief     Enumeration of layer type
 *            0. Input Layer type
 *            1. Fully Connected Layer type
 *            2. Batch Normalization Layer type
 *            3. Convolution 2D Layer type
 *            4. Pooling 2D Layer type
 *            5. Flatten Layer type
 *            6. Loss Layer type
 *            7. Activation Layer type
 *            8. Unknown
 */
typedef enum {
  LAYER_IN,
  LAYER_FC,
  LAYER_BN,
  LAYER_CONV2D,
  LAYER_POOLING2D,
  LAYER_FLATTEN,
  LAYER_ACTIVATION,
  LAYER_LOSS,
  LAYER_UNKNOWN
} LayerType;

/**
 * @brief     Enumeration of Weight Initialization Type
 *            0. WEIGHT_LECUN_NORMAL ( LeCun normal initialization )
 *            1. WEIGHT_LECUN_UNIFORM (LeCun uniform initialization )
 *            2. WEIGHT_XAVIER_NORMAL ( Xavier normal initialization )
 *            3. WEIGHT_XAVIER_UNIFORM ( Xavier uniform initialization )
 *            4. WEIGHT_HE_NORMAL ( He normal initialization )
 *            5. WEIGHT_HE_UNIFORM ( He uniform initialization )
 */
typedef enum {
  WEIGHT_LECUN_NORMAL,
  WEIGHT_LECUN_UNIFORM,
  WEIGHT_XAVIER_NORMAL,
  WEIGHT_XAVIER_UNIFORM,
  WEIGHT_HE_NORMAL,
  WEIGHT_HE_UNIFORM,
  WEIGHT_UNKNOWN
} WeightIniType;

/**
 * @brief   Print Options when printing layer info
 */
typedef enum {
  // clang-format off
  PRINT_INST_INFO  = (1 << 0), /**< Option to print type & instance address info */
  PRINT_SHAPE_INFO = (1 << 1), /**< Option to print shape information, invalid before initiation*/
  PRINT_PROP       = (1 << 2), /**< Option to print properties */
  PRINT_PROP_META  = (1 << 3), /**< Option to print properties that describe meta info
                                    e.g) layer activation type for non-activation layer. */
  PRINT_WEIGHTS    = (1 << 4), /**< Option to print weights */
  PRINT_METRIC     = (1 << 5)  /**< Option to print metrics (currently loss only) */
  // clang-format on
} LayerPrintOption;

/**
 * @class   Layer Base class for layers
 * @brief   Base class for all layers
 */
class Layer {
public:
  Layer() :
    name(std::string()),
    last_layer(false),
    bias_init_zero(false),
    type(LAYER_UNKNOWN),
    loss(0.0f),
    activation_type(ACT_NONE),
    weight_decay(),
    weight_ini_type(WEIGHT_XAVIER_UNIFORM),
    flatten(false),
    trainable(true),
    param_size(0),
    num_inputs(1),
    num_outputs(1) {}

  /**
   * @brief     Destructor of Layer Class
   */
  virtual ~Layer(){};

  /**
   *  @brief  Move constructor of Layer.
   *  @param[in] Layer &&
   */
  Layer(Layer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs Layer to be moved.
   */
  virtual Layer &operator=(Layer &&rhs) = default;

  /**
   * @brief     Forward Propagation of a layer
   * @param[in] in List of Input Tensors taken by this layer
   * @retval    List of Output Tensors
   */
  virtual sharedConstTensor forwarding(sharedConstTensor in) = 0;

  /**
   * @brief     Back Propagation of a layer
   * @param[in] in List of Derivative Tensor from the next layer
   * @param[in] iteration Iteration value for the Optimizer
   * @retval    Derivative List of Tensor for the previous layer
   */
  virtual sharedConstTensor backwarding(sharedConstTensor in,
                                        int iteration) = 0;

  /**
   * @brief     Initialize the layer
   *            - Weight(Height, Width), Bias(1, Width)
   * @param[in] last last layer
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  virtual int initialize(bool last) = 0;

  /**
   * @brief     read layer Weight & Bias data from file
   * @note      derived class can call this to get/save updatableParams
   * @param[in] file input file stream
   */
  virtual void read(std::ifstream &file);

  /**
   * @brief     save layer Weight & Bias data from file
   * @note      derived class can call this to get/save updatableParams
   * @param[in] file output file stream
   */
  virtual void save(std::ofstream &file);

  /**
   * @brief     set Property of layer
   * @param[in] values values of property
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int setProperty(std::vector<std::string> values);

  /**
   * @brief     Property Enumeration
   *            0. input shape : string
   *            1. bias zero : bool
   *            2. normalization : bool
   *            3. standardization : bool
   *            4. activation : string (type)
   *            5. epsilon : float
   *            6. weight_decay : string (type)
   *            7. weight_decay_lambda : float
   *            8. unit : int
   *            9. weight_ini : string (type)
   *            10. filter_size : int
   *            11. kernel_size : ( n , m )
   *            12. stride : ( n, m )
   *            13. padding : ( n, m )
   *            14. pooling_size : ( n,m )
   *            15. pooling : max, average, global_max, global_average
   *            16. flatten : bool
   *            17. name : string (type)
   *            18. num_inputs : unsigned int (minimum 1)
   *            19. num_outputs : unsigned int (minimum 1)
   */
  enum class PropertyType {
    input_shape = 0,
    bias_init_zero = 1,
    normalization = 2,
    standardization = 3,
    activation = 4,
    epsilon = 5,
    weight_decay = 6,
    weight_decay_lambda = 7,
    unit = 8,
    weight_ini = 9,
    filter = 10,
    kernel_size = 11,
    stride = 12,
    padding = 13,
    pooling_size = 14,
    pooling = 15,
    flatten = 16,
    name = 17,
    num_inputs = 18,
    num_outputs = 19,
    unknown = 20
  };

  /**
   * @brief setProperty by PropertyType
   * @note By passing empty string, this can validate if @a type is valid
   * @param[in] type property type to be passed
   * @param[in] value value to be passed, if empty string is passed, do nothing
   * but throws error when @a type is invalid
   * @exception exception::not_supported     when property type is not valid for
   * the particular layer
   * @exception std::invalid_argument invalid argument
   */
  virtual void setProperty(const PropertyType type,
                           const std::string &value = "");

  /**
   * @brief     Optimizer Setter
   * @param[in] opt Optimizer
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int setOptimizer(Optimizer &opt);

  /**
   * @brief     Activation Setter
   * @param[in] activation activation type
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int setActivation(ActiType activation);

  /**
   * @brief     Activation Type Getter
   * @retval    Activation Type.
   */
  ActiType getActivationType() { return this->activation_type; }

  /**
   * @brief     Layer type Setter
   * @param[in] type layer type
   */
  void setType(LayerType type) { this->type = type; }

  /**
   * @brief     Layer type Getter
   * @retval type LayerType
   */
  LayerType getType() { return type; }

  /**
   * @brief     Copy Layer
   * @param[in] l Layer to be copied
   */
  virtual void copy(std::shared_ptr<Layer> l);

  /**
   * @brief     check hyper parameter for the layer
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int checkValidation();

  /**
   * @brief     set weight decay parameters
   * @param[in] w struct for weight decay
   */
  void setWeightDecay(WeightDecayParam w) { weight_decay = w; }

  /**
   * @brief  get Tensor Dimension
   * @retval TensorDim Tensor Dimension
   */
  TensorDim &getTensorDim() { return dim; }

  /**
   * @brief  set if this is last layer of Network
   * @param[in] last true/false
   */
  void setLast(bool last) { last_layer = last; }

  /**
   * @brief  get if this is last layer of Network
   * @retval last true/false
   */
  bool getLast() { return last_layer; }

  /**
   * @brief  set bias initialize with zero
   * @param[in] zero true/false
   */
  void setBiasZero(bool zero) { bias_init_zero = zero; }

  /**
   * @brief  set Weight Initialization Type
   * @param[in] wini WeightIniType
   */
  void setWeightInit(WeightIniType wini) { weight_ini_type = wini; }

  /**
   * @brief  initialize Weight
   * @param[in] w_dim TensorDim
   * @param[in] init_type Weight Initialization Type
   * @param[out] status Status
   * @retval Tensor Initialized Tensor
   */
  Tensor initializeWeight(TensorDim w_dim, WeightIniType init_type,
                          int &status);

  void setInputDimension(TensorDim d) { input_dim = d; }

  TensorDim getOutputDimension() { return output_dim; }

  TensorDim getInputDimension() { return input_dim; }

  /**
   * @brief  get the loss value added by this layer
   * @retval loss value
   */
  float getLoss() { return loss; }

  /**
   * @brief     set trainable for this layer
   * @param[in] train to enable/disable train
   */
  void setTrainable(bool train) { trainable = train; }

  /**
   * @brief     get updatable params of all
   * @retval    vector of all params
   */
  std::shared_ptr<UpdatableParam> getParams() { return params; }

  /**
   * @brief     get if the output of this layer must be flatten
   * @retval    flatten value
   */
  void setFlatten(bool flatten) { this->flatten = flatten; }

  /**
   * @brief     get if the output of this layer must be flatten
   * @retval    flatten value
   */
  bool getFlatten() { return flatten; }

  /**
   * @brief     Set name of the layer
   */
  int setName(std::string name);

  /**
   * @brief     Get name of the layer
   */
  std::string getName() { return name; }

  /**
   * @brief     Get base name of the layer
   */
  virtual std::string getBaseName() = 0;

  /**
   * @brief     Print layer related information. Do not override without clear
   * reason. It is recommended to override printShapeInfo, printPropertiesMeta,
   * printProperties, printMetric instead
   * @param[in] out outstream
   * @param[in] flags combination of LayerPrintOption
   */
  virtual void print(std::ostream &out, unsigned int flags = 0);

  /**
   * @brief     get data alias at param position.
   * @exception std::out_of_range for index out of range
   */
  UpdatableParam &paramsAt(const unsigned int position) {
    if (position >= param_size) {
      throw std::out_of_range("index out of range");
    }

    return params.get()[position];
  }

protected:
  /**
   * @brief     Name of the layer (works as the identifier)
   */
  std::string name;

  /**
   * @brief     check if current layer's weight decay type is l2norm
   * @return    bool is weightdecay type is L2 Norm
   */
  bool isWeightDecayL2Norm() {
    return weight_decay.type == WeightDecayType::l2norm;
  }
  /**
   * @brief     Input Tensor
   */
  Tensor input;

  /**
   * @brief     Hidden Layer Tensor which store the
   *            forwading result
   */
  Tensor hidden;

  /**
   * @brief     last layer
   */
  bool last_layer;

  /**
   * @brief     Dimension of this layer
   */
  TensorDim dim;

  /**
   * @brief     Dimension of input activation
   */
  TensorDim input_dim;

  /**
   * @brief     Dimension of output activation
   */
  TensorDim output_dim;

  /**
   * @brief     Optimizer for this layer
   */
  Optimizer opt;

  /**
   * @brief     Boolean for the Bias to set zero
   */
  bool bias_init_zero;

  /**
   * @brief     Layer type
   */
  LayerType type;

  /**
   * @brief     Loss value added by this layer
   */
  float loss;

  ActiType activation_type;

  WeightDecayParam weight_decay;

  WeightIniType weight_ini_type;

  /**
   * @brief   Output of this layer should be flattened
   */
  bool flatten;

  /*
   * @brief     making this false will skip updating this layer variables
   */
  bool trainable;

  /**
   * @brief     reserve memory for @a params and set @a param_size
   * @exception std::invalid_argument when param_size is already set and
   * shouldn't be changed again.
   */
  void setParamSize(unsigned int psize) {

    /// @todo uncomment this after fixing layer test. #293
    // if (param_size > 0) {
    //   throw std::invalid_argument("param size can't be set once it is set");
    // }

    param_size = psize;
    params = std::shared_ptr<UpdatableParam>(
      new UpdatableParam[psize], std::default_delete<UpdatableParam[]>());
  }

  /**
   * @brief     updatable params in this layer. This contains params of layers.
   * @note      UpdatableParam has weights and gradients paired.
   */
  std::shared_ptr<UpdatableParam> params;

  unsigned int param_size; /**< length of UpdatableParam * params.
                                This shouldn't be changed
                                after initiation
                                use setParamSize() to avoid
                                setting parameters twice */

  /**
   * @brief   Number of inputs this layer will requries/will operate on
   */
  unsigned int num_inputs;

  /**
   * @brief   Numer of outputs this layer will produce
   */
  unsigned int num_outputs;

private:
  /**
   * @brief     Set containing all the names of layers
   */
  static std::set<std::string> layer_names;

  /**
   * @brief     Count assigned to layer names declared by default
   */
  static int def_name_count;

  /**
   * @brief     Ensure that layer has a name
   */
  void ensureName();

  /**
   * @brief check if @a type is valid and print if prop is valid to @a out
   */
  template <typename T>
  void printIfValid(std::ostream &out, const PropertyType type, T target);

  /**
   * @brief anchor point to override if PRINT_SHAPE_INFO is enabled for
   * Layer::print()
   */
  virtual void printShapeInfo(std::ostream &out);

  /**
   * @brief anchor point to override if PRINT_PROP_META is enabled for
   * Layer::print()
   */
  virtual void printPropertiesMeta(std::ostream &out);

  /**
   * @brief anchor point to override if PRINT_PROP is enabled for Layer::print()
   */
  virtual void printProperties(std::ostream &out);

  /**
   * @brief anchor point to override if PRINT_METRIC is enabled for
   * Layer::print()
   */
  virtual void printMetric(std::ostream &out);
};

/**
 * @brief   Overriding output stream for layers and it's derived class
 */
template <typename T, typename std::enable_if_t<
                        std::is_base_of<Layer, T>::value, T> * = nullptr>
std::ostream &operator<<(std::ostream &out, T &l) {
  unsigned int option = nntrainer::LayerPrintOption::PRINT_INST_INFO |
                        nntrainer::LayerPrintOption::PRINT_SHAPE_INFO |
                        nntrainer::LayerPrintOption::PRINT_PROP |
                        nntrainer::LayerPrintOption::PRINT_PROP_META;
  l.print(out, option);
  return out;
}

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __LAYERS_H__ */
