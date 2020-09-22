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

#include <memory>
#include <set>
#include <vector>

#include <optimizer.h>
#include <tensor.h>
#include <tensor_dim.h>

namespace nntrainer {

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
 *            8. Addition Layer type
 *            9. Unknown
 */
typedef enum {
  LAYER_IN,
  LAYER_FC,
  LAYER_BN,
  LAYER_CONV2D,
  LAYER_POOLING2D,
  LAYER_FLATTEN,
  LAYER_ACTIVATION,
  LAYER_ADDITION,
  LAYER_LOSS,
  LAYER_UNKNOWN
} LayerType;

/**
 * @brief     Enumeration of Weight Initialization Type
 *            0. WEIGHT_ZEROS ( Zero initialization )
 *            1. WEIGHT_ONES ( One initialization )
 *            2. WEIGHT_LECUN_NORMAL ( LeCun normal initialization )
 *            3. WEIGHT_LECUN_UNIFORM (LeCun uniform initialization )
 *            4. WEIGHT_XAVIER_NORMAL ( Xavier normal initialization )
 *            5. WEIGHT_XAVIER_UNIFORM ( Xavier uniform initialization )
 *            6. WEIGHT_HE_NORMAL ( He normal initialization )
 *            7. WEIGHT_HE_UNIFORM ( He uniform initialization )
 */
typedef enum {
  WEIGHT_ZEROS,
  WEIGHT_ONES,
  WEIGHT_LECUN_NORMAL,
  WEIGHT_LECUN_UNIFORM,
  WEIGHT_XAVIER_NORMAL,
  WEIGHT_XAVIER_UNIFORM,
  WEIGHT_HE_NORMAL,
  WEIGHT_HE_UNIFORM,
  WEIGHT_UNKNOWN
} WeightInitializer;

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
    type(LAYER_UNKNOWN),
    loss(0.0f),
    activation_type(ACT_NONE),
    weight_regularizer(),
    weight_initializer(WEIGHT_XAVIER_UNIFORM),
    bias_initializer(WEIGHT_ZEROS),
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
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  virtual int initialize() = 0;

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
   *            6. weight_regularizer : string (type)
   *            7. weight_regularizer_constant : float
   *            8. unit : int
   *            9. weight_initializer : string (type)
   *            10. filter_size : int
   *            11. kernel_size : ( n , m )
   *            12. stride : ( n, m )
   *            13. padding : ( n, m )
   *            14. pool_size : ( n,m )
   *            15. pooling : max, average, global_max, global_average
   *            16. flatten : bool
   *            17. name : string (type)
   *            18. num_inputs : unsigned int (minimum 1)
   *            19. num_outputs : unsigned int (minimum 1)
   *            20. batch_size : unsigned int (minimum 1)
   *            21. momentum : float,
   *            22. moving_mean_initializer : string (type),
   *            23. moving_variance_initializer : string (type),
   *            24. gamma_initializer : string (type),
   *            25. beta_initializer" : string (type)
   */
  enum class PropertyType {
    input_shape = 0,
    normalization = 1,
    standardization = 2,
    activation = 3,
    epsilon = 4,
    weight_regularizer = 5,
    weight_regularizer_constant = 6,
    unit = 7,
    weight_initializer = 8,
    bias_initializer = 9,
    filters = 10,
    kernel_size = 11,
    stride = 12,
    padding = 13,
    pool_size = 14,
    pooling = 15,
    flatten = 16,
    name = 17,
    num_inputs = 18,
    num_outputs = 19,
    batch_size = 20,
    momentum = 21,
    moving_mean_initializer = 22,
    moving_variance_initializer = 23,
    gamma_initializer = 24,
    beta_initializer = 25,
    unknown
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
  void setWeightRegularizer(WeightRegularizerParam w) {
    weight_regularizer = w;
  }

  /**
   * @brief  set Weight Initialization Type
   * @param[in] wini WeightInitializer
   */
  void setWeightInit(WeightInitializer wini) { weight_initializer = wini; }

  /**
   * @brief Set the input dimension
   * @param[in] d dimension to be set
   */
  void setInputDimension(TensorDim d) { input_dim = d; }

  /**
   * @brief Get the output dimension
   * @return TensorDim dimension of the output
   */
  TensorDim getOutputDimension() { return output_dim; }

  /**
   * @brief Get the input dimension
   * @return TensorDim dimension of the input
   */
  TensorDim getInputDimension() { return input_dim; }

  /**
   * @brief Set the batch for the layer
   * @param batch Batch value to be set
   * @note This denotes the maximum batch size of input. The actual batchsize
   * of the data can be smaller in case of validation or testing
   */
  void setBatch(unsigned int batch) {
    input_dim.setTensorDim(0, batch);
    output_dim.setTensorDim(0, batch);
  }

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
  std::string getName() noexcept { return name; }

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
  bool isWeightRegularizerL2Norm() {
    return weight_regularizer.type == WeightRegularizerType::l2norm;
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
   * @brief     Layer type
   */
  LayerType type;

  /**
   * @brief     Loss value added by this layer
   */
  float loss;

  ActiType activation_type;

  WeightRegularizerParam weight_regularizer;

  WeightInitializer weight_initializer; /** initializer for weights */

  WeightInitializer bias_initializer; /** initializer for bias */

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
    if (psize == param_size)
      return;

    if (param_size > 0) {
      throw std::invalid_argument("param size can't be set once it is set");
    }

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

/**
 * @brief  initialize Weight
 * @param[in] w_dim TensorDim
 * @param[in] initializer Weight Initializer
 * @param[out] status Status
 * @retval Tensor Initialized Tensor
 */
// TODO: move out
Tensor getInitializedTensor(const TensorDim &w_dim,
                            WeightInitializer initializer);

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __LAYERS_H__ */
