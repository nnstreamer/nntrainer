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
 * @file	layer_internal.h
 * @date	04 December 2019
 * @brief	This is Layer classes of Neural Network
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */
#ifndef __LAYER_H__
#define __LAYER_H__
#ifdef __cplusplus

#include <memory>
#include <set>
#include <vector>

#include <layer.h>
#include <optimizer_internal.h>
#include <tensor.h>
#include <tensor_dim.h>
#include <weight.h>

namespace nntrainer {

struct NetBuffers {
  Tensor var;
  Tensor grad;
};

typedef std::shared_ptr<nntrainer::NetBuffers> sharedNetBuffer;
typedef std::vector<sharedNetBuffer> sharedNetBuffers;

/**
 * @brief     Enumeration of activation function type
 */
enum class ActivationType {
  ACT_TANH,    /** tanh */
  ACT_SIGMOID, /** sigmoid */
  ACT_RELU,    /** ReLU */
  ACT_SOFTMAX, /** softmax */
  ACT_NONE,    /** no op */
  ACT_UNKNOWN  /** unknown */
};

/**
 * @class   Layer Base class for layers
 * @brief   Base class for all layers
 */
class Layer : public ml::train::Layer {

  /** model classes can call private methods which arent exposed to public */
  friend class NeuralNetwork;
  friend class ModelLoader;
  friend class NetworkGraph;

public:
  /**
   * @brief     Constructor of Layer Class
   */
  Layer(
    ActivationType activation_type_ = ActivationType::ACT_NONE,
    WeightRegularizerType weight_regularizer_ = WeightRegularizerType::unknown,
    const float weight_regularizer_constant_ = 1.0f,
    WeightInitializer weight_initializer_ =
      WeightInitializer::WEIGHT_XAVIER_UNIFORM,
    WeightInitializer bias_initializer_ = WeightInitializer::WEIGHT_ZEROS,
    bool trainable_ = true, bool flatten_ = false) :
    name(std::string()),
    loss(0.0f),
    activation_type(activation_type_),
    weight_regularizer(weight_regularizer_),
    weight_regularizer_constant(weight_regularizer_constant_),
    weight_initializer(weight_initializer_),
    bias_initializer(bias_initializer_),
    flatten(flatten_),
    trainable(trainable_),
    num_weights(0),
    num_inputs(1),
    num_outputs(1) {
    input_dim.resize(1);
    output_dim.resize(1);
  }

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
  virtual void forwarding(sharedConstTensors in = {}) = 0;

  virtual sharedConstTensors forwarding_with_val(sharedConstTensors input);

  /**
   * @brief     Back Propagation of a layer
   * @param[in] in List of Derivative Tensor from the next layer
   * @param[in] iteration Iteration value for the Optimizer
   * @retval    Derivative List of Tensor for the previous layer
   */
  virtual void backwarding(int iteration, sharedConstTensors in = {}) = 0;

  virtual sharedConstTensors backwarding_with_val(int iteration,
                                                  sharedConstTensors deriv,
                                                  sharedConstTensors in = {});

  /**
   * @brief     read layer Weight & Bias data from file
   * @note      derived class can call this to get/save weights
   * @param[in] file input file stream
   */
  virtual void read(std::ifstream &file);

  /**
   * @brief     save layer Weight & Bias data from file
   * @note      derived class can call this to get/save weights
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
  int setOptimizer(std::shared_ptr<Optimizer> opt);

  /**
   * @brief Get the Optimizer object
   *
   * @return std::shared_ptr<Optimizer> optimizer
   */
  std::shared_ptr<Optimizer> getOptimizer() { return opt; }

  /**
   * @brief     Activation Type Getter
   * @retval    Activation Type.
   */
  ActivationType getActivationType() { return this->activation_type; }

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
   * @brief Get the output dimension
   * @return TensorDim dimension of the output
   */
  std::vector<TensorDim> getOutputDimension() { return output_dim; }

  /**
   * @brief Get the input dimension
   * @return TensorDim dimension of the input
   */
  std::vector<TensorDim> getInputDimension() { return input_dim; }

  /**
   * @brief  get the loss value added by this layer
   * @retval loss value
   */
  float getLoss() { return loss; }

  /**
   * @brief     set trainable for this layer
   * @param[in] train to enable/disable train
   */
  virtual void setTrainable(bool train) { trainable = train; }

  /**
   * @brief     get trainable for this layer
   * @retval train to enable/disable train
   */
  bool getTrainable() noexcept { return trainable; }

  /**
   * @brief     get all weights of the layer
   * @retval    vector of all params
   */
  std::shared_ptr<Weight> getWeights() { return weight_list; }

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
   * @brief print using PrintPreset
   *
   * @param out oustream
   * @param preset preset to be used
   */
  void printPreset(std::ostream &out,
                   PrintPreset preset = PrintPreset::PRINT_SUMMARY);

  /**
   * @brief     get data alias at param position.
   * @exception std::out_of_range for index out of range
   */
  Weight &weightAt(const unsigned int position) {
    if (position >= num_weights) {
      throw std::out_of_range("index out of range");
    }

    return weight_list.get()[position];
  }

  /**
   * @brief Get the number of weights
   *
   * @return unsigned int number of weights
   */
  unsigned int getNumWeights() { return num_weights; }

  /**
   * @brief Set the batch for the layer
   * @param batch Batch value to be set
   */
  virtual void setBatch(unsigned int batch);

  /**
   * @brief Scale the size of this layer
   * @param scalesize Scaling factor
   * @note As the final size is going to be integer and the scalesize is float,
   * the size is rounded to integer after scaling.
   * @note Layer containing local variable must define this and update their
   * shapes correspondingly. This can be called only prior to the initialization
   * of the layer.
   * @note We can assume that scale size is a non-zero positive value.
   * @note In case the scaled size is less than 0, the size must be scaled back
   * to 1 with a warning.
   * @note The layer must be re-initialized for the new size to come to effect,
   * if the layer has already been initialized. It is recommended to re-init the
   * whole model as the neighboring layers will also need re-initialization.
   */
  virtual void scaleSize(float scalesize) noexcept {}

  /**
   * @brief Resets the input and output dimension for the layer
   * @note This does not affect the number of inputs/outputs
   */
  void resetDimension() {
    input_dim.clear();
    input_dim.resize(num_inputs);
    output_dim.clear();
    output_dim.resize(num_outputs);
  }

  std::vector<Tensor> getOutputs();

  std::vector<Tensor> getDerivatives();

#ifdef ENABLE_TEST
  void resizeNetInput(unsigned int size) { net_input.resize(size); }

  void resizeNetOutput(unsigned int size) { net_hidden.resize(size); }

  unsigned int getNumInputs() { return num_inputs; }
  unsigned int getNumOutputs() { return num_outputs; }

  void setInputBuffer(unsigned int i, std::shared_ptr<NetBuffers> n_buffer) {
    if (i >= net_input.size())
      throw std::invalid_argument("Error: exceed num_input size");
    net_input[i] = n_buffer;
  }

  void setOutputBuffer(unsigned int i, std::shared_ptr<NetBuffers> n_buffer) {
    if (i >= net_hidden.size())
      throw std::invalid_argument("Error: exceed num_input size");
    net_hidden[i] = n_buffer;
  }
#endif

protected:
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
  } PrintOption;

  /**
   * @brief     Name of the layer (works as the identifier)
   */
  std::string name;

  /**
   * @brief     check if current layer's weight decay type is l2norm
   * @return    bool is weightdecay type is L2 Norm
   */
  bool isWeightRegularizerL2Norm() {
    return weight_regularizer == WeightRegularizerType::l2norm;
  }
  /**
   * @brief     Input Tensor
   */
  Tensor input;

  std::vector<std::shared_ptr<NetBuffers>> net_input;

  /**
   * @brief     Hidden Layer Tensor which store the
   *            forwading result
   */
  Tensor hidden;
  Tensor ret_derivative; /** derivative to be returned to previous layer */

  std::vector<std::shared_ptr<NetBuffers>> net_hidden;

  /**
   * @brief     Dimension of input activation
   */
  std::vector<TensorDim> input_dim;

  /**
   * @brief     Dimension of output activation
   */
  std::vector<TensorDim> output_dim;

  /**
   * @brief     Optimizer for this layer
   */
  // TODO: fix with #630
  std::shared_ptr<Optimizer> opt;

  /**
   * @brief     Loss value added by this layer
   */
  float loss;

  ActivationType activation_type;

  WeightRegularizerType weight_regularizer;

  float weight_regularizer_constant;

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
   * @brief     reserve memory for @a weight_list and set @a num_weights
   * @exception std::invalid_argument when num_weights is already set and
   * shouldn't be changed again.
   */
  void setNumWeights(unsigned int psize) {
    if (psize == num_weights)
      return;

    if (num_weights > 0) {
      throw std::invalid_argument("param size can't be set once it is set");
    }

    num_weights = psize;
    weight_list = std::shared_ptr<Weight>(new Weight[num_weights],
                                          std::default_delete<Weight[]>());
  }

  /**
   * @brief     weight_list in this layer. This contains trainable weights of
   * layers.
   */
  std::shared_ptr<Weight> weight_list;

  unsigned int num_weights; /**< length of weights.
                                This shouldn't be changed
                                after initiation
                                use setNumWeights() to avoid
                                setting parameters twice */

  /**
   * @brief   Number of inputs this layer will requries/will operate on
   */
  unsigned int num_inputs;

  /**
   * @brief   Numer of outputs this layer will produce
   */
  unsigned int num_outputs;

  /**
   * @brief     Activation Setter
   * @param[in] activation activation type
   * @throw std::invalid_argument when ActivationType is unknown
   */
  virtual void setActivation(ActivationType activation);

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
   * @brief     input layer names
   */
  std::vector<std::string> input_layers;

  /**
   * @brief     output layer names
   */
  std::vector<std::string> output_layers;

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

  /**
   * @brief     set weight decay parameters
   * @param[in] w struct for weight decay
   */
  void setWeightRegularizer(WeightRegularizerType type) {
    weight_regularizer = type;
  }

  /**
   * @brief  set Weight Initialization Type
   * @param[in] wini WeightInitializer
   */
  void setWeightInit(WeightInitializer wini) { weight_initializer = wini; }

  /**
   * @brief     get if the output of this layer must be flatten
   * @retval    flatten value
   */
  void setFlatten(bool flatten) { this->flatten = flatten; }

  /**
   * @brief     Print layer related information. Do not override without clear
   * reason. It is recommended to override printShapeInfo, printPropertiesMeta,
   * printProperties, printMetric instead
   * @param[in] out outstream
   * @param[in] flags combination of LayerPrintOption
   */
  virtual void print(std::ostream &out, unsigned int flags = 0);

  /**
   * @brief     Initialize the layer
   *            - Weight(Height, Width), Bias(1, Width)
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  virtual int initialize() = 0;

  /**
   * @brief Set the input dimension
   * @param[in] d dimension to be set
   */
  void setInputDimension(std::vector<TensorDim> d) { input_dim = d; }

  void setInputDimension(TensorDim d, unsigned int i) { input_dim[i] = d; }
};

/**
 * @brief   Overriding output stream for layers and it's derived class
 */
template <typename T, typename std::enable_if_t<
                        std::is_base_of<Layer, T>::value, T> * = nullptr>
std::ostream &operator<<(std::ostream &out, T &l) {
  l.printPreset(out, Layer::PrintPreset::PRINT_SUMMARY);
  return out;
}

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __LAYER_H__ */
