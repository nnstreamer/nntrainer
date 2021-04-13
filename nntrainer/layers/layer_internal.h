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
#include <tuple>
#include <vector>

#include <acti_func.h>
#include <common_properties.h>
#include <layer.h>
#include <manager.h>
#include <node_exporter.h>
#include <optimizer_devel.h>
#include <tensor.h>
#include <tensor_dim.h>
#include <weight.h>

namespace nntrainer {

/**
 * @class   Layer Base class for layers
 * @brief   Base class for all layers
 */
class Layer : public ml::train::Layer {

  /** model classes can call private methods which arent exposed to public */
  friend class NeuralNetwork;
  friend class NetworkGraph;

public:
  /**
   * @brief     Constructor of Layer Class
   */
  Layer(ActivationType activation_type_ = ActivationType::ACT_NONE,
        WeightRegularizer weight_regularizer_ = WeightRegularizer::NONE,
        const float weight_regularizer_constant_ = 1.0f,
        WeightInitializer weight_initializer_ =
          WeightInitializer::WEIGHT_XAVIER_UNIFORM,
        WeightInitializer bias_initializer_ = WeightInitializer::WEIGHT_ZEROS,
        bool trainable_ = true, bool flatten_ = false,
        bool distribute_ = false) :
    layer_props(props::Name()),
    loss(0.0f),
    activation_type(activation_type_),
    weight_regularizer(weight_regularizer_),
    weight_regularizer_constant(weight_regularizer_constant_),
    weight_initializer(weight_initializer_),
    bias_initializer(bias_initializer_),
    flatten(flatten_),
    trainable(trainable_),
    distribute(distribute_) {
    setNumInputs(1);
    setNumOutputs(1);
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
  virtual void forwarding(bool training = true) = 0;

  /**
   * @brief     Forward Propagation of a layer
   * @param[in] input List of Input Tensors taken by this layer
   * @param[in] label List of Label Tensors taken by this layer
   * @param[in] training if training, pass true else false. some layers have
   * diffrent behavior depending on this
   * @retval    List of Output Tensors
   */
  virtual sharedConstTensors forwarding_with_val(sharedConstTensors input,
                                                 sharedConstTensors label = {},
                                                 bool training = true);

  /**
   * @brief     calc the derivative to be passed to the previous layer
   * @retval    Derivative List of Tensor for the previous layer
   */
  virtual void calcDerivative() = 0;

  /**
   * @brief     Calculate the derivative of a layer
   */
  virtual void calcGradient(){};

  /**
   * @brief     Apply the gradient for the layer
   * @param[in] iteration Iteration value for the Optimizer
   * @param[in] optimizer Optimizer to apply the gradient
   * @note      This function is no-op if optimizer is nullptr
   */
  virtual void applyGradient(unsigned int iteration,
                             std::shared_ptr<Optimizer> optimizer) {
    if (optimizer)
      optimizer->applyGradients(weights, iteration);
  }

  /**
   * @brief     Back Propagate the derivative to the previous layer
   * @retval    Derivative List of Tensor for the previous layer
   */
  virtual void backwarding() {
    calcGradient();
    calcDerivative();
  }

  /**
   * @brief     Backward to calculate the gradient for the layer and apply it
   * @param[in] iteration Iteration value for the Optimizer
   * @param[in] deriv Derivative for the layer
   * @param[in] optimizer Optimizer to apply the gradient
   */
  virtual sharedConstTensors
  backwarding_with_val(int iteration, sharedConstTensors deriv,
                       std::shared_ptr<Optimizer> optimizer = nullptr) {
    auto ret = backwarding_with_val(deriv);
    applyGradient(iteration, optimizer);
    return ret;
  };

  /**
   * @brief     Backward to calculate the gradient for the layer
   * @param[in] deriv Derivative for the layer
   */
  virtual sharedConstTensors backwarding_with_val(sharedConstTensors deriv);

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
   * @note this shouldn't be virtual, this became virtual to support custom
   * layer. should be reverted after layer.h can fully support custom layer
   */
  virtual int setProperty(std::vector<std::string> values);

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
   *            20. momentum : float,
   *            21. moving_mean_initializer : string (type),
   *            22. moving_variance_initializer : string (type),
   *            23. gamma_initializer : string (type),
   *            24. beta_initializer" : string (type)
   *            25. modelfile : model file for loading config for backbone layer
   *            26. input_layers : string (type)
   *            27. output_layers : string (type)
   *            28. trainable :
   *            29. flip_direction
   *            30. random_translate
   *            31. in_dim : int ( input dimension for embedding layer )
   *            32. out_dim : int ( output dimesion for embedding layer )
   *            33. in_length : int ( input length for embedding layer )
   *            34. recurrent_activation :  string (type) - lstm
   *            35. distribute : bool
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
    momentum = 20,
    moving_mean_initializer = 21,
    moving_variance_initializer = 22,
    gamma_initializer = 23,
    beta_initializer = 24,
    modelfile = 25, /** model file for loading config for backbone layer */
    input_layers = 26,
    output_layers = 27,
    trainable = 28,
    flip_direction = 29,
    random_translate = 30,
    in_dim = 31,
    out_dim = 32,
    in_length = 33,
    recurrent_activation = 34,
    distribute = 35,
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
  virtual int checkValidation();

  /**
   * @brief Get the output dimension
   * @return TensorDim dimension of the output
   */
  virtual std::vector<TensorDim> getOutputDimension() { return output_dim; }

  /**
   * @brief Get the input dimension
   * @return TensorDim dimension of the input
   */
  virtual std::vector<TensorDim> getInputDimension() { return input_dim; }

  /**
   * @brief this function helps exporting the layer in a predefined format,
   * while workarounding issue caused by templated function type eraser
   *
   * @param exporter exporter that conatins exporting logic
   * @param method enum value to identify how it should be exported to
   */
  virtual void
  export_to(Exporter &exporter,
            ExportMethods method = ExportMethods::METHOD_STRINGVECTOR) const {
    exporter.save_result(layer_props, ExportMethods::METHOD_STRINGVECTOR);
  };

  /**
   * @brief  get the loss value added by this layer
   * @retval loss value
   */
  virtual float getLoss() { return loss; }

  /**
   * @brief     set trainable for this layer
   * @param[in] train to enable/disable train
   */
  virtual void setTrainable(bool train) { trainable = train; }

  /**
   * @brief     get trainable for this layer
   * @retval train to enable/disable train
   */
  virtual bool getTrainable() noexcept { return trainable; }

  /**
   * @brief     set distribute for this layer
   * @param[in] dist to enable/disable distribute
   */
  virtual void setDistribute(bool dist) { distribute = dist; }

  /**
   * @brief     get distribute for this layer
   * @retval dist to enable/disable distribute
   */
  virtual bool getDistribute() noexcept { return distribute; }

  /**
   * @brief     get all weights of the layer
   * @retval    vector of all params
   */
  virtual std::vector<Weight> getWeights() { return weights; }

  /**
   * @brief     get if the output of this layer must be flatten
   * @retval    flatten value
   */
  virtual bool getFlatten() { return flatten; }

  /**
   * @brief     Set name of the layer
   */
  virtual int setName(std::string name);

  /**
   * @brief     Get name of the layer
   */
  virtual std::string getName() noexcept {
    return std::get<props::Name>(layer_props).get();
  }

  /**
   * @brief Preset modes for printing summary for the layer
   */
  enum class PrintPreset {
    PRINT_NONE = 0,     /**< Print nothing */
    PRINT_SUMMARY,      /**< Print preset including summary information */
    PRINT_SUMMARY_META, /**< Print summary preset that includes meta information
                         */
    PRINT_ALL           /**< Print everything possible */
  };

  /**
   * @brief print using PrintPreset
   *
   * @param out oustream
   * @param preset preset to be used
   */
  virtual void printPreset(std::ostream &out,
                           PrintPreset preset = PrintPreset::PRINT_SUMMARY);

  /**
   * @brief     get data alias at param position.
   * @exception std::out_of_range for index out of range
   */
  virtual Weight &weightAt(const unsigned int position) {
    return weights[position];
  }

  /**
   * @brief Get the number of weights
   *
   * @return unsigned int number of weights
   */
  virtual unsigned int getNumWeights() { return weights.size(); }

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
  virtual void resetDimension() {
    unsigned int num_inputs = input_dim.size();
    input_dim.clear();
    input_dim.resize(num_inputs);

    unsigned int num_outputs = output_dim.size();
    output_dim.clear();
    output_dim.resize(num_outputs);
  }

  /**
   * @brief Get hidden tensors
   *
   * @return std::vector<Tensor>  get outputs
   */
  virtual std::vector<Tensor> getOutputs();

  /**
   * @brief Get derivatives tensors
   *
   * @return std::vector<Tensor> get derivatives
   */
  virtual std::vector<Tensor> getDerivatives();

  /**
   * @brief Get the Input Ref object
   *
   * @return std::vector<std::shared_ptr<Var_Grad>>&
   */
  virtual const std::vector<std::shared_ptr<Var_Grad>> &getInputRef() const {
    return net_input;
  }

  /**
   * @brief Get the Output Ref object
   *
   * @return std::vector<std::shared_ptr<Var_Grad>>&
   */
  virtual const std::vector<std::shared_ptr<Var_Grad>> &getOutputRef() const {
    return net_hidden;
  }

  /**
   * @brief Get reference to the weights
   * @retval Reference of the list of weights in the layer
   */
  virtual std::vector<Weight> &getWeightsRef() { return weights; }

  /**
   * @brief Get the Weights Ref object
   *
   * @return const std::vector<Weight>& refs of weights
   */
  virtual const std::vector<Weight> &getWeightsRef() const { return weights; }

  /**
   * @brief Set the Input Buffers object
   *
   * @param inputs inputs to set
   */
  virtual void setInputBuffers(std::vector<std::shared_ptr<Var_Grad>> inputs) {
    net_input = inputs;
  }

  /**
   * @brief Set output Buffers
   *
   * @param outputs output to set
   */
  virtual void
  setOutputBuffers(std::vector<std::shared_ptr<Var_Grad>> outputs) {
    net_hidden = outputs;
  }

  /**
   * @brief     Initialize the layer
   *            - Weight(Height, Width), Bias(1, Width)
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  virtual int initialize(Manager &manager) = 0;

  /**
   * @brief get number of input layers
   *
   * @return unsigned int input size
   */
  virtual unsigned int getNumInputs() { return input_dim.size(); }

  /**
   * @brief get number of output layers
   *
   * @return unsigned int output size
   */
  virtual unsigned int getNumOutputs() { return output_dim.size(); }

  /**
   * @brief set Number of Input Layers
   *
   * @param size size of inputs
   */
  void setNumInputs(unsigned int size) {
    if (size < 1)
      throw std::invalid_argument("Minimum number of inputs must be 1");
    input_dim.resize(size);
    net_input.resize(size);
  }

  /**
   * @brief set Number of Output Layers
   *
   * @param size size of outputs
   */
  void setNumOutputs(unsigned int size) {
    if (size < 1)
      throw std::invalid_argument("Minimum number of outputs must be 1");
    output_dim.resize(size);
    net_hidden.resize(size);
  }

  /**
   * @brief Set the input dimension
   * @param[in] d dimension to be set
   */
  void setInputDimension(const std::vector<TensorDim> &d) { input_dim = d; }

  /**
   * @brief Set the input dimension
   * @param[in] d dimension to be set
   * @param[in] i axis
   */
  void setInputDimension(const TensorDim &d, unsigned int i) {
    if (i < 0 || i > MAXDIM)
      throw std::invalid_argument(
        "axis must be greater than 0 and less then MAX_DIM : 4");
    input_dim[i] = d;
  }

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

  std::tuple<props::Name> layer_props; /**< supported properties of layer */

  /**
   * @brief     Input Tensors
   */
  std::vector<std::shared_ptr<Var_Grad>> net_input;

  /**
   * @brief Output Tensors
   */
  std::vector<std::shared_ptr<Var_Grad>> net_hidden;

  /**
   * @brief     Dimension of input activation
   */
  std::vector<TensorDim> input_dim;

  /**
   * @brief     Dimension of output activation
   */
  std::vector<TensorDim> output_dim;

  /**
   * @brief     Loss value added by this layer
   */
  float loss;

  // TODO: remove this from here
  ActivationType activation_type;

  WeightRegularizer weight_regularizer;

  float weight_regularizer_constant;

  /**
   * @brief initializer for weights
   */
  WeightInitializer weight_initializer;

  /**
   * @brief initializer for bias
   */
  WeightInitializer bias_initializer;

  // TODO: remove this from here
  /**
   * @brief   Output of this layer should be flattened
   */
  bool flatten;

  /**
   * @brief     making this false will skip updating this layer variables
   */
  bool trainable;

  /**
   * @brief     making this true will iterating along with time distribution
   */
  bool distribute;

  /**
   * @brief     weight_list in this layer. This contains all weights of the
   * layer.
   */
  std::vector<Weight> weights;

  /**
   * @brief     Activation Setter
   * @param[in] activation activation type
   * @throw std::invalid_argument when ActivationType is unknown
   */
  virtual void setActivation(ActivationType activation);

private:
  // TODO: remove this from here
  /**
   * @brief     input layer names
   */
  std::vector<std::string> input_layers;

  // TODO: remove this from here
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

using CreateLayerFunc = ml::train::Layer *(*)();
using DestroyLayerFunc = void (*)(ml::train::Layer *);

/**
 * @brief  Layer Pluggable struct that enables pluggable layer
 *
 */
typedef struct {
  CreateLayerFunc createfunc;   /**< create layer function */
  DestroyLayerFunc destroyfunc; /**< destory function */
} LayerPluggable;

/**
 * @brief pluggable layer must have this structure defined
 */
extern "C" LayerPluggable ml_train_layer_pluggable;

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __LAYER_H__ */
