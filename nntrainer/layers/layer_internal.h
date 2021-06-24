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
 *
 * @details nntrainer::Layer inherits ml::train::Layer but has been ommitted to
 * disallow static_cast between nntrainer::Layer and ml::train::Layer objects.
 */
class LayerV1 {

  /** model classes can call private methods which arent exposed to public */
  friend class NeuralNetwork;
  friend class NetworkGraph;

public:
  /**
   * @brief     Constructor of Layer Class
   */
  LayerV1(
    WeightRegularizer weight_regularizer_ = WeightRegularizer::NONE,
    const float weight_regularizer_constant_ = 1.0f,
    WeightInitializer weight_initializer_ =
      WeightInitializer::WEIGHT_XAVIER_UNIFORM,
    WeightInitializer bias_initializer_ = WeightInitializer::WEIGHT_ZEROS) :
    layer_props(),
    loss(0.0f),
    weight_regularizer(weight_regularizer_),
    weight_regularizer_constant(weight_regularizer_constant_),
    weight_initializer(weight_initializer_),
    bias_initializer(bias_initializer_) {
    setNumInputs(1);
    setNumOutputs(1);
  }

  /**
   * @brief     Destructor of Layer Class
   */
  virtual ~LayerV1() = default;

  /**
   *  @brief  Move constructor of Layer.
   *  @param[in] Layer &&
   */
  LayerV1(LayerV1 &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs Layer to be moved.
   */
  virtual LayerV1 &operator=(LayerV1 &&rhs) = default;

  /**
   * @brief Get the layer type
   * @return const std::string type representation
   */
  virtual const std::string getType() const = 0;

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
   * @brief  check if this layer requires label to be passed
   * @note   if requireLabel() == true means, for now, that it is endpoint of a
   * graph(numOutlayers == 0). label will be fed to the gradient of hidden if
   * requireLabel is true
   * @todo   If we get to have a use case for requireLabel(true) but in the
   * middle of a graph, change the semantics
   *
   * @retval true requires a label when training
   * @retval false does not require a label
   */
  virtual bool requireLabel() const { return false; }

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
   *            36. split_dimension : string (type)
   *            37. return_sequences :  bool (type) - lstm
   *            39. hidden_state_activation :  string (type) - lstm
   *            40. dropout :  float (type) - drop out rate
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
    split_dimension = 36,
    return_sequences = 37,
    hidden_state_activation = 38,
    dropout = 39,
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
   * @brief     Copy Layer
   * @param[in] l Layer to be copied
   */
  virtual void copy(std::shared_ptr<LayerV1> l);

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
   * @todo remove this when name is moved to layer_node
   */
  virtual void
  export_to(Exporter &exporter,
            ExportMethods method = ExportMethods::METHOD_STRINGVECTOR) const {
    exporter.saveResult(layer_props, method, this);
  };

  /**
   * @brief  get the loss value added by this layer
   * @retval loss value
   */
  virtual float getLoss() { return loss; }

  /**
   * @brief  check if this layer supports backwarding
   * @note   support backwarding primarily means that the layer can process the
   * derivatives and return back the gradients to the previous layer.
   * @return true if supports backwarding, else false
   */
  virtual bool supportBackwarding() const { return true; };
  /**
   * @brief     get all weights of the layer
   * @retval    vector of all params
   */
  virtual std::vector<Weight> getWeights() { return weights; }

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
    if (input_dim.size() != size) {
      /** clear is intentional to clear any previously set input dimensions */
      input_dim.clear();
      input_dim.resize(size);
    }
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
    if (output_dim.size() != size) {
      /** clear is intentional to clear any previously set output dimensions */
      output_dim.clear();
      output_dim.resize(size);
    }
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
    if (i >= getNumInputs())
      throw std::out_of_range("Setting dimensions out of bounds");
    input_dim[i] = d;
  }

  /**
   * @brief   If the current layer can support in-place
   *
   * @return  true if inplace, else false
   * @details all layers default to out of place execution
   * @note all layers default to out of place execution
   */
  virtual bool supportInPlace() const { return false; }

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

  std::tuple<> layer_props; /**< supported properties of layer */

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

  /**
   * @brief     weight_list in this layer. This contains all weights of the
   * layer.
   */
  std::vector<Weight> weights;

private:
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
                        std::is_base_of<LayerV1, T>::value, T> * = nullptr>
std::ostream &operator<<(std::ostream &out, T &l) {
  l.printPreset(out, LayerV1::PrintPreset::PRINT_SUMMARY);
  return out;
}

using CreateLayerV1Func = nntrainer::LayerV1 *(*)();
using DestroyLayerV1Func = void (*)(nntrainer::LayerV1 *);

/**
 * @brief  Layer Pluggable struct that enables pluggable layer
 *
 */
typedef struct {
  CreateLayerV1Func createfunc;   /**< create layer function */
  DestroyLayerV1Func destroyfunc; /**< destory function */
} LayerV1Pluggable;

/**
 * @brief pluggable layer must have this structure defined
 */
extern "C" LayerV1Pluggable ml_train_layerv1_pluggable;

/**
 * @brief General Layer Factory function to register Layer
 *
 * @param props property representation
 * @return std::unique_ptr<ml::train::Layer> created object
 */
template <typename T,
          std::enable_if_t<std::is_base_of<LayerV1, T>::value, T> * = nullptr>
std::unique_ptr<LayerV1>
createLayer(const std::vector<std::string> &props = {}) {
  std::unique_ptr<LayerV1> ptr = std::make_unique<T>();

  if (ptr->setProperty(props) != ML_ERROR_NONE) {
    throw std::invalid_argument("Set properties failed for layer");
  }
  return ptr;
}

/**
 * @brief   Get Layer devel from ml::train::Layer
 *
 * @param   l Layer object
 * @return  Layer devel object
 */
std::shared_ptr<LayerV1> getLayerV1Devel(std::shared_ptr<ml::train::Layer> l);

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __LAYER_H__ */
