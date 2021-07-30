// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   layer_node.cpp
 * @date   1 April 2021
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is the layer node for network graph
 */

#include <cmath>

#include <activation_layer.h>
#include <app_context.h>
#include <layer_node.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <parse_util.h>
#include <time_dist.h>

#include <base_properties.h>
#include <common_properties.h>

namespace nntrainer {

namespace props {
class ActivationType;

/**
 * @brief Flatten property, true if needs flatten layer afterwards
 */
class Flatten : public Property<bool> {
public:
  Flatten() : Property<bool>() {}               /**< has default value of 0 */
  static constexpr const char *key = "flatten"; /**< unique key to access */
  using prop_tag = bool_prop_tag;               /**< property type */
};

/**
 * @brief Distribute property, true if it distribute across layer
 *
 */
class Distribute : public Property<bool> {
public:
  Distribute() : Property<bool>() {}
  static constexpr const char *key = "distribute";
  using prop_tag = bool_prop_tag;
  bool isValid(const bool &v) const {
    return empty() || !get();
  } /**< distribute=true can be set strictly one time */
};

/**
 * @brief Loss property, this defines loss specification of layer
 *
 */
class Loss : public Property<float> {

public:
  /**
   * @brief Construct a new loss object with a default value 0.0
   *
   */
  Loss(float value = 0.0) : nntrainer::Property<float>(value) {}
  static constexpr const char *key = "loss"; /**< unique key to access */
  using prop_tag = float_prop_tag;           /**< property type */

  /**
   * @brief LossSpec validator
   *
   * @param v float to validate
   * @retval true if it is greater or equal than 0.0
   * @retval false if it is samller than 0.0
   */
  bool isValid(const float &v) const override { return !std::isnan(v); }
};

} // namespace props

/**
 * @brief Destroy the Layer Node object
 *
 */
LayerNode::~LayerNode() = default;

/**
 * @brief Layer factory creator with constructor
 */
std::unique_ptr<LayerNode>
createLayerNode(const ml::train::LayerType &type,
                const std::vector<std::string> &properties) {
  auto &ac = nntrainer::AppContext::Global();
  return createLayerNode(ac.createObject<nntrainer::Layer>(type), properties);
}

/**
 * @brief Layer factory creator with constructor
 */
std::unique_ptr<LayerNode>
createLayerNode(const std::string &type,
                const std::vector<std::string> &properties) {
  auto &ac = nntrainer::AppContext::Global();
  return createLayerNode(ac.createObject<nntrainer::Layer>(type), properties);
}

/**
 * @brief Layer factory creator with constructor
 */
std::unique_ptr<LayerNode>
createLayerNode(std::unique_ptr<nntrainer::Layer> &&layer,
                const std::vector<std::string> &properties) {
  auto lnode = std::make_unique<LayerNode>(std::move(layer));

  lnode->setProperty(properties);
  return lnode;
}

LayerNode::LayerNode(std::unique_ptr<nntrainer::Layer> &&l, size_t idx) :
  layer(std::move(l)),
  index(idx),
  finalized(false),
  activation_type(ActivationType::ACT_NONE),
  layer_node_props(new PropsType(props::Name(), props::Flatten(),
                                 props::Distribute(), props::Trainable(),
                                 props::Loss())) {
  if (layer && layer->getType() == TimeDistLayer::type) {
    std::get<props::Distribute>(*layer_node_props).set(true);
  }
}

void LayerNode::setProperty(const std::vector<std::string> &properties) {
  bool already_distributed =
    !std::get<props::Distribute>(*layer_node_props).empty() &&
    std::get<props::Distribute>(*layer_node_props).get();
  auto left_properties = loadProperties(properties, *layer_node_props);

  /// note that setting distribute is only allowed for one time.
  /// until we have layerNode::finalize and must not except timedist layer
  if (getDistribute() && !already_distributed) {
    auto &ac = nntrainer::AppContext::Global();
    std::unique_ptr<nntrainer::Layer> dlayer =
      ac.createObject<nntrainer::Layer>(TimeDistLayer::type);
    if (dlayer.get() == nullptr)
      throw std::invalid_argument("Error creating time distribution layer");
    auto *time_dist_layer = dynamic_cast<TimeDistLayer *>(dlayer.get());
    if (time_dist_layer == nullptr)
      throw std::invalid_argument("Error casting to time distribution layer");
    time_dist_layer->setDistLayer(std::move(layer));
    layer = std::move(dlayer);
  }

  std::vector<std::string> remainder;
  /// @todo: deprecate this in favor of loadProperties
  for (unsigned int i = 0; i < left_properties.size(); ++i) {

    std::string key;
    std::string value;
    std::stringstream ss;

    if (getKeyValue(left_properties[i], key, value) != ML_ERROR_NONE) {
      throw std::invalid_argument("Error parsing the property: " +
                                  left_properties[i]);
    }

    if (value.empty()) {
      ss << "value is empty: key: " << key << ", value: " << value;
      throw std::invalid_argument(ss.str());
    }

    /// @note this calls derived setProperty if available
    if (!setProperty(key, value)) {
      remainder.push_back(left_properties[i]);
    }
  }

  layer->setProperty(remainder);
}

bool LayerNode::setProperty(const std::string &key, const std::string &value) {
  using PropertyType = nntrainer::Layer::PropertyType;

  PropertyType type = static_cast<PropertyType>(parseLayerProperty(key));
  switch (type) {
  case PropertyType::input_shape: {
    std::vector<TensorDim> input_dim = init_context.getInputDimensions();
    if (getNumInputs() > 1) {
      throw std::invalid_argument("input_shape keyword is only for one input");
    }
    if (getNumInputs() == 0)
      input_dim.resize(1);

    TensorDim &in_dim = input_dim[0];
    if (!value.empty()) {
      unsigned int cache_batch_size = 1;
      /** cache original value of batch size */
      if (in_dim.batch()) {
        cache_batch_size = in_dim.batch();
        in_dim.batch(1);
      }
      int status = in_dim.setTensorDim(value.c_str());
      if (in_dim.batch() > 1) {
        ml_logw("Batch size set with input dimension %d is ignored."
                "Set batchsize property for the model to update batchsize.",
                in_dim.batch());
      }
      /** set back to cache value of dimension */
      in_dim.batch(cache_batch_size);
      throw_status(status);

      init_context = InitLayerContext(input_dim, init_context.getNumOutputs());
    }
  } break;
  case PropertyType::activation: {
    setActivation((ActivationType)parseType(value, TOKEN_ACTI));
    if (getType() == ActivationLayer::type) {
      ml_logi("Set property delegated to activation layer");
      return false;
    }
    break;
  }
  case PropertyType::input_layers: {
    static const std::regex reg("\\,+");
    std::vector<std::string> split_layers = split(value, reg);
    setInputLayers(split_layers);
    break;
  }
  default:
    return false;
  }

  return true;
}

const std::string LayerNode::getName() const noexcept {
  auto &name = std::get<props::Name>(*layer_node_props);
  return name.empty() ? "" : name.get();
}

std::ostream &operator<<(std::ostream &out, const LayerNode &l) {
  out << "[" << l.getName() << '/' << l.getType() << "]\n";
  auto print_vector = [&out](const std::vector<std::string> &layers,
                             const std::string &title) {
    out << title << "[" << layers.size() << "] ";
    for (auto &layer : layers) {
      out << layer << ' ';
    }
    out << '\n';
  };

  print_vector(l.input_layers, " input_layers");
  print_vector(l.output_layers, "output_layers");
  return out;
}

ActivationType LayerNode::getActivationType() const { return activation_type; }

ActivationType LayerNode::getActivationToBeRealized() const {
  if (getType() == ActivationLayer::type)
    return ActivationType::ACT_NONE;
  else
    return activation_type;
}

void LayerNode::setActivation(ActivationType activation) {
  if (activation == ActivationType::ACT_UNKNOWN) {
    throw std::invalid_argument("Error:have to specify activation function");
  }
  activation_type = activation;
}

const std::string LayerNode::getType() const { return getLayer()->getType(); }

bool LayerNode::getTrainable() const {
  return std::get<props::Trainable>(*layer_node_props);
}

bool LayerNode::getFlatten() const {
  auto &flatten = std::get<props::Flatten>(*layer_node_props);
  if (flatten.empty()) {
    return false;
  }
  return flatten.get();
}

bool LayerNode::getDistribute() const {
  auto &distribute = std::get<props::Distribute>(*layer_node_props);
  if (distribute.empty()) {
    return false;
  }
  return distribute.get();
}

const nntrainer::Layer *LayerNode::getLayer() const {
  if (getDistribute())
    return ((TimeDistLayer *)(layer.get()))->getDistLayer();
  else
    return layer.get();
}

nntrainer::Layer *LayerNode::getLayer() {
  if (getDistribute())
    return ((TimeDistLayer *)(layer.get()))->getDistLayer();
  else
    return layer.get();
}

void LayerNode::updateInputLayers(const std::string &from,
                                  const std::string &to) {
  for (unsigned int idx = 0; idx < input_layers.size(); ++idx) {
    if (istrequal(input_layers[idx], from)) {
      input_layers[idx] = to;
    }
  }
}

void LayerNode::updateInputLayers(const unsigned int idx,
                                  const std::string &to) {
  if (idx >= input_layers.size())
    throw std::out_of_range("Out of range for input_layers");
  input_layers[idx] = to;
}

void LayerNode::exportTo(Exporter &exporter,
                         const ExportMethods &method) const {
  exporter.saveResult(*layer_node_props, method, this);
  // TODO: update getLayer() for layerv2 and use getLayer()
  layer->exportTo(exporter, method);
  /// have layer_v2 implementation
}

void LayerNode::read(std::ifstream &file) {
  for (unsigned int i = 0; i < run_context.getNumWeights(); ++i) {
    run_context.getWeight(i).read(file);
  }
}

void LayerNode::save(std::ofstream &file) const {
  for (unsigned int i = 0; i < run_context.getNumWeights(); ++i) {
    run_context.getWeight(i).save(file);
  }
}

/**
 * @brief     Finalize creating the layer node
 */
void LayerNode::finalize() {
  /** Create init context right before finalize */
  if (finalized)
    throw std::runtime_error("Finalizing a layer which is already finalized");

  if (!init_context.validate())
    throw std::invalid_argument(
      "Invalid init context for finalizing the layer");

  if (layer)
    layer->finalize(init_context);
  finalized = true;
  run_context = RunLayerContext(getName());
}

/**
 * @brief     Forward Propagation of a layer
 */
void LayerNode::forwarding(bool training) {
  std::get<props::Loss>(*layer_node_props)
    .set(run_context.getRegularizationLoss());
  layer->forwarding(run_context, training);
}

/**
 * @brief     calc the derivative to be passed to the previous layer
 */
void LayerNode::calcDerivative() { layer->calcDerivative(run_context); }

/**
 * @brief     Calculate the derivative of a layer
 */
void LayerNode::calcGradient() { layer->calcGradient(run_context); }

/**
 * @brief Set the batch for the layer
 */
void LayerNode::setBatch(unsigned int batch) {
  run_context.setBatch(batch);
  init_context.setBatch(batch);

  if (finalized) {
    if (run_context.readyToUse()) {
      getLayer()->setBatch(run_context, batch);
    } else {
      /** run_context has not been created yet */
      getLayer()->setBatch(init_context, batch);
    }
  }
}

/**
 * @brief   If the current layer can support in-place
 */
bool LayerNode::supportInPlace() const { return getLayer()->supportInPlace(); }

/**
 * @brief  check if this layer requires label to be passed
 */
bool LayerNode::requireLabel() const { return getLayer()->requireLabel(); }

/**
 * @brief     get loss for the layer
 * @return    loss of the layer
 */
float LayerNode::getLoss() const {
  /** add loss only for loss layers */
  if (requireLabel())
    std::get<props::Loss>(*layer_node_props)
      .set(std::get<props::Loss>(*layer_node_props).get() +
           run_context.getLoss());

  return std::get<props::Loss>(*layer_node_props).get();
}

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

void LayerNode::printPreset(std::ostream &out, PrintPreset preset) {
  unsigned int flags = 0;

  switch (preset) {
  case PrintPreset::PRINT_ALL:
    flags = PRINT_WEIGHTS | PRINT_METRIC;
    /// fall through intended
  case PrintPreset::PRINT_SUMMARY_META:
    flags |= PRINT_PROP_META;
    /// fall through intended
  case PrintPreset::PRINT_SUMMARY:
    flags |= PRINT_INST_INFO | PRINT_SHAPE_INFO | PRINT_PROP | PRINT_PROP_META;
    break;
  case PrintPreset::PRINT_NONE:
    return;
  default:
    throw ::std::invalid_argument("undefined preset given");
  }
  print(out, flags);
}

void LayerNode::printShapeInfo(std::ostream &out) {
  for (unsigned int idx = 0; idx < init_context.getNumInputs(); ++idx) {
    out << "input " << init_context.getInputDimensions()[idx];
  }
  for (unsigned int i = 0; i < init_context.getNumWeights(); i++) {
    out << "weight" << std::get<0>(init_context.getWeightsSpec()[i]);
  }
  for (unsigned int idx = 0; idx < init_context.getNumOutputs(); ++idx) {
    out << "output " << init_context.getOutputDimensions()[idx];
  }
}

void LayerNode::printMetric(std::ostream &out) {
  out << "Layer loss value: " << getLoss();
}

void LayerNode::print(std::ostream &out, unsigned int flags) {
  /** @todo properly move print to LayerNode */
  if (flags & PRINT_INST_INFO) {
    out << "===================";
    if (getName().empty())
      printInstance(out, this);
    else
      out << "<" << getName() << ">" << std::endl;

    out << "Layer Type: " << getType() << std::endl;
  }

  if (flags & PRINT_SHAPE_INFO) {
    out << "======shape information: " << std::endl;
    printShapeInfo(out);
  }

  if (flags & PRINT_PROP_META) {
    out << "======meta properties: " << std::endl;
    /** @todo print local and layer properties with node_exporter */
  }

  if (flags & PRINT_PROP) {
    out << "======properties: " << std::endl;
    /** @todo print local and layer properties with node_exporter */
  }

  if (flags & PRINT_WEIGHTS) {
    out << "======weights: " << std::endl;
    for (unsigned int idx = 0; idx < init_context.getNumWeights(); idx++) {
      out << '[' << std::get<5>(init_context.getWeightsSpec()[idx]) << ']'
          << std::endl;
      if (run_context.readyToUse())
        out << run_context.getWeight(idx);
    }
  }

  if (flags & PRINT_METRIC) {
    out << "======metrics: " << std::endl;
    printMetric(out);
  }
};

}; // namespace nntrainer
