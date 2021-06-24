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

#include <activation_layer.h>
#include <app_context.h>
#include <layer_factory.h>
#include <layer_node.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
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

} // namespace props

LayerNode::LayerNode(std::shared_ptr<nntrainer::LayerV1> l, size_t idx) :
  LayerNode(nullptr, l, idx) {}

LayerNode::LayerNode(std::unique_ptr<nntrainer::Layer> &&l, size_t idx) :
  LayerNode(std::move(l), nullptr, idx) {}

LayerNode::LayerNode(std::unique_ptr<nntrainer::Layer> &&layer_v2,
                     std::shared_ptr<nntrainer::LayerV1> layer_v1, size_t idx) :
  layerv1(layer_v1),
  layer(std::move(layer_v2)),
  index(idx),
  finalized(false),
  activation_type(ActivationType::ACT_NONE),
  layer_node_props(new PropsType(props::Name(), props::Flatten(),
                                 props::Distribute(), props::Trainable())) {
  if (layerv1 && layerv1->getType() == TimeDistLayer::type) {
    std::get<props::Distribute>(*layer_node_props).set(true);
  } else if (layer && layer->getType() == TimeDistLayer::type) {
    std::get<props::Distribute>(*layer_node_props).set(true);
  }
}

/**
 * @brief Destroy the Layer Node object
 *
 */
LayerNode::~LayerNode() = default;

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
createLayerNode(std::shared_ptr<nntrainer::LayerV1> layer,
                const std::vector<std::string> &properties) {
  auto lnode = std::make_unique<LayerNode>(layer);
  if (lnode->setProperty(properties) != ML_ERROR_NONE)
    throw std::invalid_argument("Error setting layer properties.");

  return lnode;
}

/**
 * @brief Layer factory creator with constructor
 */
std::unique_ptr<LayerNode>
createLayerNode(std::unique_ptr<nntrainer::Layer> &&layer,
                const std::vector<std::string> &properties) {
  auto lnode = std::make_unique<LayerNode>(std::move(layer));
  if (lnode->setProperty(properties) != ML_ERROR_NONE)
    throw std::invalid_argument("Error setting layer properties.");

  return lnode;
}

int LayerNode::setProperty(std::vector<std::string> properties) {
  int status = ML_ERROR_NONE;
  auto left_properties = loadProperties(properties, *layer_node_props);

  /// note that setting distribute is only allowed for one time.
  /// until we have layerNode::finalize and must not except timedist layer
  if (getDistribute()) {
    if (layerv1 == nullptr) {
      layerv1 = nullptr;
      /// logic for layer v2
    } else if (layerv1->getType() != TimeDistLayer::type) {
      auto &ac = nntrainer::AppContext::Global();
      std::shared_ptr<nntrainer::LayerV1> dlayer =
        ac.createObject<nntrainer::LayerV1>(TimeDistLayer::type);
      std::static_pointer_cast<TimeDistLayer>(dlayer)->setDistLayer(layerv1);
      layerv1 = dlayer;
    }
  }

  std::vector<std::string> remainder;
  /// @todo: deprecate this in favor of loadProperties
  for (unsigned int i = 0; i < left_properties.size(); ++i) {

    std::string key;
    std::string value;

    status = getKeyValue(left_properties[i], key, value);
    NN_RETURN_STATUS();

    if (value.empty()) {
      ml_logd("value is empty for layer: %s, key: %s, value: %s",
              getName().c_str(), key.c_str(), value.c_str());
      return ML_ERROR_INVALID_PARAMETER;
    }

    /// @note this calls derived setProperty if available
    if (!setProperty(key, value)) {
      remainder.push_back(left_properties[i]);
    }
  }

  if (layerv1 == nullptr) {
    layer->setProperty(remainder);
  } else {
    auto &l = getLayer();
    return l->setProperty(remainder);
  }

  return status;
}

bool LayerNode::setProperty(const std::string &key, const std::string &value) {
  using PropertyType = nntrainer::LayerV1::PropertyType;

  PropertyType type = static_cast<PropertyType>(parseLayerProperty(key));
  switch (type) {
  case PropertyType::input_shape: {
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
    layerv1->setNumInputs(split_layers.size());
    input_layers = split_layers;
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

std::string LayerNode::getDistLayerType() const {
  if (getDistribute())
    return std::static_pointer_cast<TimeDistLayer>(layerv1)->getDistLayerType();
  else
    throw std::runtime_error(
      "Get distribution layer type for non-distributed layer");
}

ActivationType LayerNode::getActivationType() const { return activation_type; }

ActivationType LayerNode::getActivationToBeRealized() const noexcept {
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

const std::string LayerNode::getType() const {
  if (layerv1)
    return getLayer()->getType();
  else
    return layer->getType();
}

std::shared_ptr<nntrainer::LayerV1> &LayerNode::getObject() {
  return getLayer();
}

const std::shared_ptr<nntrainer::LayerV1> &LayerNode::getObject() const {
  return getLayer();
}

bool LayerNode::getTrainable() const noexcept {
  return std::get<props::Trainable>(*layer_node_props);
}

bool LayerNode::getFlatten() const noexcept {
  auto &flatten = std::get<props::Flatten>(*layer_node_props);
  if (flatten.empty()) {
    return false;
  }
  return flatten.get();
}

bool LayerNode::getDistribute() const noexcept {
  auto &distribute = std::get<props::Distribute>(*layer_node_props);
  if (distribute.empty()) {
    return false;
  }
  return distribute.get();
}

const std::shared_ptr<nntrainer::LayerV1> &LayerNode::getLayer() const {
  if (getDistribute())
    return std::static_pointer_cast<TimeDistLayer>(layerv1)->getDistLayer();
  else
    return layerv1;
}

std::shared_ptr<nntrainer::LayerV1> &LayerNode::getLayer() {
  if (getDistribute())
    return std::static_pointer_cast<TimeDistLayer>(layerv1)->getDistLayer();
  else
    return layerv1;
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
  if (layerv1 == nullptr) {
    // TODO: update getLayer() for layerv2 and use getLayer()
    layer->exportTo(exporter, method);
    /// have layer_v2 implementation
  } else {
    getLayer()->export_to(exporter, method);
  }
}

void LayerNode::read(std::ifstream &file) {
  if (layerv1 == nullptr) {
    for (unsigned int i = 0; i < run_context.getNumWeights(); ++i) {
      run_context.getWeight(i).read(file);
    }
  } else {
    getLayer()->read(file);
  }
}

void LayerNode::save(std::ofstream &file) const {
  if (layerv1 == nullptr) {
    for (unsigned int i = 0; i < run_context.getNumWeights(); ++i) {
      run_context.getWeight(i).save(file);
    }
  } else {
    getLayer()->save(file);
  }
}

/**
 * @brief     Finalize creating the layer node
 */
void LayerNode::finalize() {
  /** Create init context right before finalize */
  init_context = InitLayerContext(input_dim);
#if LAYER_V2
  layer->finalize(init_context);
#endif
  finalized = true;
}

/**
 * @brief     Forward Propagation of a layer
 */
void LayerNode::forwarding(bool training) {
  if (layerv1)
    layerv1->forwarding(training);
  else
    layer->forwarding(run_context, training);
}

/**
 * @brief     calc the derivative to be passed to the previous layer
 */
void LayerNode::calcDerivative() {
  if (layerv1)
    getLayer()->calcDerivative();
  else
    layer->calcDerivative(run_context);
}

/**
 * @brief     Calculate the derivative of a layer
 */
void LayerNode::calcGradient() {
  if (layerv1)
    getLayer()->calcGradient();
  else
    layer->calcGradient(run_context);
}

/**
 * @brief Set the batch for the layer
 */
void LayerNode::setBatch(unsigned int batch) {
  if (layerv1)
    layerv1->setBatch(batch);
  else {
    if (finalized) {
      run_context.setBatch(batch);
      layer->setBatch(run_context, batch);
    } else {
      init_context.setBatch(batch);
      layer->setBatch(init_context, batch);
    }
  }
}

/**
 * @brief   If the current layer can support in-place
 */
bool LayerNode::supportInPlace() const { return layer->supportInPlace(); }

/**
 * @brief  check if this layer requires label to be passed
 */
bool LayerNode::requireLabel() const {
  if (layerv1)
    return getLayer()->requireLabel();
  else
    return layer->requireLabel();
}

}; // namespace nntrainer
