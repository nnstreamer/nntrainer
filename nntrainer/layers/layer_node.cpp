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
  layer_node_props(
    new PropsType(props::Name(), props::Flatten(), props::Distribute())) {
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
  return createLayerNode(ac.createObject<nntrainer::LayerV1>(type), properties);
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

int LayerNode::setProperty(std::vector<std::string> properties) {
  int status = ML_ERROR_NONE;
  auto left_properties = loadProperties(properties, *layer_node_props);

  /// @todo: deprecate this in favor of loadProperties
  std::vector<std::string> remainder;
  for (unsigned int i = 0; i < left_properties.size(); ++i) {
    std::string key;
    std::string value;

    status = getKeyValue(left_properties[i], key, value);
    NN_RETURN_STATUS();

    unsigned int type = parseLayerProperty(key);

    if (value.empty()) {
      ml_logd("value is empty for layer: %s, key: %s, value: %s",
              getName().c_str(), key.c_str(), value.c_str());
      return ML_ERROR_INVALID_PARAMETER;
    }

    try {
      /// @note this calls derived setProperty if available
      setProperty(static_cast<nntrainer::LayerV1::PropertyType>(type), value);
    } catch (...) {
      remainder.push_back(left_properties[i]);
    }
  }

  /// note that setting distribute is only allowed for one time.
  /// until we have layerNode::finalize and must not except timedist layer
  if (getDistribute()) {
    if (layerv1 == nullptr) {
      /// logic for layer v2
    } else {
      auto &ac = nntrainer::AppContext::Global();
      std::shared_ptr<nntrainer::LayerV1> dlayer =
        ac.createObject<nntrainer::LayerV1>(TimeDistLayer::type);
      std::static_pointer_cast<TimeDistLayer>(dlayer)->setDistLayer(layerv1);
      layerv1 = dlayer;
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

void LayerNode::setProperty(const nntrainer::LayerV1::PropertyType type,
                            const std::string &value) {
  using PropertyType = nntrainer::LayerV1::PropertyType;
  switch (type) {
  case PropertyType::input_layers:
    if (!value.empty()) {
      static const std::regex reg("\\,+");
      std::vector<std::string> split_layers = split(value, reg);
      layerv1->setNumInputs(split_layers.size());
      input_layers = split_layers;
    }
    break;
  default:
    throw std::invalid_argument("Unknown property.");
  }
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
  /// comment intended here,
  // print_vector(l.getObject()->input_layers, " input_layers");
  // print_vector(l.getObject()->output_layers, "output_layers");
  return out;
}

std::string LayerNode::getDistLayerType() const {
  if (getDistribute())
    return std::static_pointer_cast<TimeDistLayer>(layerv1)->getDistLayerType();
  else
    throw std::runtime_error(
      "Get distribution layer type for non-distributed layer");
}

ActivationType LayerNode::getActivationType() {
  return getLayer()->getActivationType();
}

const std::string LayerNode::getType() const { return getLayer()->getType(); }

std::shared_ptr<nntrainer::LayerV1> &LayerNode::getObject() {
  return getLayer();
}

const std::shared_ptr<nntrainer::LayerV1> &LayerNode::getObject() const {
  return getLayer();
}

bool LayerNode::getTrainable() const noexcept {
  return getLayer()->getTrainable();
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
  if (finalized)
    layer->setBatch(run_context, batch);
  else
    layer->setBatch(init_context, batch);
}

/**
 * @brief   If the current layer can support in-place
 */
bool LayerNode::supportInPlace() const { return layer->supportInPlace(); }

/**
 * @brief  check if this layer requires label to be passed
 */
bool LayerNode::requireLabel() const { return layer->requireLabel(); }

}; // namespace nntrainer
