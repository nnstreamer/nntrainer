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

namespace nntrainer {

LayerNode::LayerNode(std::shared_ptr<nntrainer::Layer> l, size_t idx) :
  layer(l),
  index(idx),
  flatten(false),
  distribute(false),
  activation_type(ActivationType::ACT_NONE),
  props(props::Name()) {
  if (layer->getType() == TimeDistLayer::type)
    distribute = true;
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
createLayerNode(std::shared_ptr<nntrainer::Layer> layer,
                const std::vector<std::string> &properties) {
  auto lnode = std::make_unique<LayerNode>(layer);
  if (lnode->setProperty(properties) != ML_ERROR_NONE)
    throw std::invalid_argument("Error setting layer properties.");

  return lnode;
}

int LayerNode::setProperty(std::vector<std::string> properties) {
  int status = ML_ERROR_NONE;

  /// @todo: deprecate this in favor of loadProperties
  std::vector<std::string> remainder;
  for (unsigned int i = 0; i < properties.size(); ++i) {
    std::string key;
    std::string value;

    status = getKeyValue(properties[i], key, value);
    NN_RETURN_STATUS();

    unsigned int type = parseLayerProperty(key);

    if (value.empty()) {
      ml_logd("value is empty for layer: %s, key: %s, value: %s",
              getName().c_str(), key.c_str(), value.c_str());
      return ML_ERROR_INVALID_PARAMETER;
    }

    try {
      /// @note this calls derived setProperty if available
      setProperty(static_cast<nntrainer::Layer::PropertyType>(type), value);
    } catch (...) {
      remainder.push_back(properties[i]);
    }
  }

  status = getLayer()->setProperty(remainder);
  return status;
}

void LayerNode::setProperty(const nntrainer::Layer::PropertyType type,
                            const std::string &value) {
  int status = ML_ERROR_NONE;
  using PropertyType = nntrainer::Layer::PropertyType;

  switch (type) {
  case PropertyType::name:
    if (!value.empty()) {
      std::get<props::Name>(props).set(value);
    }
    break;
  case PropertyType::flatten:
    if (!value.empty()) {
      status = setBoolean(flatten, value);
      throw_status(status);
    }
    break;
  case PropertyType::distribute:
    if (!value.empty()) {
      status = setBoolean(distribute, value);
      throw_status(status);
      if (distribute) {
        auto &ac = nntrainer::AppContext::Global();
        std::shared_ptr<nntrainer::Layer> dlayer =
          ac.createObject<nntrainer::Layer>(TimeDistLayer::type);
        std::dynamic_pointer_cast<TimeDistLayer>(dlayer)->setDistLayer(layer);
        layer = dlayer;
      }
    }
    break;
  case PropertyType::input_layers:
    if (!value.empty()) {
      static const std::regex reg("\\,+");
      std::vector<std::string> split_layers = split(value, reg);

      layer->setNumInputs(split_layers.size());
      input_layers = split_layers;
    }
    break;
  case PropertyType::output_layers:
    if (!value.empty()) {
      static const std::regex reg("\\,+");
      std::vector<std::string> split_layers = split(value, reg);

      layer->setNumOutputs(split_layers.size());
      output_layers = split_layers;
    }
    break;
  default:
    throw std::invalid_argument("Unknown property.");
  }
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
  if (distribute)
    return std::dynamic_pointer_cast<TimeDistLayer>(layer)->getDistLayerType();
  else
    throw std::runtime_error(
      "Get distribution layer type for non-distributed layer");
}

ActivationType LayerNode::getActivationType() {
  return getLayer()->getActivationType();
}

const std::string LayerNode::getType() const { return getLayer()->getType(); }

std::shared_ptr<nntrainer::Layer> &LayerNode::getObject() { return getLayer(); }

const std::shared_ptr<nntrainer::Layer> &LayerNode::getObject() const {
  return getLayer();
}

bool LayerNode::getTrainable() const noexcept {
  return getLayer()->getTrainable();
}

const std::shared_ptr<nntrainer::Layer> &LayerNode::getLayer() const {
  if (distribute)
    return std::dynamic_pointer_cast<TimeDistLayer>(layer)->getDistLayer();
  else
    return layer;
}

std::shared_ptr<nntrainer::Layer> &LayerNode::getLayer() {
  if (distribute)
    return std::dynamic_pointer_cast<TimeDistLayer>(layer)->getDistLayer();
  else
    return layer;
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

}; // namespace nntrainer
