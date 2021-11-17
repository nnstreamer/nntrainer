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

#include <algorithm>
#include <cmath>
#include <iterator>
#include <stdexcept>
#include <utility>

#include <activation_layer.h>
#include <app_context.h>
#include <base_properties.h>
#include <common_properties.h>
#include <layer_node.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <profiler.h>
#include <time_dist.h>
#include <util_func.h>

namespace nntrainer {
static constexpr const char *FORWARD_SUFFIX = ":forward";
static constexpr const char *CALC_DERIV_SUFFIX = ":calcDeriv";
static constexpr const char *CALC_GRAD_SUFFIX = ":calcGrad";

namespace props {

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
   * @todo  detect when loss becomes Nan is useful. But it will need dedicated
   * throw
   * @param v float to validate
   * @retval true if is valid number
   * @retval false if it is nan
   */
  bool isValid(const float &v) const override {
    if (std::isnan(v)) {
      ml_logw("loss value is NAN");
    }

    return true;
  }
};

/**
 * @brief Input shape property which saves a single tensor shape
 * (practically, std::array<InputShape> is used)
 *
 */
class InputShape : public GenericShape {

public:
  static constexpr const char *key = "input_shape"; /**< unique key to access */
  using prop_tag = dimension_prop_tag;              /**< property type */
};

/**
 * @brief properties for shared from
 *
 */
class SharedFrom : public Name {
public:
  static constexpr const char *key = "shared_from"; /**< unique key to access */
  using prop_tag = str_prop_tag;                    /**< property type */
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

LayerNode::LayerNode(std::unique_ptr<nntrainer::Layer> &&l) :
  layer(std::move(l)),
  inplace(InPlace::NONE),
  needs_calc_derivative(false),
  needs_calc_gradient(false),
  output_layers(new std::vector<props::Connection>()),
  run_context(nullptr),
  layer_node_props(new PropsType(props::Name(), props::Distribute(),
                                 props::Trainable(), {}, {},
                                 props::SharedFrom())),
  layer_node_props_realization(
    new RealizationPropsType(props::Flatten(), props::Activation())),
  loss(new props::Loss()),
  regularization_loss(0.0f),
  exec_order({0, 0, 0}) {
  if (layer && layer->getType() == TimeDistLayer::type) {
    std::get<props::Distribute>(*layer_node_props).set(true);
  }
}

void LayerNode::setProperty(const std::vector<std::string> &properties) {
  auto left_properties = loadProperties(properties, *layer_node_props);
  left_properties =
    loadProperties(left_properties, *layer_node_props_realization);
  layer->setProperty(left_properties);

  if (getType() == ActivationLayer::type) {
    auto &act_prop = std::get<props::Activation>(*layer_node_props_realization);
    if (!act_prop.empty()) {
      layer->setProperty({"activation=" + to_string(act_prop)});
    }
  }
}

const unsigned LayerNode::getInputConnectionIndex(unsigned nth) const {
  auto &input_layers =
    std::get<std::vector<props::InputConnection>>(*layer_node_props);
  return input_layers.at(nth).get().getIndex();
}

const std::string &LayerNode::getInputConnectionName(unsigned nth) const {
  auto &input_layers =
    std::get<std::vector<props::InputConnection>>(*layer_node_props);
  return input_layers.at(nth).get().getName();
}

void LayerNode::setInputConnectionIndex(unsigned nth, unsigned index) {
  auto &input_layers =
    std::get<std::vector<props::InputConnection>>(*layer_node_props);
  input_layers.at(nth).get().getIndex() = index;
}

void LayerNode::setInputConnectionName(unsigned nth, const std::string &name) {
  auto &input_layers =
    std::get<std::vector<props::InputConnection>>(*layer_node_props);
  input_layers.at(nth).get().getName() = name;
}

const std::string LayerNode::getName() const noexcept {
  auto &name = std::get<props::Name>(*layer_node_props);
  return name.empty() ? "" : name.get();
}

std::ostream &operator<<(std::ostream &out, const LayerNode &l) {

  auto &input_layers =
    std::get<std::vector<props::InputConnection>>(*l.layer_node_props);

  out << "[" << l.getName() << '/' << l.getType() << "]\n";
  auto print_vector = [&out](const auto &layers, const std::string &title) {
    out << title << "[" << layers.size() << "] ";
    for (auto &layer : layers) {
      out << to_string(layer) << ' ';
    }
    out << '\n';
  };

  print_vector(input_layers, " input_layers");
  /// @todo enable this
  // print_vector(l.output_layers, "output_layers");
  return out;
}

ActivationType LayerNode::getActivationType() const {
  auto &act_prop = std::get<props::Activation>(*layer_node_props_realization);
  if (act_prop.empty()) {
    return ActivationType::ACT_NONE;
  }

  return act_prop;
}

unsigned int LayerNode::getNumInputConnections() const {
  auto &input_layers =
    std::get<std::vector<props::InputConnection>>(*layer_node_props);
  return input_layers.size();
}

unsigned int LayerNode::getNumOutputConnections() const {
  return output_layers->size();
}

const std::vector<std::string> LayerNode::getInputLayers() const {
  auto &input_layers =
    std::get<std::vector<props::InputConnection>>(*layer_node_props);
  std::vector<std::string> names;
  names.reserve(input_layers.size());
  std::transform(input_layers.begin(), input_layers.end(),
                 std::back_inserter(names),
                 [](const props::Connection &con) { return con.getName(); });
  return names;
}

const std::vector<std::string> LayerNode::getOutputLayers() const {
  std::vector<std::string> names;
  names.reserve(output_layers->size());
  std::transform(output_layers->begin(), output_layers->end(),
                 std::back_inserter(names),
                 [](const props::Connection &con) { return con.getName(); });
  return names;
}

ActivationType LayerNode::getActivationToBeRealized() const {
  if (getType() == ActivationLayer::type)
    return ActivationType::ACT_NONE;
  else
    return getActivationType();
}

const std::string LayerNode::getType() const { return getLayer()->getType(); }

bool LayerNode::getTrainable() const {
  if (run_context)
    /**
     * if a layer does not contain any weights, it will be treated as a
     * non-trainable layer.
     */
    return std::get<props::Trainable>(*layer_node_props) &
           (run_context->getNumWeights() > 0);
  else
    return std::get<props::Trainable>(*layer_node_props);
}

bool LayerNode::getFlatten() const {
  auto &flatten = std::get<props::Flatten>(*layer_node_props_realization);
  if (flatten.empty()) {
    return false;
  }
  return flatten.get();
}

std::string LayerNode::getSharedFrom() const {
  auto &shared_from = std::get<props::SharedFrom>(*layer_node_props);
  return shared_from.empty() ? "" : shared_from.get();
}

bool LayerNode::getDistribute() const {
  auto &distribute = std::get<props::Distribute>(*layer_node_props);
  if (distribute.empty()) {
    return false;
  }
  return distribute.get();
}

const nntrainer::Layer *LayerNode::getLayer() const {
  if (run_context && getDistribute())
    return static_cast<TimeDistLayer *>(layer.get())->getDistLayer();
  else
    return layer.get();
}

nntrainer::Layer *LayerNode::getLayer() {
  if (run_context && getDistribute())
    return static_cast<TimeDistLayer *>(layer.get())->getDistLayer();
  else
    return layer.get();
}

void LayerNode::updateInputLayers(const std::string &from,
                                  const std::string &to) {
  auto &input_layers =
    std::get<std::vector<props::InputConnection>>(*layer_node_props);
  for (auto &input_layer : input_layers) {
    if (istrequal(input_layer.get().getName(), from)) {
      input_layer.set({to, 0});
    }
  }
}

void LayerNode::updateInputLayers(const unsigned int idx,
                                  const std::string &to) {
  auto &input_layers =
    std::get<std::vector<props::InputConnection>>(*layer_node_props);

  input_layers.at(idx).set({to, 0});
}

void LayerNode::addInputLayers(const std::string &in_layer) {
  auto &input_layers =
    std::get<std::vector<props::InputConnection>>(*layer_node_props);
  input_layers.emplace_back(props::Connection(in_layer, 0));
}

void LayerNode::addOutputLayers(const std::string &out_layer) {
  output_layers->emplace_back(out_layer, 0);
}

void LayerNode::setInputLayers(const std::vector<std::string> &layers) {
  auto &input_layers =
    std::get<std::vector<props::InputConnection>>(*layer_node_props);
  input_layers.clear();
  input_layers.reserve(layers.size());
  std::transform(layers.begin(), layers.end(), std::back_inserter(input_layers),
                 [](const std::string &id) {
                   return props::Connection{id, 0};
                 });
}

void LayerNode::setOutputLayers(const std::vector<std::string> &layers) {
  output_layers->clear();
  output_layers->reserve(layers.size());
  std::transform(layers.begin(), layers.end(),
                 std::back_inserter(*output_layers), [](const std::string &id) {
                   return props::Connection{id, 0};
                 });
}

bool LayerNode::hasInputShapeProperty() const {
  auto &input_shapes =
    std::get<std::vector<props::InputShape>>(*layer_node_props);

  return !input_shapes.empty() &&
         std::all_of(input_shapes.begin(), input_shapes.end(),
                     [](const auto &input) { return !input.empty(); });
}

const std::vector<TensorDim> LayerNode::getInputDimensions() const {
  NNTR_THROW_IF(!run_context, std::runtime_error)
    << __func__ << " layer needs to be finalized first!";
  auto sz = run_context->getNumInputs();
  std::vector<TensorDim> dims;
  dims.reserve(sz);

  for (auto i = 0u; i < sz; ++i) {
    dims.push_back(run_context->getInput(i).getDim());
  }

  return dims;
}

const std::vector<TensorDim> LayerNode::getOutputDimensions() const {
  NNTR_THROW_IF(!run_context, std::runtime_error)
    << __func__ << " layer needs to be finalized first!";
  auto sz = run_context->getNumOutputs();
  std::vector<TensorDim> dims;
  dims.reserve(sz);

  for (auto i = 0u; i < sz; ++i) {
    dims.push_back(run_context->getOutput(i).getDim());
  }

  return dims;
}

void LayerNode::exportTo(Exporter &exporter,
                         const ExportMethods &method) const {
  exporter.saveResult(*layer_node_props, method, this);
  // TODO: update getLayer() for layerv2 and use getLayer()
  layer->exportTo(exporter, method);
  /// have layer_v2 implementation
}

void LayerNode::read(std::ifstream &file) {
  NNTR_THROW_IF(!run_context, std::runtime_error)
    << __func__ << " layer needs to be finalized first!";
  for (unsigned int i = 0; i < run_context->getNumWeights(); ++i) {
    /// @note shared weights are only be read at the first acecss
    if (run_context->isGradientLastAccess(i)) {
      run_context->getWeight(i).read(file);
    }
  }
}

void LayerNode::save(std::ofstream &file) const {
  NNTR_THROW_IF(!run_context, std::runtime_error)
    << __func__ << " layer needs to be finalized first!";
  /// @note shared weights are only be saved at the first access
  for (unsigned int i = 0; i < run_context->getNumWeights(); ++i) {
    if (run_context->isGradientLastAccess(i)) {
      run_context->getWeight(i).save(file);
    }
  }
}

/**
 * @brief     Finalize creating the layer node
 */
InitLayerContext LayerNode::finalize(const std::vector<TensorDim> &input_dims) {
  /** Create init context right before finalize */
  if (run_context)
    throw std::runtime_error("Finalizing a layer which is already finalized");

  std::vector<TensorDim> actual_input_dims;
  auto &prop_dims = std::get<std::vector<props::InputShape>>(*layer_node_props);
  auto &prop_in_layers =
    std::get<std::vector<props::InputConnection>>(*layer_node_props);

  /** prepare input dimensions */
  if (!input_dims.empty()) {
    actual_input_dims = input_dims;
    if (hasInputShapeProperty()) {
      std::vector<TensorDim> actual_prop_dims(prop_dims.begin(),
                                              prop_dims.end());
      /// if prop_dims exist, check if it's same with given input_dims
      NNTR_THROW_IF(input_dims != actual_prop_dims, std::invalid_argument)
        << "calculated input dimension is different from given input_shape "
           "property";
    }
  } else {
    NNTR_THROW_IF(!hasInputShapeProperty(), std::invalid_argument)
      << "if input dims not given, input shapes must be given by the user as "
         "property";
    /// arguably, below check can go away
    NNTR_THROW_IF((prop_dims.size() != prop_in_layers.size()) &&
                    (prop_dims.size() != 1 || !prop_in_layers.empty()),
                  std::invalid_argument)
      << "input shapes must be one if connection is not given but given "
         "dimesions size of: "
      << prop_dims.size();
    actual_input_dims =
      std::vector<TensorDim>(prop_dims.begin(), prop_dims.end());
  }

  NNTR_THROW_IF(actual_input_dims.size() < getNumInputConnections(),
                std::invalid_argument)
    << "number of input dimensions must be equal or larger "
    << "than number of input connections, node name: " << getName()
    << " num input dims: " << input_dims.size()
    << " num connections: " << getNumInputConnections();

  /** Create init context right before finalize */
  if (run_context)
    throw std::runtime_error("Finalizing a layer which is already finalized");

  /** manipulate layers if required */
  if (getType() != TimeDistLayer::type && getDistribute()) {
    std::unique_ptr<TimeDistLayer> dlayer(new TimeDistLayer());
    NNTR_THROW_IF(!dlayer, std::invalid_argument)
      << "Error creating time distribution layer";
    dlayer->setDistLayer(std::move(layer));
    layer = std::move(dlayer);
  }

  /// remove flatten and activation since it's already realized
  layer_node_props_realization = std::make_unique<RealizationPropsType>(
    props::Flatten(), props::Activation());

  auto num_outputs = output_layers->size();
  if (output_layers->empty()) {
    num_outputs = 1;
  }

  auto scope = getSharedFrom().empty() ? getName() : getSharedFrom();
  auto init_context =
    InitLayerContext(actual_input_dims, num_outputs,
                     executeInPlace() != InPlace::NONE, getName(), scope);

  layer->finalize(init_context);

#ifdef PROFILE
  auto profile_name = [this](const char *suffix) {
    return getName() + suffix + "(" + getType() + ")";
  };
#endif

  REGISTER_EVENT(profile_name(FORWARD_SUFFIX), forward_event_key);
  REGISTER_EVENT(profile_name(CALC_DERIV_SUFFIX), calc_deriv_event_key);
  REGISTER_EVENT(profile_name(CALC_GRAD_SUFFIX), calc_grad_event_key);

  return init_context;
}

/**
 * @brief     Forward Propagation of a layer
 */
void LayerNode::forwarding(bool training) {
  loss->set(run_context->getRegularizationLoss());
  START_PROFILE(forward_event_key);
  layer->forwarding(*run_context, training);
  END_PROFILE(forward_event_key);

#ifdef DEBUG
  if (!run_context->validate(getNumInputConnections() == 0, !requireLabel()))
    throw std::runtime_error("Running forwarding() layer " + getName() +
                             " invalidated the context.");
#endif

  /** add loss only for loss layers */
  if (requireLabel())
    loss->set(*loss + run_context->getLoss());
}

/**
 * @brief     calc the derivative to be passed to the previous layer
 */
void LayerNode::calcDerivative() {
  START_PROFILE(calc_deriv_event_key);
  layer->calcDerivative(*run_context);
  END_PROFILE(calc_deriv_event_key);

#ifdef DEBUG
  if (!run_context->validate(getNumInputConnections() == 0, !requireLabel()))
    throw std::runtime_error("Running calcDerivative() layer " + getName() +
                             " invalidated the context.");
#endif
}

/**
 * @brief     Calculate the derivative of a layer
 */
void LayerNode::calcGradient() {
  START_PROFILE(calc_grad_event_key);
  if (needs_calc_gradient)
    layer->calcGradient(*run_context);
  END_PROFILE(calc_grad_event_key);

#ifdef DEBUG
  if (!run_context->validate(getNumInputConnections() == 0, !requireLabel()))
    throw std::runtime_error("Running calcGradient() layer " + getName() +
                             " invalidated the context.");
#endif
}

/**
 * @brief Set the batch for the layer
 */
void LayerNode::setBatch(unsigned int batch) {
  /** @todo we won't going to need Layer::setBatch(InitLayerContext), remove it
   */
  if (hasInputShapeProperty()) {
    auto &input_shapes =
      std::get<std::vector<props::InputShape>>(*layer_node_props);
    for (auto &input_shape : input_shapes) {
      input_shape.get().batch(batch);
    }
  }

  if (run_context) {
    run_context->setBatch(batch);
    getLayer()->setBatch(*run_context, batch);
  }
}

/**
 * @brief   If the current layer can support in-place
 */
bool LayerNode::supportInPlace() const {
  ///@note below is a quick fix, we need to have a guard that this shouldn't be
  /// query until realizeProps has been finalized ( which means we will need
  /// another end point to fixate this property )
  if (getDistribute()) {
    return false;
  }
  return layer->supportInPlace();
}

/**
 * @brief  check if this layer requires label to be passed
 */
bool LayerNode::requireLabel() const { return getLayer()->requireLabel(); }

/**
 * @brief     get loss for the layer
 * @return    loss of the layer
 */
float LayerNode::getLoss() const { return *loss; }

void LayerNode::configureRunContext(const std::vector<Weight *> &weights,
                                    const std::vector<Var_Grad *> &inputs,
                                    const std::vector<Var_Grad *> &outputs,
                                    const std::vector<Var_Grad *> &tensors) {
  run_context = std::make_unique<RunLayerContext>(
    getName(), getTrainable(), 0.0f, executeInPlace() != InPlace::NONE, weights,
    inputs, outputs, tensors);
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

void LayerNode::remapIdentifiers(std::function<void(std::string &)> remap_fn) {
  NNTR_THROW_IF(isFinalized(), std::invalid_argument)
    << "cannot remap identifiers after finalized";
  auto &name = std::get<props::Name>(*layer_node_props);
  if (!name.empty()) {
    remap_fn(name.get());
  }

  auto &shared_from = std::get<props::SharedFrom>(*layer_node_props);
  if (!shared_from.empty()) {
    remap_fn(shared_from.get());
  }

  /** remap connections without touching index */
  remapConnections(
    [&remap_fn](std::string &name, unsigned &_) { remap_fn(name); });
}

void LayerNode::remapConnections(
  std::function<void(std::string &, unsigned &)> remap_fn) {
  NNTR_THROW_IF(isFinalized(), std::invalid_argument)
    << "cannot remap identifiers after finalized";

  auto &input_layers =
    std::get<std::vector<props::InputConnection>>(*layer_node_props);

  for (auto &input_layer : input_layers) {
    auto &name = input_layer.get().getName();
    auto &idx = input_layer.get().getIndex();
    remap_fn(name, idx);
  }

  for (auto &output_layer : *output_layers) {
    auto &name = output_layer.getName();
    auto &idx = output_layer.getIndex();
    remap_fn(name, idx);
  }
}

std::unique_ptr<LayerNode> LayerNode::cloneConfiguration() {
  NNTR_THROW_IF(isFinalized(), std::invalid_argument)
    << "It is prohibited to clone configuration";
  Exporter e;
  exportTo(e, ExportMethods::METHOD_STRINGVECTOR);
  e.saveResult(*layer_node_props_realization,
               ExportMethods::METHOD_STRINGVECTOR, this);
  auto props = e.getResult<ExportMethods::METHOD_STRINGVECTOR>();

  std::vector<std::string> key_val_props;
  key_val_props.reserve(props->size());
  for (auto &entry : *props) {
    key_val_props.push_back(entry.first + "=" + entry.second);
  }

  return createLayerNode(getType(), key_val_props);
}

void LayerNode::printShapeInfo(std::ostream &out) {
  for (unsigned int idx = 0; idx < getNumInputs(); ++idx) {
    out << "input " << run_context->getInput(idx).getDim();
  }
  for (unsigned int idx = 0; idx < getNumWeights(); idx++) {
    out << "weight " << run_context->getWeight(idx).getDim();
  }
  for (unsigned int idx = 0; idx < getNumOutputs(); ++idx) {
    out << "output " << run_context->getOutput(idx).getDim();
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
    if (run_context) {
      out << "======shape information: " << std::endl;
      printShapeInfo(out);
    }
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
    if (run_context) {
      out << "======weights: " << std::endl;
      for (unsigned int idx = 0; idx < getNumWeights(); idx++) {
        out << run_context->getWeight(idx);
      }
    }
  }

  if (flags & PRINT_METRIC) {
    out << "======metrics: " << std::endl;
    printMetric(out);
  }
};
}; // namespace nntrainer
