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
 *
 * @file	layer.cpp
 * @date	04 December 2019
 * @brief	This is Layers Classes for Neural Network
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */
#include <ostream>
#include <sstream>

#include <layer_internal.h>
#include <layer_node.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <parse_util.h>
#include <util_func.h>

namespace nntrainer {

void Layer::setActivation(ActivationType acti) {
  if (acti == ActivationType::ACT_UNKNOWN) {
    throw std::invalid_argument("Error:have to specify activation function");
  }
  activation_type = acti;
}

int Layer::checkValidation() {
  int status = ML_ERROR_NONE;

  if (activation_type == ActivationType::ACT_UNKNOWN) {
    ml_loge("Error: Have to set activation for this layer");
    return ML_ERROR_INVALID_PARAMETER;
  }

  return status;
}

void Layer::setBatch(unsigned int batch) {
  for (unsigned int idx = 0; idx < getNumInputs(); ++idx)
    input_dim[idx].setTensorDim(0, batch);

  for (unsigned int idx = 0; idx < getNumOutputs(); ++idx)
    output_dim[idx].setTensorDim(0, batch);
}

std::vector<Tensor> Layer::getOutputs() {
  std::vector<Tensor> ret;
  for (unsigned int i = 0; i < getNumOutputs(); ++i) {
    ret.push_back(net_hidden[i]->getVariableRef());
  }
  return ret;
}

std::vector<Tensor> Layer::getDerivatives() {
  std::vector<Tensor> ret;
  for (unsigned int i = 0; i < getNumInputs(); ++i) {
    ret.push_back(net_input[i]->getGradientRef());
  }
  return ret;
}

void Layer::copy(std::shared_ptr<Layer> l) {
  for (auto const &w : l->weights)
    weights.push_back(w.clone());

  this->input_dim = l->input_dim;
  this->output_dim = l->output_dim;
  this->activation_type = l->activation_type;
  this->loss = l->loss;
  this->weight_regularizer = l->weight_regularizer;
  this->weight_regularizer_constant = l->weight_regularizer_constant;
  this->weight_initializer = l->weight_initializer;
  this->trainable = l->trainable;
}

sharedConstTensors Layer::forwarding_with_val(sharedConstTensors input,
                                              sharedConstTensors label,
                                              bool training) {

  if (getNumInputs() != input.size()) {
    std::stringstream ss;
    ss << "Number of inputs mismatched, given: " << input.size()
       << " expected: " << getNumInputs();
    throw std::invalid_argument(ss.str().c_str());
  }

  for (unsigned int i = 0; i < getNumInputs(); ++i) {
    net_input[i]->getVariableRef() = input[i]->clone();
  }

  if (!label.empty()) {
    for (unsigned int i = 0; i < getNumOutputs(); ++i) {
      net_hidden[i]->getGradientRef() = label[i]->clone();
    }
  }

  forwarding(training);

  nntrainer::sharedConstTensors out;

  for (unsigned int i = 0; i < getNumOutputs(); ++i) {
    out.push_back(MAKE_SHARED_TENSOR(net_hidden[i]->getVariable()));
  }

  return out;
}

sharedConstTensors Layer::backwarding_with_val(sharedConstTensors label) {

  for (unsigned int i = 0; i < getNumOutputs(); ++i) {
    net_hidden[i]->getGradientRef() = label[i]->clone();
  }

  backwarding();

  nntrainer::sharedConstTensors out;

  for (unsigned int i = 0; i < getNumInputs(); ++i) {
    out.push_back(MAKE_SHARED_TENSOR(net_input[i]->getGradient()));
  }

  return out;
}

void Layer::read(std::ifstream &file) {
  for (auto &weight : weights) {
    weight.getVariableRef().read(file);
  }
}

void Layer::save(std::ofstream &file) {
  for (auto &weight : weights) {
    weight.getVariableRef().save(file);
  }
}

int Layer::setProperty(std::vector<std::string> values) {
  int status = ML_ERROR_NONE;

  try {
    values = loadProperties(values, layer_props);
  } catch (std::invalid_argument &e) {
    ml_loge("parsing property failed, reason: %s", e.what());
    return ML_ERROR_INVALID_PARAMETER;
  }

  /// @todo: deprecate this in favor of loadProperties
  for (unsigned int i = 0; i < values.size(); ++i) {
    std::string key;
    std::string value;

    status = getKeyValue(values[i], key, value);
    NN_RETURN_STATUS();

    unsigned int type = parseLayerProperty(key);

    if (value.empty()) {
      ml_logd("value is empty: key: %s, value: %s", key.c_str(), value.c_str());
      return ML_ERROR_INVALID_PARAMETER;
    }

    try {
      /// @note this calls derived setProperty if available
      setProperty(static_cast<PropertyType>(type), value);
    } catch (...) {
      ml_logd("value or key is not valid, key: %s, value: %s", key.c_str(),
              value.c_str());
      return ML_ERROR_INVALID_PARAMETER;
    }
  }
  return status;
}

void Layer::setProperty(const PropertyType type, const std::string &value) {
  int status = ML_ERROR_NONE;

  switch (type) {
  case PropertyType::input_shape: {
    if (getNumInputs() != 1) {
      throw std::invalid_argument("input_shape keyword is only for one input");
    }

    TensorDim &in_dim = input_dim[0];
    if (!value.empty()) {
      unsigned int cache_batch_size = 1;
      /** cache original value of batch size */
      if (in_dim.batch()) {
        cache_batch_size = in_dim.batch();
        in_dim.batch(1);
      }
      status = in_dim.setTensorDim(value.c_str());
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
  case PropertyType::activation:
    if (!value.empty()) {
      setActivation((ActivationType)parseType(value, TOKEN_ACTI));
    }
    break;
  case PropertyType::weight_regularizer:
    if (!value.empty()) {
      weight_regularizer =
        (WeightRegularizer)parseType(value, TOKEN_WEIGHT_REGULARIZER);
      if (weight_regularizer == WeightRegularizer::UNKNOWN) {
        throw std::invalid_argument("[Layer] Unknown Weight decay");
      }
    }
    break;
  case PropertyType::weight_regularizer_constant:
    if (!value.empty()) {
      status = setFloat(weight_regularizer_constant, value);
      throw_status(status);
    }
    break;
  case PropertyType::weight_initializer:
    if (!value.empty()) {
      weight_initializer =
        (WeightInitializer)parseType(value, TOKEN_WEIGHT_INIT);
    }
    break;
  case PropertyType::bias_initializer:
    if (!value.empty()) {
      bias_initializer = (WeightInitializer)parseType(value, TOKEN_WEIGHT_INIT);
    }
    break;
  case PropertyType::trainable:
    if (!value.empty()) {
      status = setBoolean(trainable, value);
      throw_status(status);
    }
    break;
  default:
    std::string msg =
      "[Layer] Unknown Layer Property Key for value " + std::string(value);
    throw exception::not_supported(msg);
  }
}

template <typename T>
void Layer::printIfValid(std::ostream &out, const PropertyType type,
                         const T target) {
  try {
    setProperty(type);
  } catch (exception::not_supported &e) {
    return;
  }

  out << propToStr(static_cast<unsigned int>(type)) << ": " << target
      << std::endl;
}

void Layer::printShapeInfo(std::ostream &out) {
  for (unsigned int idx = 0; idx < getNumInputs(); ++idx) {
    out << "input " << input_dim[idx];
    for (unsigned int i = 0; i < weights.size(); i++)
      out << "inner" << i << " " << weightAt(i).getVariable().getDim();
  }
  for (unsigned int idx = 0; idx < getNumOutputs(); ++idx) {
    out << "output " << output_dim[idx];
  }
}

void Layer::printPropertiesMeta(std::ostream &out) {
  printIfValid(
    out, PropertyType::activation,
    static_cast<std::underlying_type<ActivationType>::type>(activation_type));
}

void Layer::printProperties(std::ostream &out) {
  out << "Trainable: " << trainable << std::endl;
  printIfValid(out, PropertyType::weight_regularizer,
               static_cast<int>(weight_regularizer));
  printIfValid(out, PropertyType::weight_regularizer_constant,
               weight_regularizer_constant);
}

void Layer::printMetric(std::ostream &out) {
  if (loss > 0) {
    out << "Weight regularization loss: " << loss;
  }
}

void Layer::printPreset(std::ostream &out, PrintPreset preset) {
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

void Layer::print(std::ostream &out, unsigned int flags) {
  /** @todo properly move print to LayerNode */
  if (flags & PRINT_INST_INFO) {
    out << "===================";
    // if (getName().empty())
    //   printInstance(out, this);
    // else
    //   out << "<" << getName() << ">" << std::endl;

    out << "Layer Type: " << getType() << std::endl;
  }

  if (flags & PRINT_SHAPE_INFO) {
    out << "======shape information: " << std::endl;
    printShapeInfo(out);
  }

  if (flags & PRINT_PROP_META) {
    out << "======meta properties: " << std::endl;
    printPropertiesMeta(out);
  }

  if (flags & PRINT_PROP) {
    out << "======properties: " << std::endl;
    printProperties(out);
  }

  if (flags & PRINT_WEIGHTS) {
    out << "======weights: " << std::endl;
    for (auto const &weight : weights) {
      out << '[' << weight.getName() << ']' << std::endl;
      out << weight.getVariable();
    }
  }

  if (flags & PRINT_METRIC) {
    out << "======metrics: " << std::endl;
    printMetric(out);
  }
};

std::shared_ptr<Layer> getLayerDevel(std::shared_ptr<ml::train::Layer> l) {
  return std::static_pointer_cast<LayerNode>(l)->getObject();
}

} /* namespace nntrainer */
