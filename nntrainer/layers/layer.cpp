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

#include <layer_internal.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <optimizer_factory.h>
#include <parse_util.h>
#include <util_func.h>

namespace nntrainer {

void Layer::setActivation(ActivationType acti) {
  if (acti == ActivationType::ACT_UNKNOWN) {
    throw std::invalid_argument("Error:have to specify activation function");
  }
  activation_type = acti;
}

int Layer::setOptimizer(std::shared_ptr<Optimizer> opt) {
  this->opt = createOptimizer(opt->getType(), *opt);
  return this->opt->initialize(weights, true);
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
  for (unsigned int idx = 0; idx < num_inputs; ++idx)
    input_dim[idx].setTensorDim(0, batch);

  for (unsigned int idx = 0; idx < num_outputs; ++idx)
    output_dim[idx].setTensorDim(0, batch);
}

std::vector<Tensor> Layer::getOutputs() {
  std::vector<Tensor> ret;
  for (unsigned int i = 0; i < num_outputs; ++i) {
    ret.push_back(net_hidden[i]->var);
  }
  return ret;
}

std::vector<Tensor> Layer::getDerivatives() {
  std::vector<Tensor> ret;
  for (unsigned int i = 0; i < num_inputs; ++i) {
    ret.push_back(net_input[i]->var);
  }
  return ret;
}

void Layer::copy(std::shared_ptr<Layer> l) {
  for (auto const &w : weights)
    weights.push_back(w.clone());

  // TODO: fix this #630
  this->opt = l->opt;
  this->input_dim = l->input_dim;
  this->output_dim = l->output_dim;
  this->input.copy(l->input);
  this->hidden.copy(l->hidden);
  this->activation_type = l->activation_type;
  this->loss = l->loss;
  this->weight_regularizer = l->weight_regularizer;
  this->weight_regularizer_constant = l->weight_regularizer_constant;
  this->weight_initializer = l->weight_initializer;
  this->flatten = l->flatten;
  this->trainable = l->trainable;
  this->num_inputs = l->num_inputs;
  this->num_outputs = l->num_outputs;
}

sharedConstTensors Layer::forwarding_with_val(sharedConstTensors input) {

  for (unsigned int i = 0; i < num_inputs; ++i) {
    net_input[i]->var = input[i]->clone();
  }

  if (num_outputs != net_hidden.size())
    net_hidden.resize(num_outputs);

  forwarding();

  nntrainer::sharedConstTensors out;

  for (unsigned int i = 0; i < num_outputs; ++i) {
    out.push_back(MAKE_SHARED_TENSOR(net_hidden[i]->var));
  }

  return out;
}

sharedConstTensors Layer::backwarding_with_val(int iteration,
                                               sharedConstTensors deriv,
                                               sharedConstTensors in) {

  for (unsigned int i = 0; i < num_outputs; ++i) {
    net_hidden[i]->var = deriv[i]->clone();
  }

  if (num_inputs != net_input.size())
    net_input.resize(num_inputs);

  // TODO Need to fix to use LossLayer::type instead of "loss". But cyclic
  // includes!
  if (istrequal(getType(), "loss")) {
    backwarding(iteration, in);
  } else {
    backwarding(iteration, deriv);
  }

  nntrainer::sharedConstTensors out;

  for (unsigned int i = 0; i < num_inputs; ++i) {
    out.push_back(MAKE_SHARED_TENSOR(net_input[i]->var));
  }

  return out;
}

void Layer::read(std::ifstream &file) {
  for (auto &weight : weights) {
    weight.getVariableRef().read(file);
  }
  if (opt)
    opt->read(file);
}

void Layer::save(std::ofstream &file) {
  for (auto &weight : weights) {
    weight.getVariableRef().save(file);
  }
  if (opt)
    opt->save(file);
}

void Layer::applyGradient(unsigned int iteration) {
  if (trainable && !weights.empty()) {
    opt->apply_gradients(weights, iteration);
  }
}

int Layer::setProperty(std::vector<std::string> values) {
  int status = ML_ERROR_NONE;

  for (unsigned int i = 0; i < values.size(); ++i) {
    std::string key;
    std::string value;

    status = getKeyValue(values[i], key, value);
    NN_RETURN_STATUS();

    unsigned int type = parseLayerProperty(key);

    if (value.empty()) {
      ml_logd("value is empty for layer: %s, key: %s, value: %s",
              getName().c_str(), key.c_str(), value.c_str());
      return ML_ERROR_INVALID_PARAMETER;
    }

    try {
      /// @note this calls derived setProperty if available
      setProperty(static_cast<PropertyType>(type), value);
    } catch (...) {
      ml_logd("value or key is not valid for layer: %s, key: %s, value: %s",
              getName().c_str(), key.c_str(), value.c_str());
      return ML_ERROR_INVALID_PARAMETER;
    }
  }
  return status;
}

void Layer::setProperty(const PropertyType type, const std::string &value) {
  int status = ML_ERROR_NONE;

  switch (type) {
  case PropertyType::name:
    if (!value.empty()) {
      status = setName(value);
      throw_status(status);
    }
    break;
  case PropertyType::input_shape: {
    if (num_inputs != 1) {
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
  case PropertyType::flatten:
    if (!value.empty()) {
      status = setBoolean(flatten, value);
      throw_status(status);
    }
    break;
  case PropertyType::weight_regularizer:
    if (!value.empty()) {
      weight_regularizer =
        (WeightRegularizerType)parseType(value, TOKEN_WEIGHT_REGULARIZER);
      if (weight_regularizer == WeightRegularizerType::unknown) {
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
  case PropertyType::input_layers:
    if (!value.empty()) {
      static const std::regex reg("\\,+");
      std::vector<std::string> concat_layers = split(value, reg);

      num_inputs = concat_layers.size();
      input_dim.resize(num_inputs);
      input_layers.clear();
      for (unsigned int i = 0; i < num_inputs; ++i)
        input_layers.push_back(concat_layers[i]);
    }
    break;
  case PropertyType::output_layers:
    if (!value.empty()) {
      static const std::regex reg("\\,+");
      std::vector<std::string> concat_layers = split(value, reg);

      num_outputs = concat_layers.size();
      output_dim.resize(num_outputs);
      output_layers.clear();
      for (unsigned int i = 0; i < num_outputs; ++i)
        output_layers.push_back(concat_layers[i]);
    }
    break;
  default:
    std::string msg =
      "[Layer] Unknown Layer Property Key for value " + std::string(value);
    throw exception::not_supported(msg);
  }
}

int Layer::setName(std::string name_) {
  if (name_.empty())
    return ML_ERROR_INVALID_PARAMETER;

  name = name_;
  return ML_ERROR_NONE;
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
  for (unsigned int idx = 0; idx < num_inputs; ++idx) {
    out << "input " << input_dim[idx];
    for (unsigned int i = 0; i < weights.size(); i++)
      out << "inner" << i << " " << weightAt(i).getVariable().getDim();
  }
  for (unsigned int idx = 0; idx < num_outputs; ++idx) {
    out << "output " << output_dim[idx];
  }
}

void Layer::printPropertiesMeta(std::ostream &out) {
  printIfValid(
    out, PropertyType::activation,
    static_cast<std::underlying_type<ActivationType>::type>(activation_type));
  printIfValid(out, PropertyType::flatten, flatten);
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

} /* namespace nntrainer */
