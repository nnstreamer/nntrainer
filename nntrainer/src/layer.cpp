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

#include <layer.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <parse_util.h>
#include <util_func.h>

namespace nntrainer {

int Layer::setActivation(ActiType acti) {
  int status = ML_ERROR_NONE;
  if (acti == ACT_UNKNOWN) {
    ml_loge("Error:have to specify activation function");
    return ML_ERROR_INVALID_PARAMETER;
  }
  activation_type = acti;

  return status;
}

int Layer::setOptimizer(Optimizer &opt) {
  this->opt.setType(opt.getType());
  this->opt.setOptParam(opt.getOptParam());

  return this->opt.initialize(params, param_size, true);
}

int Layer::checkValidation() {
  int status = ML_ERROR_NONE;
  if (type == LAYER_UNKNOWN) {
    ml_loge("Error: Layer type is unknown");
    return ML_ERROR_INVALID_PARAMETER;
  }

  if (activation_type == ACT_UNKNOWN) {
    ml_loge("Error: Have to set activation for this layer");
    return ML_ERROR_INVALID_PARAMETER;
  }

  return status;
}

void Layer::copy(std::shared_ptr<Layer> l) {
  setParamSize(l->param_size);
  for (unsigned int i = 0; i < l->param_size; ++i) {
    paramsAt(i) = l->paramsAt(i);
  }
}

void Layer::read(std::ifstream &file) {
  for (unsigned int i = 0; i < param_size; ++i) {
    paramsAt(i).weight.read(file);
  }
}

void Layer::save(std::ofstream &file) {
  for (unsigned int i = 0; i < param_size; ++i) {
    paramsAt(i).weight.save(file);
  }
}

Tensor Layer::initializeWeight(const TensorDim &w_dim,
                               WeightInitializer initializer, int &status) {

  Tensor w = Tensor(w_dim);

  if (initializer == WEIGHT_UNKNOWN) {
    ml_logw("Warning: Weight Initalization Type is not set. "
            "WEIGHT_XAVIER_NORMAL is used by default");
    initializer = WEIGHT_XAVIER_NORMAL;
  }

  switch (initializer) {
  case WEIGHT_ZEROS:
    w.setZero();
    break;
  case WEIGHT_ONES:
    w.setValue(1.0f);
    break;
  case WEIGHT_LECUN_NORMAL:
    w.setRandNormal(0.0f, sqrt(1.0f / w_dim.height()));
    break;
  case WEIGHT_XAVIER_NORMAL:
    w.setRandNormal(0.0f, sqrt(2.0f / (w_dim.width() + w_dim.height())));
    break;
  case WEIGHT_HE_NORMAL:
    w.setRandNormal(0.0f, sqrt(2.0f / (w_dim.height())));
    break;
  case WEIGHT_LECUN_UNIFORM:
    w.setRandUniform(-1.0f * sqrt(1.0f / w_dim.height()),
                     sqrt(1.0f / w_dim.height()));
    break;
  case WEIGHT_XAVIER_UNIFORM:
    w.setRandUniform(-1.0f * sqrt(6.0f / (w_dim.height() + w_dim.width())),
                     sqrt(6.0 / (w_dim.height() + w_dim.width())));
    break;
  case WEIGHT_HE_UNIFORM:
    w.setRandUniform(-1.0f * sqrt(6.0f / (w_dim.height())),
                     sqrt(6.0 / (w_dim.height())));
    break;
  default:
    break;
  }
  return w;
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
      return ML_ERROR_INVALID_PARAMETER;
    }

    try {
      /// @note this calls derived setProperty if available
      setProperty(static_cast<PropertyType>(type), value);
    } catch (...) {
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
  case PropertyType::input_shape:
    if (!value.empty()) {
      unsigned int cache_batch_size = 1;
      /** cache original value of batch size */
      if (input_dim.batch()) {
        cache_batch_size = input_dim.batch();
        input_dim.batch(1);
      }
      status = input_dim.setTensorDim(value.c_str());
      if (input_dim.batch() > 1) {
        ml_logw("Batch size set with input dimension %d is ignored."
                "Set batchsize property for the model to update batchsize.",
                input_dim.batch());
      }
      /** set back to cache value of dimension */
      input_dim.batch(cache_batch_size);
      throw_status(status);
    }
    break;
  case PropertyType::batch_size:
    if (!value.empty()) {
      unsigned int batch_size;
      status = setUint(batch_size, value);
      throw_status(status);
      input_dim.batch(batch_size);
    }
    break;
  case PropertyType::activation:
    if (!value.empty()) {
      status = setActivation((ActiType)parseType(value, TOKEN_ACTI));
      throw_status(status);
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
      weight_regularizer.type =
        (WeightRegularizerType)parseType(value, TOKEN_WEIGHT_REGULARIZER);
      if (weight_regularizer.type == WeightRegularizerType::unknown) {
        throw std::invalid_argument("[Layer] Unknown Weight decay");
      }
    }
    break;
  case PropertyType::weight_regularizer_constant:
    if (!value.empty()) {
      status = setFloat(weight_regularizer.constant, value);
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
  default:
    std::string msg =
      "[Layer] Unknown Layer Property Key for value " + std::string(value);
    throw exception::not_supported(msg);
  }
}

int Layer::setName(std::string name) {
  if (name.empty())
    return ML_ERROR_INVALID_PARAMETER;

  this->name = name;
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
  out << "input " << input_dim;
  for (unsigned int i = 0; i < param_size; i++)
    out << "inner" << i << " " << paramsAt(i).weight.getDim();
  out << "output " << output_dim;
}

void Layer::printPropertiesMeta(std::ostream &out) {
  printIfValid(out, PropertyType::activation, activation_type);
  printIfValid(out, PropertyType::flatten, flatten);
}

void Layer::printProperties(std::ostream &out) {
  out << "Trainable: " << trainable << std::endl;
  printIfValid(out, PropertyType::weight_regularizer,
               static_cast<int>(weight_regularizer.type));
  printIfValid(out, PropertyType::weight_regularizer_constant,
               weight_regularizer.constant);
}

void Layer::printMetric(std::ostream &out) {
  if (loss > 0) {
    out << "Weight regularization loss: " << loss;
  }
}

void Layer::print(std::ostream &out, unsigned int flags) {
  if (flags & PRINT_INST_INFO) {
    out << "===================";
    printInstance(out, this);

    out << "Layer Type: " << type << std::endl;
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
    for (unsigned int i = 0; i < param_size; ++i) {
      out << '[' << paramsAt(i).name << ']' << std::endl;
      out << paramsAt(i).weight;
    }
  }

  if (flags & PRINT_METRIC) {
    out << "======metrics: " << std::endl;
    printMetric(out);
  }
};

} /* namespace nntrainer */
