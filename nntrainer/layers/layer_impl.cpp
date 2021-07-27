// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   layer_impl.h
 * @date   21 June 2021
 * @brief  This is base layer implementation class
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */
#include <layer_impl.h>

#include <string>
#include <vector>

#include <common_properties.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <parse_util.h>

namespace nntrainer {

LayerImpl::LayerImpl() :
  layer_impl_props(std::make_unique<std::tuple<>>()),
  weight_regularizer(WeightRegularizer::NONE),
  weight_regularizer_constant(1.0f),
  weight_initializer(Tensor::Initializer::XAVIER_UNIFORM),
  bias_initializer(Tensor::Initializer::ZEROS) {}

void LayerImpl::setProperty(const std::vector<std::string> &values) {
  loadProperties(values, *layer_impl_props);

  /// @todo: deprecate this in favor of loadProperties
  for (unsigned int i = 0; i < values.size(); ++i) {
    std::string key;
    std::string value;
    std::stringstream ss;

    if (getKeyValue(values[i], key, value) != ML_ERROR_NONE) {
      throw std::invalid_argument("Error parsing the property: " + values[i]);
    }

    if (value.empty()) {
      ss << "value is empty: key: " << key << ", value: " << value;
      throw std::invalid_argument(ss.str());
    }

    /// @note this calls derived setProperty if available
    setProperty(key, value);
  }
}

void LayerImpl::setProperty(const std::string &type_str,
                            const std::string &value) {
  using PropertyType = nntrainer::Layer::PropertyType;

  int status = ML_ERROR_NONE;
  nntrainer::Layer::PropertyType type =
    static_cast<nntrainer::Layer::PropertyType>(parseLayerProperty(type_str));

  switch (type) {
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
        (Tensor::Initializer)parseType(value, TOKEN_WEIGHT_INIT);
    }
    break;
  case PropertyType::bias_initializer:
    if (!value.empty()) {
      bias_initializer =
        (Tensor::Initializer)parseType(value, TOKEN_WEIGHT_INIT);
    }
    break;
  default:
    std::string msg =
      "[Layer] Unknown Layer Property Key for value " + std::string(value);
    throw exception::not_supported(msg);
  }
}

void LayerImpl::exportTo(Exporter &exporter,
                         const ExportMethods &method) const {}

} // namespace nntrainer
