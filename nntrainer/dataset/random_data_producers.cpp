// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   random_data_producers.cpp
 * @date   09 July 2021
 * @brief  This file contains various random data producers
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */
#include <random_data_producers.h>

#include <base_properties.h>
#include <node_exporter.h>
#include <util_func.h>

namespace nntrainer {

/**
 * @brief Props containing min value
 *
 */
class PropsMin : public Property<float> {
public:
  /**
   * @brief Construct a new props min object with a default value
   *
   * @param value default value
   */
  PropsMin(float value = 0.0f) : nntrainer::Property<float>(value) {}
  static constexpr const char *key = "min"; /**< unique key to access */
  using prop_tag = float_prop_tag;          /**< property type */
};

/**
 * @brief Props containing max value
 *
 */
class PropsMax : public Property<float> {
public:
  /**
   * @brief Construct a new props max object with a default value
   *
   * @param value default value
   */
  PropsMax(float value = 1.0f) : nntrainer::Property<float>(value) {}
  static constexpr const char *key = "max"; /**< unique key to access */
  using prop_tag = float_prop_tag;          /**< property type */
};

/**
 * @brief Props containing number of samples
 * A random data producer has theoretical size. number of samples is used to set
 * theoretical size of the random data producer's data size
 *
 */
class PropsNumSamples : public Property<unsigned int> {
public:
  /**
   * @brief Construct a new props data size object with a default value
   *
   * @param value default value
   */
  PropsNumSamples(unsigned int value = 512) :
    nntrainer::Property<unsigned int>(value) {}
  static constexpr const char *key = "num_samples"; /**< unique key to access */
  using prop_tag = uint_prop_tag;                   /**< property type */
};

RandomDataOneHotProducer::RandomDataOneHotProducer() :
  rd_one_hot_props(new Props()) {}

RandomDataOneHotProducer::~RandomDataOneHotProducer() {}

const std::string RandomDataOneHotProducer::getType() const {
  return RandomDataOneHotProducer::type;
}

unsigned int
RandomDataOneHotProducer::size(const std::vector<TensorDim> &input_dims,
                               const std::vector<TensorDim> &label_dims) const {
  return std::get<PropsNumSamples>(*rd_one_hot_props).get();
}

void RandomDataOneHotProducer::setProperty(
  const std::vector<std::string> &properties) {
  auto left = loadProperties(properties, *rd_one_hot_props);
  NNTR_THROW_IF(!left.empty(), std::invalid_argument)
    << "There are unparsed properties, size: " << left.size();
}

DataProducer::Generator
RandomDataOneHotProducer::finalize(const std::vector<TensorDim> &input_dims,
                                   const std::vector<TensorDim> &label_dims) {
  /** check if the given producer is ready to finalize */
  nntrainer::PropsMin min_;
  nntrainer::PropsMax max_;
  std::tie(min_, max_, std::ignore) = *rd_one_hot_props;

  /// @todo expand this to non onehot case
  NNTR_THROW_IF(std::any_of(label_dims.begin(), label_dims.end(),
                            [](const TensorDim &dim) {
                              return dim.channel() != 1 || dim.height() != 1;
                            }),
                std::invalid_argument)
    << "Label dimension containing channel or height not allowed";

  NNTR_THROW_IF(min_.get() > max_.get(), std::invalid_argument)
    << "Min value is bigger then max value, min: " << min_.get()
    << "max: " << max_.get();

  NNTR_THROW_IF(size(input_dims, label_dims) == 0, std::invalid_argument)
    << "size is zero, dataproducer does not provide anything";

  /** prepare states for the generator */
  std::vector<std::uniform_int_distribution<unsigned int>> label_chooser_;
  label_chooser_.reserve(label_dims.size());
  std::transform(label_dims.begin(), label_dims.end(),
                 std::back_inserter(label_chooser_),
                 [this](const TensorDim &label_dim) {
                   return std::uniform_int_distribution<unsigned int>(
                     0, label_dim.width() - 1);
                 });

  std::mt19937 rng;
  rng.seed(getSeed());
  auto sz = size(input_dims, input_dims);
  /** DataProducer::Generator */
  return [rng, sz, input_dims, label_dims, min_ = min_.get(), max_ = max_.get(),
          current_iteration = 0ULL,
          label_chooser = std::move(label_chooser_)]() mutable {
    if (current_iteration++ == sz / input_dims[0].batch()) {
      current_iteration = 0;
      return DataProducer::Iteration(true, {}, {});
    }

    auto populate_tensor = [&](const TensorDim &input_dim) {
      Tensor t(input_dim);
      t.setRandUniform(min_, max_);

      return t;
    };

    auto populate_label =
      [&](const TensorDim &label_dim,
          std::uniform_int_distribution<unsigned int> &label_dist_) {
        Tensor t(label_dim);
        t.setZero();
        for (unsigned int b = 0; b < t.batch(); ++b) {
          t.setValue(b, 0, 0, label_dist_(rng), 1);
        }
        return t;
      };

    std::vector<Tensor> inputs;
    inputs.reserve(input_dims.size());

    std::vector<Tensor> labels;
    labels.reserve(label_dims.size());

    std::transform(input_dims.begin(), input_dims.end(),
                   std::back_inserter(inputs), populate_tensor);

    std::transform(label_dims.begin(), label_dims.end(), label_chooser.begin(),
                   std::back_inserter(labels), populate_label);

    return DataProducer::Iteration(false, inputs, labels);
  };
}
} // namespace nntrainer
