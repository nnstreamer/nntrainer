
// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   raw_file_data_producer.cpp
 * @date   12 July 2021
 * @brief  This file contains raw file data producers, reading from a file
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <raw_file_data_producer.h>

#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include <common_properties.h>
#include <nntrainer_error.h>
#include <node_exporter.h>
#include <util_func.h>

namespace nntrainer {

RawFileDataProducer::RawFileDataProducer() : raw_file_props(new PropTypes()) {}
RawFileDataProducer::~RawFileDataProducer() {}

const std::string RawFileDataProducer::getType() const {
  return RawFileDataProducer::type;
}

unsigned int
RawFileDataProducer::size(const std::vector<TensorDim> &input_dims,
                          const std::vector<TensorDim> &label_dims) const {
  auto size_accumulator = [](const unsigned int &a, const TensorDim &b) {
    return a + b.getFeatureLen();
  };

  auto sample_size =
    std::accumulate(input_dims.begin(), input_dims.end(), 0u, size_accumulator);
  sample_size = std::accumulate(label_dims.begin(), label_dims.end(),
                                sample_size, size_accumulator);

  auto path_prop = std::get<props::FilePath>(*raw_file_props);
  auto file_size = path_prop.file_size();

  /// checking alignment is a good way to make check if a file is valid,
  /// unfortunately, our test dataset does not have this property
  /// (trainingSet.dat, valSet.dat, testSet.dat) after checking, we can
  /// uncomment below line.
  // NNTR_THROW_IF((file_size % sample_size * RawFileDataProducer::pixel_size !=
  // 0),
  //               std::invalid_argument)
  //   << " Given file does not align with the given sample size, sample size: "
  //   << sample_size << " file_size: " << file_size;

  return path_prop.file_size() /
         (sample_size * RawFileDataProducer::pixel_size);
}

void RawFileDataProducer::setProperty(
  const std::vector<std::string> &properties) {
  auto left = loadProperties(properties, *raw_file_props);
  NNTR_THROW_IF(!left.empty(), std::invalid_argument)
    << "There is unparsed properties, size: " << left.size();
}

DataProducer::Generator
RawFileDataProducer::finalize(const std::vector<TensorDim> &input_dims,
                              const std::vector<TensorDim> &label_dims) {

  /****************** Validation ****************/
  auto sz = size(input_dims, label_dims);
  auto batch = input_dims[0].batch();

  NNTR_THROW_IF(sz < batch, std::invalid_argument)
    << "calculated sample size is less than a batch";

  auto path_prop = std::get<props::FilePath>(*raw_file_props);

  auto size_accumulator = [](const unsigned int &a, const TensorDim &b) {
    return a + b.getFeatureLen();
  };

  auto sample_size =
    std::accumulate(input_dims.begin(), input_dims.end(), 0u, size_accumulator);
  sample_size = std::accumulate(label_dims.begin(), label_dims.end(),
                                sample_size, size_accumulator);

  /// below works when checking alignment is correct
  // auto sample_size = path_prop.file_size() / (sz *
  // RawFileDataProducer::pixel_size);

  /****************** Prepare states ****************/
  std::mt19937 rng_;
  rng_.seed(getSeed());
  auto idxes_ = std::vector<unsigned int>();
  idxes_.reserve(sz);
  /// idxes point to the file position in bytes where a sample starts
  std::generate_n(std::back_inserter(idxes_), sz,
                  [sample_size, current = 0ULL]() mutable {
                    auto c = current;
                    current += sample_size * RawFileDataProducer::pixel_size;
                    return c;
                  });
  /// @todo remove shuffle from here as we are migrating this to element wise
  /// operator
  std::shuffle(idxes_.begin(), idxes_.end(), rng_);

  auto file =
    std::make_shared<std::ifstream>(path_prop.get(), std::ios::binary);
  auto iter = idxes_.begin();

  return [batch, input_dims, label_dims, rng = rng_, idxes = std::move(idxes_),
          file, iter]() mutable -> DataProducer::Iteration {
    if (std::distance(iter, idxes.end()) < static_cast<std::ptrdiff_t>(batch)) {
      std::shuffle(idxes.begin(), idxes.end(), rng);
      iter = idxes.begin();
      return DataProducer::Iteration(true, {}, {});
    }

    std::vector<Tensor> inputs;
    inputs.reserve(input_dims.size());
    for (unsigned int i = 0; i < input_dims.size(); ++i) {
      inputs.emplace_back(input_dims[i]);
    }

    std::vector<Tensor> labels;
    labels.reserve(label_dims.size());
    for (unsigned int i = 0; i < label_dims.size(); ++i) {
      labels.emplace_back(label_dims[i]);
    }

    for (unsigned int b = 0; b < batch; ++b) {
      file->seekg(*iter, std::ios_base::beg);
      for (auto &input : inputs) {
        Tensor input_slice = input.getBatchSlice(b, 1);
        input_slice.read(*file);
      }
      for (auto &label : labels) {
        Tensor label_slice = label.getBatchSlice(b, 1);
        label_slice.read(*file);
      }

      iter++;
    }

    return DataProducer::Iteration(false, inputs, labels);
  };
}
} // namespace nntrainer
