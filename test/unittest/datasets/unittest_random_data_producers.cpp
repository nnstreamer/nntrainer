// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file unittest_random_data_producers.cpp
 * @date 12 July 2021
 * @brief Random data producers test (Param Tests)
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <gtest/gtest.h>

#include <algorithm>

#include <data_producer_common_tests.h>
#include <random_data_producers.h>

DataProducerValidatorType random_onehot_validator(float min, float max) {
  /// input validator: every value is in range of min, max
  auto input_valid = [min, max](const nntrainer::Tensor &t) {
    auto data = t.getData();
    for (unsigned int i = 0; i < t.length(); ++i) {
      if (*data < min || max < *data) {
        return false;
      }
      data++;
    }
    return true;
  };

  /// label validator: sum of all is equal to batch
  auto label_valid = [](const nntrainer::Tensor &t) {
    /// @todo better to check batch by batch
    return fabs(t.batch() - t.sum({0, 1, 2, 3}).getValue(0, 0, 0, 0) < 1e-7);
  };

  auto f = [input_valid,
            label_valid](const std::vector<nntrainer::Tensor> &inputs,
                         const std::vector<nntrainer::Tensor> &labels) -> bool {
    bool is_inputs_valid =
      std::all_of(inputs.begin(), inputs.end(), input_valid);
    bool is_labels_valid =
      std::all_of(labels.begin(), labels.end(), label_valid);
    return is_inputs_valid && is_labels_valid;
  };

  return f;
}
auto random_onehot_success = DataProducerSemanticsParamType(
  createDataProducer<nntrainer::RandomDataOneHotProducer>,
  {"min=0", "max=1", "size=10"}, {{3, 2, 4, 5}}, {{3, 1, 1, 10}},
  random_onehot_validator(0, 1), DataProducerSemanticsExpectedResult::SUCCESS);

auto random_onehot_min_over_max = DataProducerSemanticsParamType(
  createDataProducer<nntrainer::RandomDataOneHotProducer>,
  {"min=2", "max=1", "size=10"}, {{3, 2, 4, 5}}, {{3, 1, 1, 10}}, nullptr,
  DataProducerSemanticsExpectedResult::FAIL_AT_FINALIZE);

auto random_onehot_invalid_label_shape = DataProducerSemanticsParamType(
  createDataProducer<nntrainer::RandomDataOneHotProducer>, {}, {{3, 2, 4, 5}},
  {{3, 1, 2, 10}}, nullptr,
  DataProducerSemanticsExpectedResult::FAIL_AT_FINALIZE);

INSTANTIATE_TEST_CASE_P(RandomOneHot, DataProducerSemantics,
                        ::testing::Values(random_onehot_success,
                                          random_onehot_min_over_max));
