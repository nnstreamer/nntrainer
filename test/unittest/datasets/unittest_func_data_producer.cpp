// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file unittest_func_data_producer.cpp
 * @date 12 July 2021
 * @brief Function data producers (Param Tests)
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <gtest/gtest.h>

#include <data_producer_common_tests.h>
#include <func_data_producer.h>
#include <tensor.h>

namespace {
std::vector<nntrainer::TensorDim> input_shapes = {{1, 2, 4, 5}, {1, 2, 3, 4}};
std::vector<nntrainer::TensorDim> label_shapes = {{1, 1, 1, 10}, {1, 1, 1, 2}};
int user_data = 0;

int getSample(float **outVec, float **outLabel, bool *last, void *user_data) {
  /** test user data is given correctly */
  int *ud = reinterpret_cast<int *>(user_data);
  *ud += 1;

  /** first input/label is all zero, second input/label is all one */
  auto first_input = nntrainer::Tensor::Map(
    *outVec, input_shapes[0].getDataLen() * sizeof(float), input_shapes[0]);
  first_input.setValue(0);

  auto second_input = nntrainer::Tensor::Map(
    *(outVec + 1), input_shapes[1].getDataLen() * sizeof(float),
    input_shapes[1]);
  second_input.setValue(1);

  auto first_label = nntrainer::Tensor::Map(
    *outLabel, label_shapes[0].getDataLen() * sizeof(float), label_shapes[0]);
  first_label.setValue(0);

  auto second_label = nntrainer::Tensor::Map(
    *(outLabel + 1), label_shapes[1].getDataLen() * sizeof(float),
    label_shapes[1]);
  second_label.setValue(1);
  *last = false;

  return 0;
};

int getSample_error(float **outVec, float **outLabel, bool *last,
                    void *user_data) {
  return -1;
}

bool validate(const std::vector<nntrainer::Tensor> &inputs,
              const std::vector<nntrainer::Tensor> &labels) {
  if (user_data == 0 || inputs.size() != 2 || labels.size() != 2) {
    return false;
  }

  nntrainer::Tensor expected_first_input(input_shapes[0]);
  expected_first_input.setValue(0);
  nntrainer::Tensor expected_second_input(input_shapes[1]);
  expected_second_input.setValue(1);

  nntrainer::Tensor expected_first_label(label_shapes[0]);
  expected_first_label.setValue(0);
  nntrainer::Tensor expected_second_label(label_shapes[1]);
  expected_second_label.setValue(1);

  return inputs[0] == expected_first_input &&
         inputs[1] == expected_second_input &&
         labels[0] == expected_first_label &&
         labels[1] == expected_second_label;
};

} // namespace

std::unique_ptr<nntrainer::DataProducer>
createConstantSampleProducer(const std::vector<std::string> &properties = {}) {
  std::unique_ptr<nntrainer::DataProducer> ptr =
    std::make_unique<nntrainer::FuncDataProducer>(getSample, &user_data);
  return ptr;
}

std::unique_ptr<nntrainer::DataProducer>
createErrorSampleProducer(const std::vector<std::string> &properties = {}) {
  std::unique_ptr<nntrainer::DataProducer> ptr =
    std::make_unique<nntrainer::FuncDataProducer>(getSample_error, nullptr);
  return ptr;
}

std::unique_ptr<nntrainer::DataProducer>
createNullSampleProducer(const std::vector<std::string> &properties = {}) {
  std::unique_ptr<nntrainer::DataProducer> ptr =
    std::make_unique<nntrainer::FuncDataProducer>(nullptr, nullptr);
  return ptr;
}

auto func_success = DataProducerSemanticsParamType(
  createConstantSampleProducer, {}, input_shapes, label_shapes, validate,
  DataProducerSemanticsExpectedResult::SUCCESS);

auto func_error = DataProducerSemanticsParamType(
  createErrorSampleProducer, {}, input_shapes, label_shapes, nullptr,
  DataProducerSemanticsExpectedResult::FAIL_AT_GENERATOR_CALL);

auto func_nullptr = DataProducerSemanticsParamType(
  createNullSampleProducer, {}, input_shapes, label_shapes, nullptr,
  DataProducerSemanticsExpectedResult::FAIL_AT_FINALIZE);

INSTANTIATE_TEST_CASE_P(Func, DataProducerSemantics,
                        ::testing::Values(func_success, func_error,
                                          func_nullptr));

INSTANTIATE_TEST_CASE_P(Func, DataProducerSemantics_samples,
                        ::testing::Values(func_success, func_error,
                                          func_nullptr));
