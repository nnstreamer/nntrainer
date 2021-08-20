// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file unittest_raw_file_data_producer.cpp
 * @date 12 July 2021
 * @brief raw file data producers (Param Tests)
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <gtest/gtest.h>

#include <data_producer_common_tests.h>
#include <raw_file_data_producer.h>
#include <tensor.h>

#include <nntrainer_test_util.h>

static const std::string getTestResPath(const std::string &file) {
  return getResPath(file, {"test"});
}

namespace {
std::vector<nntrainer::TensorDim> input_shapes = {{20, 3, 32, 32}};
std::vector<nntrainer::TensorDim> label_shapes = {{20, 1, 1, 10}};

bool validate(const std::vector<nntrainer::Tensor> &inputs,
              const std::vector<nntrainer::Tensor> &labels) {
  return true;
};
} // namespace

auto training_set = DataProducerSemanticsParamType(
  createDataProducer<nntrainer::RawFileDataProducer>,
  {"path=" + getTestResPath("trainingSet.dat")}, {{20, 3, 32, 32}},
  {{20, 1, 1, 10}}, validate, DataProducerSemanticsExpectedResult::SUCCESS);

auto valSet = DataProducerSemanticsParamType(
  createDataProducer<nntrainer::RawFileDataProducer>,
  {"path=" + getTestResPath("valSet.dat")}, {{3, 32, 32}}, {{1, 1, 10}},
  validate, DataProducerSemanticsExpectedResult::SUCCESS);

auto testSet = DataProducerSemanticsParamType(
  createDataProducer<nntrainer::RawFileDataProducer>,
  {"path=" + getTestResPath("testSet.dat")}, {{3, 32, 32}}, {{1, 1, 10}},
  validate, DataProducerSemanticsExpectedResult::SUCCESS);

auto batch_too_big = DataProducerSemanticsParamType(
  createDataProducer<nntrainer::RawFileDataProducer>,
  {"path=" + getTestResPath("testSet.dat")}, {{50000, 3, 32, 32}},
  {{50000, 1, 1, 10}}, nullptr,
  DataProducerSemanticsExpectedResult::FAIL_AT_FINALIZE);

INSTANTIATE_TEST_CASE_P(RawFile, DataProducerSemantics,
                        ::testing::Values(training_set, valSet, testSet));
