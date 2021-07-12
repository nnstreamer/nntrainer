// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file data_producer_common_tests.h
 * @date 12 July 2021
 * @brief Common test for nntrainer data producers (Param Tests)
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
#ifndef __DATA_PRODUCER_COMMON_TESTS_H__
#define __DATA_PRODUCER_COMMON_TESTS_H__

#include <gtest/gtest.h>

#include <vector>

#include <data_producers.h>
#include <tensor.h>
#include <tensor_dim.h>

using DataProducerFactoryType =
  std::function<std::unique_ptr<nntrainer::DataProducer>(
    const std::vector<std::string> &)>;

using DataProducerValidatorType =
  std::function<bool(const std::vector<nntrainer::Tensor> &,
                     const std::vector<nntrainer::Tensor> &)>;

/**
 * @brief Data Producer Semantics expected result
 *
 */
enum class DataProducerSemanticsExpectedResult {
  SUCCESS = 0,                /**< SUCCESS */
  FAIL_AT_FINALIZE = 1,       /**< FAIL AT FINALIZE */
  FAIL_AT_GENERATOR_CALL = 2, /**< FAIL AT GENERATOR CALL */
};

using DataProducerSemanticsParamType =
  std::tuple<DataProducerFactoryType /**< layer factory */,
             std::vector<std::string> /**< properties */,
             std::vector<nntrainer::TensorDim> /**< input dimensions */,
             std::vector<nntrainer::TensorDim> /**< label dimensions */,
             DataProducerValidatorType /**< validator if any */,
             DataProducerSemanticsExpectedResult /**< expected result */>;

/**
 * @brief Dataset Producer Semantics Tests
 *
 */
class DataProducerSemantics
  : public ::testing::TestWithParam<DataProducerSemanticsParamType> {
public:
  /**
   * @brief SetUp test cases here
   *
   */
  virtual void SetUp();

  /**
   * @brief do here if any memory needs to be released
   *
   */
  virtual void TearDown();

protected:
  std::unique_ptr<nntrainer::DataProducer>
    producer;                                   /**< producer to be tested */
  std::vector<nntrainer::TensorDim> input_dims; /**< input dims */
  std::vector<nntrainer::TensorDim> label_dims; /**< output dims */
  DataProducerValidatorType validator;          /**< result validator */
  DataProducerSemanticsExpectedResult result;   /**< expected result */
};

/**
 * @brief Create a Data Producer object
 *
 * @tparam T inherited class of data producer
 * @param props properties
 * @return std::unique_ptr<nntrainer::DataProducer> created producer object
 */
template <typename T,
          std::enable_if_t<std::is_base_of<nntrainer::DataProducer, T>::value,
                           T> * = nullptr>
std::unique_ptr<nntrainer::DataProducer>
createDataProducer(const std::vector<std::string> &props = {}) {
  std::unique_ptr<nntrainer::DataProducer> ptr = std::make_unique<T>();
  ptr->setProperty(props);
  return ptr;
}

#endif // __DATA_PRODUCER_COMMON_TESTS_H__
