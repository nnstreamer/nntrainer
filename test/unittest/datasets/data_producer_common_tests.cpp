// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file data_producer_common_tests.cpp
 * @date 12 July 2021
 * @brief Common test for nntrainer data producers (Param Tests)
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <gtest/gtest.h>

#include <data_producer_common_tests.h>

void DataProducerSemantics::SetUp() {
  auto [producerFactory, properties, input_dims_, label_dims_, validator_,
        result_] = GetParam();

  producer = producerFactory(properties);
  input_dims = std::move(input_dims_);
  label_dims = std::move(label_dims_);
  result = result_;
  validator = std::move(validator_);

  if (result != DataProducerSemanticsExpectedResult::SUCCESS) {
    ASSERT_EQ(validator, nullptr)
      << "Given expected result of not success, validator must be empty!";
  }
}

void DataProducerSemantics::TearDown() {}

TEST_P(DataProducerSemantics, finalize_pn) {
  if (result == DataProducerSemanticsExpectedResult::FAIL_AT_FINALIZE) {
    EXPECT_ANY_THROW(producer->finalize(input_dims, label_dims));
  } else {
    EXPECT_NO_THROW(producer->finalize(input_dims, label_dims));
  }
}

TEST_P(DataProducerSemantics, error_once_or_not_pn) {
  if (result == DataProducerSemanticsExpectedResult::FAIL_AT_FINALIZE) {
    return; // skip this test
  }

  auto generator = producer->finalize(input_dims, label_dims);
  if (result == DataProducerSemanticsExpectedResult::FAIL_AT_GENERATOR_CALL) {
    EXPECT_ANY_THROW(generator());
  } else {
    EXPECT_NO_THROW(generator());
  }
}

TEST_P(DataProducerSemantics, fetch_one_epoch_or_10_iteration_pn) {
  if (result != DataProducerSemanticsExpectedResult::SUCCESS) {
    return; // skip this test
  }

  auto generator = producer->finalize(input_dims, label_dims);
  auto sz = producer->size(input_dims, label_dims);
  bool has_fixed_size = sz != nntrainer::DataProducer::SIZE_UNDEFINED;

  if (!has_fixed_size) {
    sz = 10;
  }

  for (unsigned i = 0; i < sz; ++i) {
    auto [last, ins, labels] = generator();

    ASSERT_FALSE(last) << " reached last at iteration: " << i << '\n';
    if (validator) {
      ASSERT_TRUE(validator(ins, labels))
        << " failed validation for iteration: " << i << '\n';
    }
  }

  if (has_fixed_size) {
    {
      auto result = generator();
      bool last = std::get<0>(result);
      EXPECT_TRUE(last);
    }

    {
      auto [last, ins, labels] = generator();
      EXPECT_TRUE(validator(ins, labels))
        << "failed last validation after one epoch\n";
      EXPECT_FALSE(last);
    }
  }
}
