// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file unittest_data_iteration.cpp
 * @date 11 Aug 2021
 * @brief Sample Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <gtest/gtest.h>

#include <data_iteration.h>
#include <tensor.h>
#include <tensor_dim.h>

TEST(DataIteration, construct_p) {
  EXPECT_NO_THROW(nntrainer::Iteration({{3, 1, 1, 1}, {3, 1, 1, 2}},
                                       {{3, 1, 1, 10}, {3, 2, 4, 5}}));

  {
    auto iter = nntrainer::Iteration({{3, 1, 1, 1}, {3, 1, 1, 2}},
                                     {{3, 1, 1, 10}, {3, 2, 4, 5}});

    EXPECT_NO_THROW(iter.getInputsRef());
    EXPECT_NO_THROW(iter.getLabelsRef());
  }
  {
    const auto iter = nntrainer::Iteration({{3, 1, 1, 1}, {3, 1, 1, 2}},
                                           {{3, 1, 1, 10}, {3, 2, 4, 5}});

    EXPECT_NO_THROW(iter.getInputsRef());
    EXPECT_NO_THROW(iter.getLabelsRef());

    for (auto i = iter.begin(); i != iter.end(); ++i) {
      EXPECT_EQ(i->getInputsRef().front().getDim(),
                nntrainer::TensorDim(1, 1, 1, 1));
      EXPECT_EQ(i->getInputsRef().back().getDim(),
                nntrainer::TensorDim(1, 1, 1, 2));
      EXPECT_EQ(i->getLabelsRef().front().getDim(),
                nntrainer::TensorDim(1, 1, 1, 10));
      EXPECT_EQ(i->getLabelsRef().back().getDim(),
                nntrainer::TensorDim(1, 2, 4, 5));
    }
  }
}

TEST(DataIteration, constructEmptyInput_n) {
  EXPECT_THROW(nntrainer::Iteration({}, {{3, 1, 1, 10}, {3, 2, 4, 5}}),
               std::invalid_argument);
}

TEST(DataIteration, constructDifferentBatchSize_n) {
  EXPECT_THROW(nntrainer::Iteration({{3, 1, 1, 1}, {2, 1, 1, 2}},
                                    {{3, 1, 1, 10}, {3, 2, 4, 5}}),
               std::invalid_argument);
}

TEST(DataSample, constructSample_p) {
  auto iter = nntrainer::Iteration({{3, 1, 1, 1}, {3, 1, 1, 2}},
                                   {{3, 1, 1, 10}, {3, 2, 4, 5}});

  EXPECT_NO_THROW(nntrainer::Sample(iter, 0));
  EXPECT_NO_THROW(nntrainer::Sample(iter, 1));
  EXPECT_NO_THROW(nntrainer::Sample(iter, 2));
}

TEST(DataSample, constructOutOfBatch_n) {
  auto iter = nntrainer::Iteration({{3, 1, 1, 1}, {3, 1, 1, 2}},
                                   {{3, 1, 1, 10}, {3, 2, 4, 5}});

  EXPECT_ANY_THROW(nntrainer::Sample(iter, 3));
  EXPECT_ANY_THROW(nntrainer::Sample(iter, 4));
}
