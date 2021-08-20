// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file unittest_databuffer.cpp
 * @date 13 July 2021
 * @brief databuffer unittest
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <gtest/gtest.h>

#include <databuffer.h>
#include <random_data_producers.h>

#include <memory>

TEST(DataBuffer, getGenerator_p) {
  std::unique_ptr<nntrainer::DataProducer> prod =
    std::make_unique<nntrainer::RandomDataOneHotProducer>();

  nntrainer::DataBuffer db(std::move(prod));
  db.setProperty({"buffer_size=3", "min=1", "max=2", "num_samples=3"});

  auto [generator, size] = db.getGenerator({{3, 1, 1, 2}}, {{3, 1, 1, 1}});

  EXPECT_EQ(size, 3u);
}

TEST(DataBuffer, fetchIteration_p) {
  std::unique_ptr<nntrainer::DataProducer> prod =
    std::make_unique<nntrainer::RandomDataOneHotProducer>();

  nntrainer::DataBuffer db(std::move(prod));
  db.setProperty({"buffer_size=9", "min=1", "max=2", "num_samples=10"});

  { /// invalidate iq after epoch is finished
    auto future_iq = db.startFetchWorker({{3, 1, 1, 2}}, {{3, 1, 1, 1}});

    for (unsigned int i = 0; i < 3; ++i) {
      auto iteration_view = db.fetch();
      EXPECT_FALSE(iteration_view.isEmpty());
      auto &iter = iteration_view.get();
      auto &inputs = iter.getInputsRef();
      auto &labels = iter.getLabelsRef();

      EXPECT_EQ(inputs.size(), 1u);
      EXPECT_EQ(inputs[0].getDim(), nntrainer::TensorDim({3, 1, 1, 2}));
      EXPECT_EQ(labels.size(), 1u);
      EXPECT_EQ(labels[0].getDim(), nntrainer::TensorDim({3, 1, 1, 1}));
    }
    {
      auto iteration_view = db.fetch();
      EXPECT_FALSE(iteration_view.isEmpty());
      EXPECT_EQ(iteration_view.get().batch(), 1u); // partial batch is allowed
    }
    {
      auto iteration_view = db.fetch(); // no more iteration
      EXPECT_TRUE(iteration_view.isEmpty());
    }

    future_iq.get();
  }
}

TEST(DataBuffer, fetchWithoutStart_n) {
  std::unique_ptr<nntrainer::DataProducer> prod =
    std::make_unique<nntrainer::RandomDataOneHotProducer>();

  nntrainer::DataBuffer db(std::move(prod));
  db.setProperty({"buffer_size=2", "min=1", "max=2", "num_samples=2"});
  EXPECT_THROW(db.fetch(), std::runtime_error);
}

TEST(DataBuffer, fetchAfterBqIsDeleted_n) {
  std::unique_ptr<nntrainer::DataProducer> prod =
    std::make_unique<nntrainer::RandomDataOneHotProducer>();

  nntrainer::DataBuffer db(std::move(prod));
  db.setProperty({"buffer_size=4", "min=1", "max=2", "num_samples=3"});
  auto future_bq = db.startFetchWorker({{4, 1, 1, 2}}, {{4, 1, 1, 1}});
  future_bq.get();
  EXPECT_THROW(db.fetch(), std::runtime_error);
}
