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

TEST(DataBuffer, batcher_p) {
  std::unique_ptr<nntrainer::DataProducer> prod =
    std::make_unique<nntrainer::RandomDataOneHotProducer>();

  nntrainer::DataBuffer db(std::move(prod));
  db.setProperty({"buffer_size=2", "min=1", "max=2", "num_samples=3"});

  auto generator = db.batcher({{3, 1, 1, 2}}, {{3, 1, 1, 1}});
  auto [last, inputs, labels] = generator();

  EXPECT_FALSE(last);
  EXPECT_EQ(inputs.size(), 1u);
  EXPECT_EQ(inputs[0].getDim(), nntrainer::TensorDim({3, 1, 1, 2}));
  EXPECT_EQ(labels.size(), 1u);
  EXPECT_EQ(labels[0].getDim(), nntrainer::TensorDim({3, 1, 1, 1}));
}

TEST(DataBuffer, fetch_p) {
  std::unique_ptr<nntrainer::DataProducer> prod =
    std::make_unique<nntrainer::RandomDataOneHotProducer>();

  nntrainer::DataBuffer db(std::move(prod));
  db.setProperty({"buffer_size=2", "min=1", "max=2", "num_samples=10"});

  { /// invalidate bq after epoch is finished
    auto future_bq = db.startFetchWorker({{3, 1, 1, 2}}, {{3, 1, 1, 1}});

    for (unsigned int i = 0; i < 3; ++i) {
      auto [last, inputs, labels] = *db.fetch();

      EXPECT_FALSE(last);
      EXPECT_EQ(inputs.size(), 1u);
      EXPECT_EQ(inputs[0].getDim(), nntrainer::TensorDim({3, 1, 1, 2}));
      EXPECT_EQ(labels.size(), 1u);
      EXPECT_EQ(labels[0].getDim(), nntrainer::TensorDim({3, 1, 1, 1}));
    }
    auto [last, inputs, labels] = *db.fetch();
    EXPECT_TRUE(last);

    future_bq.get();
  }

  { /// invalidate bq during epoch
    auto future_bq = db.startFetchWorker({{4, 1, 1, 2}}, {{4, 1, 1, 1}});
    db.fetch();
    future_bq.get();
  }

  { /// remainder is over batchsize so return last
    auto future_bq = db.startFetchWorker({{11, 1, 1, 2}}, {{11, 1, 1, 1}});
    auto [last, inputs, labels] = *db.fetch();
    EXPECT_TRUE(last);
    future_bq.get();
  }
}

TEST(DataBuffer, fetchWithoutStart_n) {
  std::unique_ptr<nntrainer::DataProducer> prod =
    std::make_unique<nntrainer::RandomDataOneHotProducer>();

  nntrainer::DataBuffer db(std::move(prod));
  db.setProperty({"buffer_size=2", "min=1", "max=2", "num_samples=3"});
  EXPECT_THROW(db.fetch(), std::runtime_error);
}

TEST(DataBuffer, fetchAfterBqIsDeleted_n) {
  std::unique_ptr<nntrainer::DataProducer> prod =
    std::make_unique<nntrainer::RandomDataOneHotProducer>();

  nntrainer::DataBuffer db(std::move(prod));
  db.setProperty({"buffer_size=2", "min=1", "max=2", "num_samples=3"});
  auto future_bq = db.startFetchWorker({{4, 1, 1, 2}}, {{4, 1, 1, 1}});
  future_bq.get();
  EXPECT_THROW(db.fetch(), std::runtime_error);
}

TEST(DataBuffer, fetchAfterDataEnd_n) {
  std::unique_ptr<nntrainer::DataProducer> prod =
    std::make_unique<nntrainer::RandomDataOneHotProducer>();

  nntrainer::DataBuffer db(std::move(prod));
  db.setProperty({"buffer_size=2", "min=1", "max=2", "num_samples=3"});
  auto future_bq = db.startFetchWorker({{4, 1, 1, 2}}, {{4, 1, 1, 1}});
  db.fetch();
  EXPECT_THROW(db.fetch(), std::runtime_error);
  future_bq.get();
}
