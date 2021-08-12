// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file unittest_batch_queue.cpp
 * @date 12 July 2021
 * @brief Batch Queue Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <gtest/gtest.h>

#include <batch_queue.h>
#include <random_data_producers.h>
#include <tensor.h>

#include <algorithm>
#include <future>
#include <thread>
#include <tuple>
#include <vector>

nntrainer::DataProducer::Iteration data(size_t key) {
  return {true, std::vector<nntrainer::Tensor>(key), {}};
};

void test_data(const nntrainer::DataProducer::Iteration &dat,
               size_t expected_key) {
  EXPECT_EQ(std::get<1>(dat).size(), expected_key);
}

TEST(BatchQueue, pushPop_p) {
  nntrainer::BatchQueue bq(1);

  EXPECT_NO_THROW(bq.wait_and_push(data(1)));
  auto result = bq.wait_and_pop();
  test_data(*result, 1);
}

TEST(BatchQueue, threadedPushPops_p) {
  /** preparing primitives */
  using namespace std::chrono_literals;
  auto push_after = [](nntrainer::BatchQueue &bq, const auto &duration,
                       size_t key) {
    std::this_thread::sleep_for(duration);
    EXPECT_NO_THROW(bq.wait_and_push(data(key)));
  };
  auto pop_after = [](nntrainer::BatchQueue &bq, const auto &duration,
                      size_t key) {
    std::this_thread::sleep_for(duration);
    auto result = bq.wait_and_pop();
    test_data(*result, key);
  };

  std::vector<std::future<void>> futures;
  {
    futures.clear();
    /// 0     -> push(1)
    /// 250ms -> pop(1)
    nntrainer::BatchQueue bq(1);
    futures.push_back(
      std::async(std::launch::async, push_after, std::ref(bq), 0ms, 1));
    futures.push_back(
      std::async(std::launch::async, pop_after, std::ref(bq), 250ms, 1));
    for (auto &future : futures) {
      future.get();
    }
  }

  {
    futures.clear();
    /// 0     -> pop(1)
    /// 250ms -> push(1)
    nntrainer::BatchQueue bq(1);
    futures.push_back(
      std::async(std::launch::async, pop_after, std::ref(bq), 0ms, 1));
    futures.push_back(
      std::async(std::launch::async, push_after, std::ref(bq), 250ms, 1));
    for (auto &future : futures) {
      future.get();
    }
  }

  {
    futures.clear();
    /// 0     -> push(1)
    /// 300ms -> push(2)
    /// 300ms -> pop(1)
    /// 500ms -> push(3)
    /// 600ms -> push(4) (waits)
    /// 750ms -> pop(2)
    /// 1000ms-> pop(3)
    nntrainer::BatchQueue bq(2);
    futures.push_back(
      std::async(std::launch::async, push_after, std::ref(bq), 0ms, 1));
    futures.push_back(
      std::async(std::launch::async, push_after, std::ref(bq), 300ms, 2));
    futures.push_back(
      std::async(std::launch::async, pop_after, std::ref(bq), 300ms, 1));
    futures.push_back(
      std::async(std::launch::async, push_after, std::ref(bq), 500ms, 3));
    futures.push_back(
      std::async(std::launch::async, push_after, std::ref(bq), 600ms, 4));
    futures.push_back(
      std::async(std::launch::async, pop_after, std::ref(bq), 750ms, 2));
    futures.push_back(
      std::async(std::launch::async, pop_after, std::ref(bq), 1000ms, 3));
    for (auto &future : futures) {
      future.get();
    }
  }
}

using IterQueueTestParamType =
  std::tuple<unsigned int /**< queue size */,
             std::vector<nntrainer::TensorDim> /**< input dimensions */,
             std::vector<nntrainer::TensorDim> /**< label dimensions */>;

/**
 * @brief Iteration Queue Test
 */
class IterQueueScenarios
  : public ::testing::TestWithParam<IterQueueTestParamType> {
public:
  /**
   * @brief SetUp test cases here
   *
   */
  virtual void SetUp() {
    auto &[q_size, input_dims, label_dims] = GetParam();
    iq = std::make_unique<nntrainer::IterationQueue>(q_size, input_dims,
                                                     label_dims);
    auto producer = std::make_unique<nntrainer::RandomDataOneHotProducer>();
    producer->setProperty({"num_samples=10000"});
    sample_getter = producer->finalize_sample(input_dims, label_dims);
    sum_from_producer = 0;
    sum_from_consumer = 0;
  }

  virtual void produceSample(unsigned int size) {
    auto sample_view = iq->requestEmpty();
    auto &sample = sample_view.get();
    auto &inputs = sample.getInputsRef();
    auto &labels = sample.getLabelsRef();
    sample_getter(size, inputs, labels);
    sum_from_producer += getSum(inputs, labels);
  }

  virtual void consumeIteration() {
    auto iter_view = iq->requestFilled();
    auto &iter = iter_view.get();
    auto &inputs = iter.getInputsRef();
    auto &labels = iter.getLabelsRef();
    sum_from_consumer += getSum(inputs, labels);
  }

  /**
   * @brief do here if any memory needs to be released
   *
   */
  virtual void TearDown(){};

protected:
  /**
   * @brief Get a single value (sum) for the given inputs and outputs, this is
   * to effectively reduce a tensor to a single value
   *
   * @param inputs inputs
   * @param labels labels
   * @return long double single value which sums everything
   */
  long double getSum(const std::vector<nntrainer::Tensor> &inputs,
                     const std::vector<nntrainer::Tensor> &labels) {
    auto accumulator = [](long double old_sum, const nntrainer::Tensor &t) {
      return old_sum + (long double)t.sum({0, 1, 2, 3}).getValue(0, 0, 0, 0);
    };

    long double sum =
      std::accumulate(inputs.begin(), inputs.end(), 0.0l, accumulator);
    return std::accumulate(labels.begin(), labels.end(), sum, accumulator);
  }

  long double sum_from_producer;
  long double sum_from_consumer;
  nntrainer::DataProducer::Generator_sample sample_getter;
  std::unique_ptr<nntrainer::IterationQueue> iq;
  std::vector<nntrainer::TensorDim> input_dims; /**< input dims */
  std::vector<nntrainer::TensorDim> label_dims; /**< output dims */
};

TEST_P(IterQueueScenarios, produceAndConsumeSingle_p) {
  auto batch_size = iq->batch();

  for (unsigned int i = 0; i < batch_size; ++i) {
    produceSample(i);
  }
  consumeIteration();

  EXPECT_FLOAT_EQ(sum_from_producer, sum_from_consumer);
}

TEST_P(IterQueueScenarios, produceAndConsumeOnce_p) {
  auto q_size = iq->slots();
  auto q_size_in_sample = q_size * iq->batch();
  /// step1: fill buffer to the queue (filling 0 ~ 11th samples)
  for (unsigned int i = 0; i < q_size_in_sample; ++i) {
    produceSample(i);
  }

  /// step2: consume the filled buffer from the queue
  for (unsigned int i = 0; i < q_size; ++i) {
    consumeIteration();
  }

  EXPECT_FLOAT_EQ(sum_from_producer, sum_from_consumer);
}

TEST_P(IterQueueScenarios, produceAndConsumeSyncTwice_p) {
  auto q_size = iq->slots();
  auto q_size_in_sample = q_size * iq->batch();
  /// step1: fill buffer to the queue (filling full queue)
  for (unsigned int i = 0; i < q_size_in_sample; ++i) {
    produceSample(i);
  }

  /// step2: consume the filled buffer from the queue
  for (unsigned int i = 0; i < q_size; ++i) {
    consumeIteration();
  }

  /// step3: fill buffer to the queue (filling full queue)
  for (unsigned int i = q_size_in_sample; i < q_size_in_sample * 2; ++i) {
    produceSample(i);
  }

  /// step4: consume the filled buffer from the queue
  for (unsigned int i = 0; i < q_size; ++i) {
    consumeIteration();
  }

  EXPECT_FLOAT_EQ(sum_from_producer, sum_from_consumer);
}

TEST_P(IterQueueScenarios, produceAndConsumeSyncMixed_p) {
  auto q_size = iq->slots();
  auto q_size_in_sample = q_size * iq->batch();
  /// step1: fill buffer to the queue (filling half samples)
  for (unsigned int i = 0; i < q_size_in_sample / 2; ++i) {
    produceSample(i);
  }

  /// step2: consume the filled buffer from the queue
  for (unsigned int i = 0; i < q_size / 2; ++i) {
    consumeIteration();
  }

  /// step3: fill buffer to the queue (filling rest half samples)
  for (unsigned int i = q_size_in_sample / 2; i < q_size_in_sample; ++i) {
    produceSample(i);
  }

  /// step4: consume the filled buffer from the queue
  for (unsigned int i = q_size / 2; i < q_size; ++i) {
    consumeIteration();
  }

  EXPECT_FLOAT_EQ(sum_from_producer, sum_from_consumer);
}

IterQueueTestParamType multi_slot_multi_batch = {
  4 /** queue size */,
  {{3, 2, 4, 5}, {3, 4, 5, 7}} /** input_dims*/,
  {{3, 1, 1, 8}, {3, 1, 1, 2}} /** label_dims */};

IterQueueTestParamType single_slot_multi_batch = {
  1 /** queue size */,
  {{3, 2, 4, 5}, {3, 4, 5, 7}} /** input_dims*/,
  {{3, 1, 1, 8}, {3, 1, 1, 2}} /** label_dims */};

IterQueueTestParamType multi_slot_single_batch = {
  3 /** queue size */,
  {{1, 2, 4, 5}, {1, 4, 5, 7}} /** input_dims*/,
  {{1, 1, 1, 8}, {1, 1, 1, 2}} /** label_dims */};

IterQueueTestParamType single_slot_single_batch = {
  1 /** queue size */,
  {{1, 2, 4, 5}, {1, 4, 5, 7}} /** input_dims*/,
  {{1, 1, 1, 8}, {1, 1, 1, 2}} /** label_dims */};

INSTANTIATE_TEST_CASE_P(IterQueue, IterQueueScenarios,
                        ::testing::Values(multi_slot_multi_batch,
                                          single_slot_multi_batch,
                                          multi_slot_single_batch,
                                          single_slot_single_batch));

TEST(IterQueue, constructEmptySlots_n) {
  EXPECT_ANY_THROW(nntrainer::IterationQueue(0, {{1}}, {{1}}));
}

TEST(IterQueue, constructEmptyInput_n) {
  EXPECT_ANY_THROW(nntrainer::IterationQueue(1, {}, {{1}}));
}

TEST(IterQueue, constructNotConsistentBatchSizeBetweenInputs_n) {
  EXPECT_ANY_THROW(
    nntrainer::IterationQueue(1, {{3, 1, 1, 10}, {2, 1, 1, 10}}, {}));
}

TEST(IterQueue, constructNotConsistentBatchSizeInLabel_n) {
  EXPECT_ANY_THROW(nntrainer::IterationQueue(1, {{3, 1, 1, 10}, {3, 1, 1, 10}},
                                             {{2, 1, 1, 10}}));
}

TEST(IterQueue, constructNotConsistentBatchSizeInLabel2_n) {
  EXPECT_ANY_THROW(nntrainer::IterationQueue(1, {{3, 1, 1, 10}, {3, 1, 1, 10}},
                                             {{3, 1, 1, 10}, {2, 1, 1, 10}}));
}
