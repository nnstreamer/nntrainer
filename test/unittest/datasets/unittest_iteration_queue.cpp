// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file unittest_iteration_queue.cpp
 * @date 12 July 2021
 * @brief Iteration Queue Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <gtest/gtest.h>

#include <iteration_queue.h>
#include <random_data_producers.h>
#include <tensor.h>

#include <algorithm>
#include <future>
#include <nntrainer_test_util.h>
#include <thread>
#include <tuple>
#include <vector>

using namespace std::chrono_literals;

/** THIS number can be divided with 3 and 4 is multiple of current multiple
 * batch scenario setup, as a @todo this should be better part of a
 * IterQueueTestParamType */
static constexpr unsigned int DATA_SIZE = 384u;
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
    producer->setProperty({"num_samples=512"});
    sample_getter = producer->finalize(input_dims, label_dims);
    this->input_dims = input_dims;
    this->label_dims = label_dims;
    sum_from_producer = 0;
    sum_from_consumer = 0;
  }

  virtual void
  produceSample(unsigned int size,
                const std::chrono::milliseconds *duration = nullptr) {
    auto sample_view = iq->requestEmptySlot();
    if (sample_view.isEmpty()) {
      throw std::runtime_error("sample_view is empty!");
    }
    auto &sample = sample_view.get();
    auto &inputs = sample.getInputsRef();
    auto &labels = sample.getLabelsRef();
    if (duration) {
      std::this_thread::sleep_for(*duration);
    }
    std::lock_guard<std::mutex> lg(producer_mutex);
    sample_getter(size, inputs, labels);
    sum_from_producer += getSum(inputs, labels);
  }

  virtual std::future<void>
  produceSampleAfter(unsigned int size,
                     const std::chrono::milliseconds &duration) {
    return std::async(std::launch::async, [this, size, duration] {
      produceSample(size, &duration);
    });
  }

  virtual std::future<bool>
  consumeIterationAfter(const std::chrono::milliseconds &duration) {
    return std::async(std::launch::async, [this, duration] {
      std::this_thread::sleep_for(duration);
      return consumeIteration();
    });
  }

  virtual bool consumeIteration() {
    auto iter_view = iq->requestFilledSlot();
    if (iter_view.isEmpty()) {
      return false;
    }

    auto &iter = iter_view.get();
    auto &inputs = iter.getInputsRef();
    auto &labels = iter.getLabelsRef();
    std::lock_guard<std::mutex> lg(consumer_mutex);
    sum_from_consumer += getSum(inputs, labels);
    return true;
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

  /** @todo currently there is an error between those two around 1e-8, this
   * should be changed to integer based comparison */
  long double
    sum_from_producer; /**< sum of all the dataset from producer side*/
  long double
    sum_from_consumer; /**< sum of all the dataset from consumer side */

  mutable std::mutex producer_mutex, consumer_mutex;
  nntrainer::DataProducer::Generator sample_getter;
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

TEST_P(IterQueueScenarios,
       produceAndConsumAsyncForDeterminedSizeConsumerRunningFirst_p) {

  auto producer = std::async(std::launch::async, [this]() {
    std::this_thread::sleep_for(50ms);
    for (unsigned int i = 0u; i < DATA_SIZE; ++i) {
      produceSample(i);
    }
  });

  auto consumer = std::async(std::launch::async, [this]() {
    for (unsigned int i = 0u; i < DATA_SIZE / iq->batch(); ++i) {
      consumeIteration();
    }
  });

  producer.get();
  consumer.get();

  EXPECT_FLOAT_EQ(sum_from_producer, sum_from_consumer);
}

TEST_P(IterQueueScenarios,
       produceAndConsumAsyncForDeterminedSizeProducerRunningFirst_p) {

  auto producer = std::async(std::launch::async, [this]() {
    for (unsigned int i = 0u; i < DATA_SIZE; ++i) {
      produceSample(i);
    }
  });

  auto consumer = std::async(std::launch::async, [this]() {
    std::this_thread::sleep_for(50ms);
    for (unsigned int i = 0u; i < DATA_SIZE / iq->batch(); ++i) {
      consumeIteration();
    }
  });

  producer.get();
  consumer.get();

  EXPECT_FLOAT_EQ(sum_from_producer, sum_from_consumer);
}

TEST_P(IterQueueScenarios,
       produceAndConsumAsyncForUnknownSizeProducerRunningFirst_p) {
  auto producer = std::async(std::launch::async, [this]() {
    for (unsigned int i = 0u; i < DATA_SIZE; ++i) {
      produceSample(i);
    }
    iq->notifyEndOfRequestEmpty();
  });

  auto consumer = std::async(std::launch::async, [this]() {
    std::this_thread::sleep_for(50ms);
    while (consumeIteration()) {
    }
  });

  producer.get();
  consumer.get();

  EXPECT_FLOAT_EQ(sum_from_producer, sum_from_consumer);
}

TEST_P(IterQueueScenarios,
       produceAndConsumAsyncForUnknownSizeConsumerRunningFirst_p) {
  auto producer = std::async(std::launch::async, [this]() {
    std::this_thread::sleep_for(50ms);
    for (unsigned int i = 0u; i < DATA_SIZE; ++i) {
      produceSample(i);
    }
    iq->notifyEndOfRequestEmpty();
  });

  auto consumer = std::async(std::launch::async, [this]() {
    while (consumeIteration()) {
    }
  });

  producer.get();
  consumer.get();

  EXPECT_FLOAT_EQ(sum_from_producer, sum_from_consumer);
}

TEST_P(IterQueueScenarios, produceAndConsumPartiallyFilledBatch_p) {
  auto b = iq->batch();
  if (b == 1) {
    return; /// if batch is one, there is no partially filled batch
  }

  auto getSumOfPartiallyFilledTensor =
    [](const nntrainer::Tensor &t, unsigned int actual_batch) -> long double {
    long double sum = 0;
    nntrainer::Tensor result = t.sum_by_batch();
    for (unsigned int i = 0; i < actual_batch; ++i) {
      sum += result.getValue(i, 0, 0, 0);
    }
    return sum;
  };

  for (unsigned int i = 0; i < b - 1; ++i) {
    produceSample(i);
  }

  iq->notifyEndOfRequestEmpty();
  {
    auto iter_view = iq->requestFilledSlot();
    if (iter_view.isEmpty()) {
      throw std::invalid_argument("iter view is empty!");
    }
    nntrainer::Iteration &it = iter_view.get();

    auto &inputs = it.getInputsRef();
    auto &labels = it.getLabelsRef();

    for (auto &input : inputs) {
      sum_from_consumer += getSumOfPartiallyFilledTensor(input, it.batch());
    }
    for (auto &label : labels) {
      sum_from_consumer += getSumOfPartiallyFilledTensor(label, it.batch());
    }
  }
  EXPECT_FALSE(consumeIteration());
  EXPECT_FLOAT_EQ(sum_from_producer, sum_from_consumer);
}

/**
 * When calling notifytEndOfRequestEmpty(), there are four possible states the
 * queue is in.
 *
 * notified there will be no filling while...
 * 1. the last buffer is already consumed and already marked as emptied
 * 2. the last buffer has moved to filled_q.
 * 3. the last buffer is being filled, and there are multiple buffers being
 * filled
 * 4. the last buffer is being served
 *
 */

TEST_P(IterQueueScenarios, caseOneNotifyAfterConsumingIsFinished_p) {

  std::promise<void> consume_done;
  auto producer = std::async(std::launch::async, [this, &consume_done]() {
    for (unsigned int i = 0u; i < DATA_SIZE; ++i) {
      produceSample(i);
    }
    consume_done.get_future().get();
    iq->notifyEndOfRequestEmpty();
  });

  auto consumer = std::async(std::launch::async, [this, &consume_done]() {
    for (unsigned int i = 0u; i < DATA_SIZE / iq->batch(); ++i) {
      consumeIteration();
    }
    consume_done.set_value();
    EXPECT_FALSE(consumeIteration());
  });

  producer.get();
  consumer.get();

  EXPECT_FLOAT_EQ(sum_from_producer, sum_from_consumer);
}

TEST_P(IterQueueScenarios, caseTwoNotifyAfterTheLastBufferHasMovedToFilledQ_p) {
  std::vector<std::future<void>> producer_result;

  unsigned int number_of_producing = iq->batch() * iq->slots();
  producer_result.reserve(number_of_producing);
  for (unsigned int i = 0; i < number_of_producing; ++i) {
    producer_result.push_back(produceSampleAfter(i, 50ms));
  }

  for (auto &fut : producer_result) {
    fut.get();
  }
  iq->notifyEndOfRequestEmpty();

  for (unsigned int i = 0; i < iq->slots(); ++i) {
    EXPECT_TRUE(consumeIteration());
  }
  EXPECT_FALSE(consumeIteration());

  EXPECT_FLOAT_EQ(sum_from_producer, sum_from_consumer);
}

TEST_P(IterQueueScenarios, caseThreeNotifyAfterTheLastBufferIsBeingFilled_p) {
  std::future<void> notify_result;
  {
    std::queue<nntrainer::ScopedView<nntrainer::Sample>> scoped_views;
    unsigned int number_of_producing = iq->batch() * iq->slots();
    for (unsigned int i = 0; i < number_of_producing; ++i) {
      scoped_views.push(iq->requestEmptySlot());
      if (scoped_views.back().isEmpty()) {
        throw std::runtime_error("sample was empty");
      }
      auto &sample = scoped_views.back().get();
      auto &inputs = sample.getInputsRef();
      auto &labels = sample.getLabelsRef();
      sample_getter(i, inputs, labels);
      sum_from_producer += getSum(inputs, labels);
    }

    notify_result =
      std::async(std::launch::async, [this] { iq->notifyEndOfRequestEmpty(); });
    std::this_thread::sleep_for(50ms);
    /// delaying destroying scoped_views to simulate samples are in
    /// the state of being filled
  }
  notify_result.get();

  for (unsigned int i = 0; i < iq->slots(); ++i) {
    EXPECT_TRUE(consumeIteration());
  }
  EXPECT_FALSE(consumeIteration());
  EXPECT_FLOAT_EQ(sum_from_producer, sum_from_consumer);
}

TEST_P(IterQueueScenarios, caseFourNotifyAfterTheLastBufferIsBeingServed_p) {
  std::future<void> notify_result;
  unsigned int number_of_producing = iq->batch() * iq->slots();

  std::queue<nntrainer::ScopedView<nntrainer::Sample>> scoped_views;
  for (unsigned int i = 0; i < number_of_producing; ++i) {
    produceSample(i);
  }

  for (unsigned int i = 0; i < iq->slots() - 1; ++i) {
    EXPECT_TRUE(consumeIteration());
  }
  {
    auto iter_view = iq->requestFilledSlot();
    notify_result =
      std::async(std::launch::async, [this] { iq->notifyEndOfRequestEmpty(); });
    if (iter_view.isEmpty()) {
      throw std::invalid_argument("iter_view is empty!");
    }
    auto &iter = iter_view.get();
    auto &inputs = iter.getInputsRef();
    auto &labels = iter.getLabelsRef();
    sum_from_consumer += getSum(inputs, labels);
    EXPECT_FALSE(consumeIteration());

    std::this_thread::sleep_for(50ms);
    /// delay here to delay destroying iter_view to simulate
    /// notifyEndOfRequestEmpty() is being called during destroying the last
    /// buffer
  }

  notify_result.get();
  EXPECT_FLOAT_EQ(sum_from_producer, sum_from_consumer);
}

/** @todo add a test that simulates multiple workers, for now this is impossible
 * because we don't have thread safe worker */

TEST_P(IterQueueScenarios, notifyEndTwice_n) {
  iq->notifyEndOfRequestEmpty();
  EXPECT_ANY_THROW(iq->notifyEndOfRequestEmpty());
}

TEST_P(IterQueueScenarios, notifyEndAndTryRequestEmpty_n) {
  iq->notifyEndOfRequestEmpty();
  EXPECT_ANY_THROW(iq->requestEmptySlot());
}

TEST_P(IterQueueScenarios, ScopedViewSampleHandlesThrowWhileFillingFails_n) {
  std::future<void> notify_result;

  std::promise<void> ready_promise, t1_ready_promise, t2_ready_promise;
  std::shared_future<void> ready_future(ready_promise.get_future());
  auto request_fail = std::async(std::launch::async, [&, this] {
    t1_ready_promise.set_value();
    ready_future.wait();
    auto sample_view = iq->requestEmptySlot();
    throw std::invalid_argument("while filling, it failed");
  });

  auto try_consume = std::async(std::launch::async, [&, this] {
    t2_ready_promise.set_value();
    ready_future.wait();
    auto iter_view = iq->requestFilledSlot();
    EXPECT_TRUE(iter_view.isEmpty());
  });

  t1_ready_promise.get_future().wait();
  t2_ready_promise.get_future().wait();
  ready_promise.set_value();

  EXPECT_THROW(request_fail.get(), std::invalid_argument);
  iq->notifyEndOfRequestEmpty();
  try_consume.wait();
}

TEST_P(IterQueueScenarios, ScopedViewIterationHandlesThrowWhileFillingFails_n) {
  std::future<void> notify_result;

  std::promise<void> ready_promise, t1_ready_promise, t2_ready_promise;
  std::shared_future<void> ready_future(ready_promise.get_future());
  auto feed_data = std::async(std::launch::async, [&, this] {
    t1_ready_promise.set_value();
    ready_future.wait();
    for (unsigned i = 0; i < iq->batch(); ++i) {
      auto sample_view = iq->requestEmptySlot();
    }
  });

  auto consume_fail = std::async(std::launch::async, [&, this] {
    t2_ready_promise.set_value();
    ready_future.wait();
    auto iter_view = iq->requestFilledSlot();
    throw std::invalid_argument("while using, it failed");
  });

  t1_ready_promise.get_future().wait();
  t2_ready_promise.get_future().wait();
  ready_promise.set_value();

  EXPECT_THROW(consume_fail.get(), std::invalid_argument);
  feed_data.wait();
  EXPECT_ANY_THROW(iq->requestEmptySlot());
  EXPECT_TRUE(iq->requestFilledSlot().isEmpty());
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

GTEST_PARAMETER_TEST(IterQueue, IterQueueScenarios,
                     ::testing::Values(multi_slot_multi_batch,
                                       single_slot_multi_batch,
                                       multi_slot_single_batch,
                                       single_slot_single_batch));

TEST(IterQueue, constructEmptySlots_01_n) {
  EXPECT_ANY_THROW(nntrainer::IterationQueue(0, {}, {}));
}

TEST(IterQueue, constructEmptySlots_02_n) {
  EXPECT_ANY_THROW(nntrainer::IterationQueue(0, {}, {{1}}));
}

TEST(IterQueue, constructEmptySlots_03_n) {
  EXPECT_ANY_THROW(nntrainer::IterationQueue(0, {{1}}, {{1}}));
}

TEST(IterQueue, constructEmptyInput_n) {
  EXPECT_ANY_THROW(nntrainer::IterationQueue(1, {}, {{1}}));
}

TEST(IterQueue, constructNotConsistentBatchSizeBetweenInputs_01_n) {
  EXPECT_ANY_THROW(
    nntrainer::IterationQueue(1, {{3, 1, 1, 10}, {2, 1, 1, 10}}, {}));
}

TEST(IterQueue, constructNotConsistentBatchSizeBetweenInputs_02_n) {
  EXPECT_ANY_THROW(nntrainer::IterationQueue(1, {{3, 1}, {2, 1}}, {{1, 0}}));
}

TEST(IterQueue, constructNotConsistentBatchSizeInLabel_01_n) {
  EXPECT_ANY_THROW(nntrainer::IterationQueue(1, {{3, 1, 1, 10}, {3, 1, 1, 10}},
                                             {{2, 1, 1, 10}}));
}

TEST(IterQueue, constructNotConsistentBatchSizeInLabel_02_n) {
  EXPECT_ANY_THROW(nntrainer::IterationQueue(1, {{3, 1, 1, 10}, {3, 1, 1, 10}},
                                             {{3, 1, 1, 10}, {2, 1, 1, 10}}));
}
