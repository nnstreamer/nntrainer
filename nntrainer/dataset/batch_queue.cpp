// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   batch_queue.cpp
 * @date   13 July 2021
 * @brief  This file contains thread safe queue
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */
#include <batch_queue.h>
#include <chrono>

#include <mutex>
#include <nntrainer_error.h>
#include <shared_mutex>

using namespace std::literals::chrono_literals;

namespace nntrainer {

BatchQueue::BatchQueue(unsigned int queue_capacity_) :
  queue_capacity(queue_capacity_) {
  NNTR_THROW_IF(queue_capacity == 0, std::invalid_argument)
    << "queue capacity of zero not supported!";
}

BatchQueue::BatchQueue(const BatchQueue &rhs) :
  queue_capacity(rhs.queue_capacity) {}

BatchQueue &BatchQueue::operator=(const BatchQueue &rhs) {
  if (this == &rhs) {
    return *this;
  }
  this->queue_capacity = rhs.queue_capacity;
  return *this;
}

void BatchQueue::wait_and_push(T &&data) noexcept {
  std::unique_lock<std::shared_mutex> lk(q_mutex);
  q_writer_cv.wait(lk, [this] { return q.size() != queue_capacity; });
  q.push(std::make_unique<T>(data));
  q_reader_cv.notify_one();
}

std::unique_ptr<BatchQueue::T> BatchQueue::wait_and_pop() noexcept {
  std::unique_lock<std::shared_mutex> lk(q_mutex);
  q_reader_cv.wait(lk, [this] { return !q.empty(); });

  /// @note this invalidates q.front(), but it is okay because it is locked and
  /// popped right away
  auto ptr = std::move(q.front());
  q.pop();
  q_writer_cv.notify_one();

  return ptr;
}

bool BatchQueue::isFull() const {
  std::shared_lock<std::shared_mutex> lk(q_mutex);
  return queue_capacity == q.size();
}

bool BatchQueue::isEmpty() const {
  std::shared_lock<std::shared_mutex> lk(q_mutex);
  return q.empty();
}

} // namespace nntrainer
