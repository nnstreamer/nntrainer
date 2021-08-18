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

IterationQueue::IterationQueue(
  unsigned int num_slots, const std::vector<ml::train::TensorDim> &input_dims,
  const std::vector<ml::train::TensorDim> &label_dims) :
  being_filled(nullptr),
  num_being_filled(0),
  flow_state(IterationQueue::FlowState::FLOW_STATE_OPEN) {
  NNTR_THROW_IF(num_slots == 0, std::invalid_argument)
    << "number of slots must be more then zero";

  iterations.reserve(num_slots);
  for (decltype(num_slots) i = 0; i < num_slots; ++i) {
    iterations.emplace_back(input_dims, label_dims, this);
    empty_q.push(&iterations.back());
  }
}

IterationQueue::~IterationQueue() {
  std::scoped_lock lg(empty_mutex, filled_mutex);

  /// if an iteration is not included in either empty_q or filled_q, that
  /// means it's either being filled or being served. Which means it will be
  /// dangerous to destroy @a this, we might want to wait on the destructor if
  /// we can assure this can stay no except
  if (empty_q.size() + filled_q.size() < iterations.size()) {
    ml_logw(
      "Destroying the iteration queue, while some buffers are being used");
  }
}

ScopedView<Sample> IterationQueue::requestEmpty() {
  std::scoped_lock lg(empty_mutex);
  NNTR_THROW_IF(flow_state != FlowState::FLOW_STATE_OPEN, std::invalid_argument)
    << "Calling requestEmpty() after notifyEndOfRequestEmpty() breaks "
       "invariant";

  /// below is useful information when debugging iteration queue, but there will
  /// be to much log if we turn the log on. so leaving it as a comment for now.
  // std::cout << "[requestEmpty] empty_q.size(): " << empty_q.size()
  // << " being_filled: " << num_being_filled
  // << " filled_q.size():  " << filled_q.size() << '\n';

  if (being_filled == nullptr ||
      current_iterator + 1 == being_filled->get().end()) {
    being_filled = empty_q.waitAndPop();
    num_being_filled++;
    current_iterator = being_filled->get().begin();
  } else {
    current_iterator++;
  }

  auto view = ScopedView<Sample>(&(*current_iterator),
                                 [current_being_filed = this->being_filled] {
                                   current_being_filed->markSampleFilled();
                                 });
  return view;
}

ScopedView<Iteration> IterationQueue::requestFilled() {
  std::scoped_lock lock(filled_mutex);

  /// below is useful information when debugging iteration queue, but there will
  /// be to much log if we turn the log on. so leaving it as a comment for now.
  // std::cout << "[requestFilled] empty_q.size(): " << empty_q.size()
  // << " num being filled: " << num_being_filled
  // << " filled_q.size(): " << filled_q.size() << '\n';
  if (flow_state == FlowState::FLOW_STATE_STOPPED) {
    return ScopedView<Iteration>(nullptr);
  }

  auto iteration = filled_q.waitAndPop();
  if (iteration == nullptr) {
    NNTR_THROW_IF(flow_state != FlowState::FLOW_STATE_STOP_REQUESTED,
                  std::runtime_error)
      << "the queue has either already stopped or running, but trying stopping "
         "without requesting stop, queue size: "
      << iterations.size() << " num currently empty: " << empty_q.size()
      << " num being filled: " << num_being_filled
      << " filled_q.size(): " << filled_q.size();

    flow_state = FlowState::FLOW_STATE_STOPPED;
    return ScopedView<Iteration>(nullptr);
  }

  return ScopedView<Iteration>(&iteration->get(),
                               [this, iteration] { markEmpty(iteration); });
}

void IterationQueue::notifyEndOfRequestEmpty() {
  std::unique_lock lg(empty_mutex);
  NNTR_THROW_IF(flow_state != FlowState::FLOW_STATE_OPEN, std::invalid_argument)
    << "notifyEndOfRequestEmpty() must be called once";

  /// below is useful information when debugging iteration queue, but there will
  /// be to much log if we turn the log on. so leaving it as a comment for now.
  // std::cout << "[notifyEnd] empty_q.size(): " << empty_q.size()
  //           << " num being filled: " << num_being_filled
  //           << " filled_q.size(): " << filled_q.size() << '\n';

  flow_state = FlowState::FLOW_STATE_STOP_REQUESTED;
  if (being_filled) {
    being_filled->setEndSample(current_iterator + 1);
  }
  notify_emptied_cv.wait(lg, [this] { return num_being_filled == 0; });
  filled_q.push(nullptr);
}

IterationQueue::MarkableIteration::MarkableIteration(
  const std::vector<ml::train::TensorDim> &input_dims,
  const std::vector<ml::train::TensorDim> &label_dims, IterationQueue *iq) :
  num_observed(0),
  iteration(input_dims, label_dims),
  iq(iq) {}

IterationQueue::MarkableIteration::MarkableIteration(MarkableIteration &&rhs) :
  num_observed(rhs.num_observed),
  iteration(std::move(rhs.iteration)),
  iq(rhs.iq) {}

IterationQueue::MarkableIteration &IterationQueue::MarkableIteration::
operator=(MarkableIteration &&rhs) {
  if (this == &rhs) {
    return *this;
  }
  std::scoped_lock lock(this->notify_mutex, rhs.notify_mutex);
  std::swap(iteration, rhs.iteration);
  std::swap(iq, rhs.iq);
  std::swap(num_observed, rhs.num_observed);
  return *this;
}

void IterationQueue::markFilled(MarkableIteration *iteration) /** noexcept */ {
  std::unique_lock lg(empty_mutex);
  num_being_filled--;
  filled_q.push(iteration);
  notify_emptied_cv.notify_all();
}

void IterationQueue::markEmpty(MarkableIteration *iteration) /** noexcept */ {
  empty_q.push(iteration);
}

void IterationQueue::MarkableIteration::markSampleFilled() {
  std::scoped_lock notify_lock_guard(notify_mutex);
  num_observed++;
  if (num_observed == iteration.batch()) {
    iq->markFilled(this);
    num_observed = 0;
  }
}

void IterationQueue::MarkableIteration::setEndSample(
  std::vector<Sample>::iterator sample_iterator) {
  std::scoped_lock notify_lock_guard(notify_mutex);
  auto old_batch = iteration.batch();
  if (sample_iterator != iteration.end()) {
    iteration.setEndSample(sample_iterator);
  }
  auto new_batch = iteration.batch();
  /// if batch has changed, check if this batch is partially filled and should
  /// be moved
  if (old_batch != new_batch && num_observed == new_batch) {
    /// warning: iq has to be locked with iq->empty_mutex
    iq->num_being_filled--;
    iq->filled_q.push(this);
    iq->notify_emptied_cv.notify_all();
    num_observed = 0;
  }
}

} // namespace nntrainer
