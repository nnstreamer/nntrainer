// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   iteration_queue.cpp
 * @date   13 July 2021
 * @brief  This file contains thread safe queue
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */
#include <chrono>
#include <iteration_queue.h>

#include <mutex>
#include <nntrainer_error.h>
#include <shared_mutex>

using namespace std::literals::chrono_literals;

namespace nntrainer {

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
  batch_size = iterations.front().get().batch();
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

ScopedView<Sample> IterationQueue::requestEmptySlot() {
  std::scoped_lock lg(empty_mutex);
  auto current_flow_state = flow_state.load();
  NNTR_THROW_IF(current_flow_state != FlowState::FLOW_STATE_OPEN,
                std::invalid_argument)
    << "the queue expect state of "
    << static_cast<unsigned>(FlowState::FLOW_STATE_OPEN) << " but met "
    << static_cast<unsigned>(current_flow_state);

  /// below is useful information when debugging iteration queue, but there will
  /// be too much log if we turn the log on. so leaving it as a comment for now.
  // std::cout << "[requestEmptySlot] empty_q.size(): " << empty_q.size()
  // << " being_filled: " << num_being_filled
  // << " filled_q.size():  " << filled_q.size() << '\n';

  if (being_filled == nullptr ||
      current_iterator + 1 == being_filled->get().end()) {
    being_filled = empty_q.waitAndPop();
    being_filled->reset();
    num_being_filled++;
    current_iterator = being_filled->get().begin();
  } else {
    current_iterator++;
  }

  auto view = ScopedView<Sample>(
    &(*current_iterator),
    [current_being_filed = this->being_filled] {
      current_being_filed->markSampleFilled();
    },
    [this, current_being_filled = this->being_filled] {
      std::unique_lock lg(empty_mutex);
      this->markEmpty(current_being_filled);
      num_being_filled--;
      notify_emptied_cv.notify_all();
    });
  return view;
}

ScopedView<Iteration> IterationQueue::requestFilledSlot() {
  std::scoped_lock lock(filled_mutex);

  /// below is useful information when debugging iteration queue, but there will
  /// be too much log if we turn the log on. so leaving it as a comment for now.
  // std::cout << "[requestFilledSlot] empty_q.size(): " << empty_q.size()
  // << " num being filled: " << num_being_filled
  // << " filled_q.size(): " << filled_q.size() << '\n';
  if (flow_state.load() == FlowState::FLOW_STATE_STOPPED) {
    return ScopedView<Iteration>(nullptr);
  }

  auto iteration = filled_q.waitAndPop();
  if (iteration == nullptr) {
    auto stop_request_state = FlowState::FLOW_STATE_STOP_REQUESTED;
    bool exchange_result = flow_state.compare_exchange_strong(
      stop_request_state, FlowState::FLOW_STATE_STOPPED);
    NNTR_THROW_IF(!exchange_result, std::runtime_error)
      << "the queue has either already stopped or running, but trying stopping "
         "without requesting stop, queue size: "
      << iterations.size() << " num currently empty: " << empty_q.size()
      << " filled_q.size(): " << filled_q.size();

    return ScopedView<Iteration>(nullptr);
  }

  return ScopedView<Iteration>(
    &iteration->get(), [this, iteration] { markEmpty(iteration); },
    [this, iteration] {
      std::unique_lock lock(filled_mutex);
      flow_state.store(FlowState::FLOW_STATE_STOPPED);
      markEmpty(iteration);
    });
}

void IterationQueue::notifyEndOfRequestEmpty() {
  std::unique_lock lg(empty_mutex);
  auto open_state = FlowState::FLOW_STATE_OPEN;

  /// we have to defined ordering of having stop_requested -> push nullptr to
  /// filled_q -> stopped so when the case of changing to stopped it has to push
  /// nullptr to empty_q, and filled_q to wake them up and stop. this has
  /// potential cases that weren't considered. let's change this to a simpler
  /// mechanisms to wait on conditional variable.
  bool exchange_result = flow_state.compare_exchange_strong(
    open_state, FlowState::FLOW_STATE_STOP_REQUESTED);
  NNTR_THROW_IF(!exchange_result, std::invalid_argument)
    << "the queue expect state of " << static_cast<unsigned>(open_state)
    << " but met " << static_cast<unsigned>(flow_state.load());
  /// below is useful information when debugging iteration queue, but there will
  /// be too much log if we turn the log on. so leaving it as a comment for now.
  // std::cout << "[notifyEnd] empty_q.size(): " << empty_q.size()
  //           << " num being filled: " << num_being_filled
  //           << " filled_q.size(): " << filled_q.size() << '\n';

  if (being_filled) {
    being_filled->setEndSample(current_iterator + 1);
  }
  notify_emptied_cv.wait(lg, [this] { return num_being_filled == 0; });
  filled_q.push(nullptr);
}

void IterationQueue::markFilled(MarkableIteration *iteration) {
  std::unique_lock lg(empty_mutex);
  num_being_filled--;
  filled_q.push(iteration);
  lg.unlock();
  notify_emptied_cv.notify_all();
}

void IterationQueue::markEmpty(MarkableIteration *iteration) {
  empty_q.push(iteration);
}

IterationQueue::MarkableIteration::MarkableIteration(
  const std::vector<ml::train::TensorDim> &input_dims,
  const std::vector<ml::train::TensorDim> &label_dims, IterationQueue *iq) :
  num_observed(0), iteration(input_dims, label_dims), iq(iq) {}

IterationQueue::MarkableIteration::MarkableIteration(MarkableIteration &&rhs) :
  iteration(std::move(rhs.iteration)), iq(rhs.iq) {
  std::lock_guard notify_lock_guard(notify_mutex);
  num_observed = rhs.num_observed;
}

void IterationQueue::MarkableIteration::reset() {
  std::lock_guard notify_lock_guard(notify_mutex);
  num_observed = 0;
  iteration.setEndSample();
}

IterationQueue::MarkableIteration &
IterationQueue::MarkableIteration::operator=(MarkableIteration &&rhs) {
  if (this == &rhs) {
    return *this;
  }
  std::scoped_lock lock(this->notify_mutex, rhs.notify_mutex);
  std::swap(iteration, rhs.iteration);
  std::swap(iq, rhs.iq);
  std::swap(num_observed, rhs.num_observed);
  return *this;
}

void IterationQueue::MarkableIteration::markSampleFilled() {
  std::unique_lock notify_lock_guard(notify_mutex);
  num_observed++;
  if (num_observed == iteration.batch()) {
    num_observed = 0;
    notify_lock_guard.unlock();
    iq->markFilled(this);
    notify_lock_guard.lock();
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
#if DEBUG
    NNTR_THROW_IF_CLEANUP(iq->empty_mutex.try_lock(), std::runtime_error,
                          [this] { iq->empty_mutex.unlock(); })
      << "iteration queue must be locked already but empty_mutex is not "
         "locked.";
#endif
    /// warning: iq has to be locked with iq->empty_mutex
    iq->num_being_filled--;
    iq->filled_q.push(this);
    iq->notify_emptied_cv.notify_all();
    num_observed = 0;
  }
}

} // namespace nntrainer
