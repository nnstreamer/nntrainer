// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   batch_queue.h
 * @date   13 July 2021
 * @brief  This file contains thread safe queue for batch
 * @note   This file is made to easily extend to type T, although it didn't to
 * save compile time
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */
#ifndef __BATCH_QUEUE_H__
#define __BATCH_QUEUE_H__

#include <condition_variable>
#include <functional>
#include <memory>
#include <queue>
#include <shared_mutex>
#include <tuple>

#include <data_iteration.h>
#include <data_producer.h>
#include <nntrainer_log.h>
#include <tensor.h>
#include <tensor_dim.h>
namespace nntrainer {

/**
 * @brief Thread Safe batch queue Queue
 *
 */
class BatchQueue {
public:
  using T = DataProducer::Iteration; /**< Iteration as type T to leave room to
                                        extend the class to type T */

  /**
   * @brief Construct a new batch queue Queue object
   * @note this is not the size of buffersize, but it is @a
   * buffersize/batch_size the size never changes after the BatchQueue has been
   * created
   * @param queue_capacity_ max queue size
   */
  BatchQueue(unsigned int queue_capacity_);

  /**
   * @brief Construct a new batch queue Queue object
   * @note this does not copy the original queue, but only queue size
   * @param rhs batch queue queue to copy
   */
  BatchQueue(const BatchQueue &rhs);

  /**
   * @brief Copy-assign batch queue queue
   *
   * @param rhs batch queue queue to copy
   * @return BatchQueue& new queue
   */
  BatchQueue &operator=(const BatchQueue &rhs);

  /**
   * @brief push data to queue, if the batch queue is full, wait if full
   *
   * @param data data to put inside the batch queue
   */
  void wait_and_push(T &&data) noexcept;

  /**
   * @brief pop data from the queue, wait if empty
   *
   * @return std::unique_ptr<T> data
   */
  std::unique_ptr<T> wait_and_pop() noexcept;

  /**
   * @brief check if current queue is full
   *
   * @return bool true if full
   */
  bool isFull() const;

  /**
   * @brief check if current queue is empty
   *
   * @return bool true if empty
   */
  bool isEmpty() const;

private:
  unsigned int queue_capacity;
  mutable std::shared_mutex q_mutex;
  std::condition_variable_any q_reader_cv;
  std::condition_variable_any q_writer_cv;

  std::queue<std::unique_ptr<T>> q;
};

/**
 * @brief A view container that calls a callback on destruct
 * @note the callback must be noexcept, and the given underlying data must
 * outlive the lifetime of this class
 *
 * @tparam T underlying type
 */
template <typename T> class ScopedView {
public:
  /**
   * @brief Construct a new Scoped View object
   *
   * @param data_ reference of the underlying data
   * @param on_notify_ callback to be called on exit, this is not copied but
   * reused
   */
  ScopedView(T *data_, std::function<void(void)> &&on_notify_) :
    data(data_),
    on_notify(std::forward<std::function<void(void)>>(on_notify_)) {}

  ScopedView(const ScopedView &rhs) = delete;
  ScopedView &operator=(const ScopedView &rhs) = delete;

  ScopedView(ScopedView &&rhs) = default;
  ScopedView &operator=(ScopedView &&rhs) = default;

  /**
   * @brief Destroy the Scoped View object, callback is called at this time
   *
   */
  ~ScopedView() {
    try {
      on_notify();
    } catch (...) {
      ml_loge("while notifiying, error happened");
    }
  }

  /**
   * @brief get the underlying data
   *
   * @return T & reference to the underlying data
   */
  T &get() { return *data; }

  /**
   * @brief get the underlying data
   *
   * @return T & reference to the underlying data
   */
  T const &get() const { return *data; }

private:
  T *data; /**< underlying data pointer */
  std::function<void(void)>
    on_notify; /**< called when destroyed without error */
};

/**
 * @brief Iteration queue that owns the buffer for input / labels
 * @detail
 *
 * - requestEmpty() will give a ScopedView<sample>
 *     Destructing the returned object will notify the iteration that is done
 * filling the sample. Once iteration is done filling, it will internally call
 * IterationQueue::markFilled();
 * - requestFilled() will give a ScopedView<Iteration>
 *     Destructing this will notify the queue that is done used (internally
 * calls IterationQueue::markEmpty())
 *
 * @todo apply this to the databuffer
 * @todo prepare thread safe queue and apply
 */
class IterationQueue {
public:
  /**
   * @brief Construct a new Iteration Queue object
   * @note  input_dimension and label_dimension should include the batch, if
   * IterationQueue::batch() is zero, it means it's invalid
   * @param num_slots number of slots this batch queue will allocate, it should
   * be buffersize/batchsize
   * @param input_dims input dimensions
   * @param label_dims label dimensions
   */
  IterationQueue(unsigned int num_slots,
                 const std::vector<ml::train::TensorDim> &input_dims,
                 const std::vector<ml::train::TensorDim> &label_dims);

  /**
   * @brief request empty sample from the queue.
   * @note There is race condition between requesting empty, race condition with
   * mark_ready should be handled by using MT_safe queue.
   * @return ScopedView<Sample> sample view. Destroying the returned object will
   * signal the queue that the sample is filled.
   */
  ScopedView<Sample> requestEmpty();

  /**
   * @brief request filled iteration from the queue.
   * @note race condition here can be handled by using MT_safe queue
   * @return ScopedView<Iteration> Ieration view. Destroying the returned object
   * will signal the queue that the sample is done using.
   */
  ScopedView<Iteration> requestFilled();

private:
  /**
   * @brief A wrapper object around @c Iteration which marks filled when filling
   * sample is done
   * @note the given @a iteration_ and @a bq_ must outleave the lifetime of this
   * class
   *
   */
  class MarkableIteration {
  public:
    /**
     * @brief Construct a new Markable Iteration object
     *
     * @param input_dims input dimensions
     * @param label_dims label dimensions
     * @param iq_ iteration queue view to notify
     */
    MarkableIteration(const std::vector<ml::train::TensorDim> &input_dims,
                      const std::vector<ml::train::TensorDim> &label_dims,
                      IterationQueue *iq);

    /**
     * @brief Construct a new Markable Iteration object
     *
     * @param rhs right side to move
     */
    MarkableIteration(MarkableIteration &&rhs);

    /**
     * @brief Move Assignement operator
     *
     * @param rhs rhs to move
     * @return MarkableIteration& markable iteration
     */
    MarkableIteration &operator=(MarkableIteration &&rhs);

    /**
     * @brief mark iteration that one sample is filled
     * @todo make this function noexcept
     */
    void markSampleFilled() /** noexcept */;

    /**
     * @brief get underlying iteration
     *
     * @return Iteration& iteration
     */
    Iteration &get() { return iteration; }

  private:
    mutable std::mutex notify_mutex;
    Iteration iteration;
    IterationQueue *iq;
    unsigned int num_observed;
  };

  /**
   * @brief mark the given iteration filled
   * @todo make this noexcept with the thread safe queue
   * @param iteration iteration to mark it as filled
   */
  void markFilled(MarkableIteration *iteration) /** noexcept */;

  /**
   * @brief mark the given iteration empty
   * @todo make this noexcept with the thread safe queue
   * @param iteration iteration to mark it as emptied
   */
  void markEmpty(MarkableIteration *iteration) /** noexcept */;

  std::vector<MarkableIteration> iterations; /** allocated iterations */
  MarkableIteration *being_filled; /**< iteration that is being filled now */

  std::vector<Sample>::iterator current_iterator;

  /// @todo use mt safe queue
  std::queue<MarkableIteration *> empty_q;  /** iterations to be filled */
  std::queue<MarkableIteration *> filled_q; /** iterations to be served */
};

} // namespace nntrainer

#endif // __BATCH_QUEUE_H__
