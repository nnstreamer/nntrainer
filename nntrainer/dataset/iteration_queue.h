// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   iteration_queue.h
 * @date   13 July 2021
 * @brief  This file contains thread safe queue for a single iteration
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */
#ifndef __ITERATION_QUEUE_H__
#define __ITERATION_QUEUE_H__

#include <atomic>
#include <condition_variable>
#include <functional>
#include <memory>
#include <queue>
#include <shared_mutex>
#include <stdexcept>
#include <tuple>

#include <data_iteration.h>
#include <data_producer.h>
#include <nntrainer_log.h>
#include <tensor.h>
#include <tensor_dim.h>
namespace nntrainer {

/**
 * @brief Thread Safe Queue implementation dedicated for the non-owing pointer
 *
 * @tparam original type of the view (T * will be pushed and pop)
 */
template <typename T> class ViewQueue {
public:
  /**
   * @brief Construct a new queue
   */
  ViewQueue() : q() {}

  /**
   * @brief push data to queue
   *
   * @param data data to put
   */
  void push(T *data) {
    {
      std::unique_lock<std::shared_mutex> lk(q_mutex);
      q.push(data);
    }

    q_cv.notify_one();
  }

  /**
   * @brief pop data from the queue, wait if empty
   * @note when fail to get, this will return nullptr (eg) when interrupt
   * happens)
   * @return T* view of the data
   */
  T *waitAndPop() {
    std::unique_lock<std::shared_mutex> lk(q_mutex);
    q_cv.wait(lk, [this] { return !q.empty(); });
    auto ptr = q.front();
    q.pop();

    return ptr;
  }

  /**
   * @brief check if current queue is empty
   *
   * @return bool true if empty
   */
  bool isEmpty() const {
    std::shared_lock<std::shared_mutex> lk(q_mutex);
    return q.empty();
  }

  /**
   * @brief check if current queue is empty
   *
   * @return bool true if empty
   */
  typename std::queue<T *>::size_type size() const {
    std::shared_lock<std::shared_mutex> lk(q_mutex);
    return q.size();
  }

private:
  mutable std::shared_mutex q_mutex;
  std::condition_variable_any q_cv;

  std::queue<T *> q;
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
   * @param on_notify_ callback to be called on exit
   * @param on_error_ callback to be called on error
   */
  ScopedView(T *data_, std::function<void(void)> &&on_notify_ = nullptr,
             std::function<void(void)> &&on_error_ = nullptr) :
    data(data_),
    on_notify(std::forward<std::function<void(void)>>(on_notify_)),
    on_error(std::forward<std::function<void(void)>>(on_error_)) {}

  ScopedView(const ScopedView &rhs) = delete;
  ScopedView &operator=(const ScopedView &rhs) = delete;

  ScopedView(ScopedView &&rhs) = default;
  ScopedView &operator=(ScopedView &&rhs) = default;

  /**
   * @brief check if scoped view contains any underlying data
   *
   * @return bool if data is empty
   */
  bool isEmpty() { return data == nullptr; }

  /**
   * @brief Destroy the Scoped View object, callback is called at this time
   *
   */
  ~ScopedView() {
    try {
      if (std::uncaught_exceptions()) {
        if (on_error) {
          on_error();
        }
      } else {
        if (on_notify) {
          on_notify();
        }
      }
    } catch (...) {
      ml_loge("while handling on_notify or on_error, error happened");
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
  std::function<void(void)> on_error; /**< called when destroyed with error */
};

/**
 * @brief Iteration queue that owns the buffer for input / labels
 * @details
 *
 * - requestEmptySlot() will give a ScopedView<sample>
 *     Destructing the returned object will notify the iteration that is done
 * filling the sample. Once iteration is done filling, it will internally call
 * IterationQueue::markFilled();
 * - requestFilledSlot() will give a ScopedView<Iteration>
 *     Destructing this will notify the queue that is done used (internally
 * calls IterationQueue::markEmpty())
 *
 * @details For an iteration there can be four state.
 * 1. The buffer is empty, waiting to be filled (will be in empty_q)
 * 2. The buffer is being filled sample by sample, waiting to be marked as
 * filled.
 * 3. The buffer is filled, waiting to be served (will be in filled_q)
 * 4. The buffer is being served, waiting to be marked as emptied.
 * @todo apply this to the databuffer
 * @todo handle error case: 1. when ScopedView<Sample> has met throw
 *                          2. when ScopedView<Iteration> has met throw
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
   * @brief Destroy the Iteration Queue object
   *
   */
  ~IterationQueue();

  /**
   * @brief request empty sample from the queue.
   * @note User must check if ScopedView actually has a value by calling
   * copedView::isEmpty()
   * @return ScopedView<Sample> sample view. ScopedView::isEmpty() == true
   * if there is no more data coming. Destroying the returned object will
   * signal the queue that the sample is filled.
   */
  ScopedView<Sample> requestEmptySlot();

  /**
   * @brief request filled iteration from the queue.
   * @note User must check if ScopedView actually has a value by calling
   * copedView::isEmpty()
   * @return ScopedView<Iteration> Ieration view. ScopedView::isEmpty() == true
   * if there is no more data coming. Destroying the returned object will
   * signal the queue that the sample is done using.
   *
   */
  ScopedView<Iteration> requestFilledSlot();

  /**
   * @brief get slot size, slot size is number of batches inside the queue
   *
   * @return unsigned int num slot
   */
  unsigned int slots() { return iterations.size(); }

  /**
   * @brief get size of batch for one iteration
   *
   * @return unsigned int size of batch
   */
  unsigned int batch() { return batch_size; }

  /**
   * @brief notifyEndOfRequest, when the producing by requestEmptySlot has
   * finished.
   * @note It is important that the owner of this class must ensure that there
   * will be no more requestEmptySlot call after this. This means that, in case
   * of multiple workers, the manager of the worker(producer) must know every
   * producer has finished. and call this function other than each worker call
   * this function.
   *
   */
  void notifyEndOfRequestEmpty();

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
     * @brief reset num observation and internal batch size of iteration
     *
     */
    void reset();

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
     * @brief update end sample to the given iterator and mark last
     * @note after updating end iterator, this can be markFilled() if every
     * sample is already filled
     *
     * @param iterator non-inclusive iterator to mark the last
     */
    void setEndSample(std::vector<Sample>::iterator sample_iterator);

    /**
     * @brief get underlying iteration
     *
     * @return Iteration& iteration
     */
    Iteration &get() { return iteration; }

  private:
    unsigned int num_observed; /**< number of observed samples which were passed
                                  to the callee and notified done filling */
    mutable std::mutex
      notify_mutex;      /**< mutex which should be locked when try to notify */
    Iteration iteration; /**< underlying iteration that this class owns */
    IterationQueue *iq;  /**< view of iteration queue */
  };

  /**
   * @brief Queue running state enum
   *
   */
  enum class FlowState {
    FLOW_STATE_OPEN = 0,           /**< nothing */
    FLOW_STATE_STOP_REQUESTED = 1, /**< request stop */
    FLOW_STATE_STOPPED = 2,        /**< queue is fully stopped */
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

  std::vector<MarkableIteration> iterations; /**< allocated iterations */
  MarkableIteration *being_filled; /**< last iteration that is being filled */
  std::vector<Sample>::iterator
    current_iterator; /**< current sample iteration of being_filled */

  mutable std::mutex empty_mutex; /**< mutex to be used when it is mutually
                                     exclusive to the requesting empty slots */
  unsigned int
    num_being_filled; /**< number of iteration that is in being_filled state */
  mutable std::mutex
    filled_mutex; /**< mutex to be used when it is mutually exclusive to the
                     requesting filled slots */
  std::condition_variable_any
    notify_emptied_cv; /**< conditional variable to wait based on the
                           num_being_filled */
  std::atomic<FlowState> flow_state; /**< flow state of the queue */

  unsigned int batch_size;
  ViewQueue<MarkableIteration> empty_q;  /**< iterations to be filled */
  ViewQueue<MarkableIteration> filled_q; /**< iterations to be served */
};

} // namespace nntrainer

#endif // __ITERATION_QUEUE_H__
