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

#include <queue>

#include <condition_variable>
#include <data_producers.h>
#include <memory>
#include <shared_mutex>

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

  /**
   * @todo consider using circular buffer if this is too slow
   *
   */
  std::queue<std::unique_ptr<T>> q;
};

} // namespace nntrainer

#endif // __BATCH_QUEUE_H__
