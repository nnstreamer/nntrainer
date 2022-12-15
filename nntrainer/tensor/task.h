// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2022 Jiho Chu <jiho.chu@samsung.com>
 *
 * @file   task.h
 * @date   04 Nov 2022
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jiho Chu <jiho.chu@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Task class
 *
 */

#ifndef __TASK_H__
#define __TASK_H__

#include <atomic>
#include <chrono>
#include <functional>

namespace nntrainer {

/**
 * @class   Task
 * @brief   task class
 */
class Task {
public:
  /**
   * @brief work function
   */
  using Work = std::function<int(std::atomic_bool &running, void *data)>;

  /**
   * @brief task type
   */
  enum Type {
    SYNC,
    ASYNC,
  };

  /**
   * @brief task state
   */
  enum State {
    CREATED,
    STARTED,
    PROCESSING,
    DONE,
  };

  /**
   * @brief Task constructor
   *
   */
  explicit Task(Work w, void *user_data) :
    state(CREATED),
    work(w),
    data(user_data) {}

  /**
   * @brief Task destructor
   *
   */
  virtual ~Task() = default;

  /**
   * @brief Get type of task
   *
   * @result type of task @ref Task::Type
   */
  virtual const Task::Type getType(void) { return Type::SYNC; }

  /**
   * @brief Task destructor
   *
   * @return true if state is after started, else false
   */
  bool started(void) { return state >= State::STARTED; }

  /**
   * @brief Task destructor
   *
   * @return true if state is after done, else false
   */
  bool done(void) { return state >= State::DONE; }

  /**
   * @brief Get work of task
   *
   * @return work function @ref Task::Work
   */
  const Work getWork(void) { return work; }

  /**
   * @brief Task destructor
   *
   * @return user data
   */
  void *getData(void) { return data; }

  /**
   * @brief Set task state
   *
   * @param s state to be set
   */
  void setState(State s) { state = s; }

private:
  State state;
  Work work;
  void *data;
};

/**
 * @class   TaskAsync
 * @brief   Async task class
 */
template <typename T = std::chrono::milliseconds>
class TaskAsync : public Task {
public:
  enum Priority {
    HIGH,
    MID,
    LOW,
  };

  /**
   * @brief TaskAsync constructor
   *
   */
  explicit TaskAsync(Work work, void *user_data) :
    Task(work, user_data),
    timeout(T::max()),
    priority(Priority::MID) {}

  /**
   * @brief TaskAsync destructor
   *
   */
  virtual ~TaskAsync() = default;

  /**
   * @brief Get type of task
   *
   * @result type of task @ref Task::Type
   */
  virtual const Task::Type getType(void) override { return Task::Type::ASYNC; };

  /**
   * @brief Set timeout
   *
   * @param time timeout in T
   */
  virtual void setTimeout(int64_t time) { timeout = T(time); }

  /**
   * @brief Get timeout
   *
   * @param time timeout in T
   */
  virtual int64_t getTimeout(void) const { return timeout.count(); }

  /**
   * @brief Set priority
   *
   * @param pri priority to be set
   */
  virtual void setPriority(Priority pri) { priority = pri; }

  /**
   * @brief Get priority
   *
   * @return priority of task
   */
  virtual Priority getPriority() { return priority; }

private:
  T timeout;         /**< timeout time */
  Priority priority; /**< task priority */
};

} // namespace nntrainer

#endif /** __TASK_H__ */
