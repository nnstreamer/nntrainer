// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   profiler.h
 * @date   09 December 2020
 * @brief  Profiler related codes to be used to benchmark things
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */
#ifndef __PROFILER_H__
#define __PROFILER_H__

#include <chrono>
#include <future>
#include <iosfwd>
#include <list>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using timepoint = std::chrono::time_point<std::chrono::steady_clock>;

#ifndef PROFILE

#define PROFILE_TIME_START(event_key)
#define PROFILE_TIME_END(event_key)
#define PROFILE_TIME_REGISTER_EVENT(event_key, event_str)
#define PROFILE_MEM_ALLOC(ptr, size, str)
#define PROFILE_MEM_DEALLOC(ptr)

#else /** PROFILE */

#define PROFILE_TIME_START(event_key) \
  nntrainer::profile::Profiler::Global().start(event_key)

#define PROFILE_TIME_END(event_key) \
  nntrainer::profile::Profiler::Global().end(event_key)

#define PROFILE_TIME_REGISTER_EVENT(event_key, event_str)                 \
  do {                                                                    \
    event_key =                                                           \
      nntrainer::profile::Profiler::Global().registerTimeItem(event_str); \
  } while (0)

#define PROFILE_MEM_ALLOC(ptr, size, str) \
  nntrainer::profile::Profiler::Global().alloc(ptr, size, str)

#define PROFILE_MEM_DEALLOC(ptr) \
  nntrainer::profile::Profiler::Global().dealloc(ptr)

#endif /** PROFILE */

namespace nntrainer {

namespace profile {

enum PROFILE_EVENT {
  EVENT_TIME_START = 0,
  EVENT_TIME_END = 1,
  EVENT_MEM_ALLOC = 2,
  EVENT_MEM_DEALLOC = 3,
};

/**
 * @brief Data for each profile event
 *
 */
struct ProfileEventData {
public:
  /**
   * @brief Construct a new ProfileEventData struct
   *
   */
  ProfileEventData(int item, size_t size, std::string str,
                   std::chrono::microseconds dur) :
    time_item(item),
    total_alloc_size(size),
    event_str(str),
    duration(dur) {}

  /* for time profile */
  int time_item;

  /* for memory profile */
  size_t total_alloc_size;

  /* common data */
  std::string event_str;
  std::chrono::microseconds duration;
};

class Profiler;

/**
 * @brief Generic profile listener class to attach to a profiler,
 * this can be inherited to create a custom profile listener
 */
class ProfileListener {
public:
  /**
   * @brief Construct a new Profile Listener object
   *
   */
  explicit ProfileListener() = default;

  /**
   * @brief Destroy the Base Profile Listener object
   *
   */
  virtual ~ProfileListener() noexcept = default;

  /**
   * @brief A callback function to be called from a profiler
   *
   * @param event event type
   * @param data event data
   */
  virtual void notify(PROFILE_EVENT event,
                      const std::shared_ptr<ProfileEventData> data) = 0;

  /**
   * @brief resets the listener to the inital state for a particular key
   *
   * @param time_item time item which will be reset
   */
  virtual void reset(const int time_item, const std::string &str) = 0;

  /**
   * @brief get the latest result of a event
   *
   * @param time_item time item to query the result
   * @return const std::chrono::microseconds
   */
  virtual const std::chrono::microseconds result(const int time_item) = 0;

  /**
   * @brief report the result
   *
   * @param out outstream object to make a report
   */
  virtual void report(std::ostream &out) const = 0;
};

/**
 * @brief Generic Profiler Listener
 *
 */
class GenericProfileListener : public ProfileListener {
public:
  /**
   * @brief Construct a new GenericProfile Listener object
   *
   * @param warmups_ ignore first @a warmups_ records when making time report
   */
  explicit GenericProfileListener(int warmups_ = 0) :
    ProfileListener(),
    start_time(std::chrono::steady_clock::now()),
    warmups(warmups_) {}

  /**
   * @brief Destroy the Generic Profile Listener object
   *
   */
  virtual ~GenericProfileListener() = default;

  /**
   * @brief A callback function to be called from a profiler
   *
   * @param event event type
   * @param data event data
   */
  virtual void notify(PROFILE_EVENT event,
                      const std::shared_ptr<ProfileEventData> data) override;

  /**
   * @copydoc ProfileListener::reset(const int time_item)
   */
  virtual void reset(const int time_item, const std::string &str) override;

  /**
   * @copydoc ProfileListener::result(const int event)
   */
  virtual const std::chrono::microseconds result(const int event) override;

  /**
   * @copydoc ProfileListener::report(std::ostream &out)
   */
  virtual void report(std::ostream &out) const override;

private:
  /**
   * @brief Called when time event occurs
   *
   */
  void onNotifyTimeEvent(PROFILE_EVENT event, const int time_item,
                         const std::string &str,
                         const std::chrono::microseconds &duration);

  /**
   * @brief Called when memory event occurs
   *
   */
  void onNotifyMemoryEvent(PROFILE_EVENT event, const size_t total_alloc_size,
                           const std::string &str,
                           const std::chrono::microseconds &duration);

  std::chrono::time_point<std::chrono::steady_clock> start_time;
  unsigned int warmups;

  static constexpr int CUR = 0;
  static constexpr int MIN = 1;
  static constexpr int MAX = 2;
  static constexpr int SUM = 3;
  static constexpr int CNT = 4;

  std::unordered_map<int, std::tuple<std::chrono::microseconds, /** CUR */
                                     std::chrono::microseconds, /** MIN */
                                     std::chrono::microseconds, /** MAX */
                                     std::chrono::microseconds, /** SUM */
                                     unsigned int /** CNT */>>
    time_taken;

  std::list<
    std::tuple<PROFILE_EVENT, size_t, std::string, std::chrono::microseconds>>
    mem_taken;

  std::unordered_map<int, std::string> names;
};

/**
 * @brief   Overriding output stream for layers and it's derived class
 */
template <typename T,
          typename std::enable_if_t<std::is_base_of<ProfileListener, T>::value,
                                    T> * = nullptr>
std::ostream &operator<<(std::ostream &out, T &l) {
  l.report(out);
  return out;
}

/**
 * @brief Profiler object
 *
 */
class Profiler {
public:
  /**
   * @brief Construct a new Profiler object
   *
   */
  Profiler() : total_size(0) {}

  /**
   * @brief Deleted constructor
   *
   */
  Profiler(const Profiler &) = delete;
  Profiler &operator=(const Profiler &) = delete;

  /**
   *
   * @brief Get Global Profiler
   *
   * @return Profiler&
   */
  static Profiler &Global();

  /**
   * @brief start time profile
   *
   * @param time_item time item to be recorded
   */
  void start(const int time_item);

  /**
   * @brief end time profile and notify to the listeners
   *
   * @param time_item time item to be finished
   */
  void end(const int time_item);

  /**
   * @brief trace memory allocation
   *
   * @param ptr allocated memory pointer
   * @param size amount of allocated memory
   * @param str information string
   */
  void alloc(const void *ptr, size_t size, const std::string &str);

  /**
   * @brief trace memory de-allocation
   *
   * @param ptr de-allocated memory pointer
   */
  void dealloc(const void *ptr);

  /**
   * @brief subscribe a listener to the profiler
   *
   * @param listener listener to register, listener must call unsubscribe on
   * destruction
   * @param events event listeners are subscribing, if empty listener subscribes
   * to all events
   * @throw std::invalid_argument if listener is already registered
   */
  void subscribe(std::shared_ptr<ProfileListener> listener,
                 const std::set<int> &time_item = {});

  /**
   * @brief unsubscribe a listener from the profiler
   *
   * @param listener listener to unsubscribe
   */
  void unsubscribe(std::shared_ptr<ProfileListener> listener);

  /**
   * @brief registerEvent to record.
   * @note Call to the function shouldn't be inside a critical path
   *
   * @return int return NEGATIVE integer to distinguish from reserved events
   */
  int registerTimeItem(const std::string &name);

private:
  /**
   * @brief notify the result
   *
   * @param event event to notify
   * @param value measured value from the profiler
   */
  void notifyListeners(PROFILE_EVENT event,
                       const std::shared_ptr<ProfileEventData> data);

  std::unordered_set<std::shared_ptr<ProfileListener>>
    listeners; /**< event listeners */

  std::unordered_map<int, std::string>
    time_item_names; /**< registered item names (time_item, string) */
  std::unordered_map<int, timepoint>
    time_item_times; /**< registered time items (time_item, time) */
  std::unordered_map<int, std::set<std::shared_ptr<ProfileListener>>>
    time_item_listeners;
  /**< registered listeners for each itemtems (time_item, listeners) */

  std::unordered_map<const void *, std::tuple<size_t, timepoint, std::string>>
    allocates; /**< allocated memory information (ptr, (size, time, info) */

  std::atomic<std::size_t> total_size; /**< total allocated memory size */

  std::mutex listeners_mutex; /**< protect listeners */
  std::mutex allocates_mutex; /**< protect allocates */
  std::mutex registr_mutex;   /**< protect custom event registration */
};

} // namespace profile

} // namespace nntrainer

#endif /** __PROFILER_H__ */
