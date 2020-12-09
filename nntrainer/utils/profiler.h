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
#ifndef PROFILE

#define START_PROFILE(event_key)
#define END_PROFILE(event_key)

#else

#define START_PROFILE(event_key)                         \
  do {                                                   \
    auto &prof = nntrainer::profile::Profiler::Global(); \
    prof.start(event_key);                               \
  } while (0);

#define END_PROFILE(event_key)                           \
  do {                                                   \
    auto &prof = nntrainer::profile::Profiler::Global(); \
    prof.end(event_key);                                 \
  } while (0);

#endif /** PROFILE */

#include <chrono>
#include <future>
#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
namespace nntrainer {
namespace profile {

typedef enum {
  NN_FORWARD = 0 /**< Neuralnet single inference without loss calculation */,
  TEMP = 999 /**< Temporary event */
} EVENT;

/**
 * @brief get string representation of event
 *
 * @return std::string name
 */
std::string event_to_str(const int event);

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
   * @param profiler_ profiler that this listener is bound to. Unsubscribe will
   * be called when destruction
   * @param events events for this profiler to listen to
   */
  ProfileListener(Profiler *profiler_, std::vector<int> events);

  /**
   * @brief Destroy the Base Profile Listener object
   *
   */
  virtual ~ProfileListener() noexcept;

  /**
   * @brief A callback function to be called from a profiler
   *
   * @param event event key to store the result
   * @param value time value from the profiler
   */
  virtual void onNotify(const int event,
                        const std::chrono::milliseconds &value) = 0;

  /**
   * @brief resets the listener to the inital state for a particular key
   *
   * @param event event which profiler should notice
   */
  virtual void reset(const int event) = 0;

  /**
   * @brief get the latest result of a event
   *
   * @param event event to query the result
   * @return const std::chrono::milliseconds
   */
  virtual const std::chrono::milliseconds result(const int event) = 0;

  /**
   * @brief report the result
   *
   * @param out outstream object to make a report
   */
  virtual void report(std::ostream &out) const = 0;

private:
  Profiler *profiler;
};

class GenericProfileListener : public ProfileListener {
public:
  /**
   * @brief Construct a new GenericProfile Listener object
   *
   * @param profiler profiler that this listener is bound to. pass null if empty
   * @param warmups_ ignore first @a warmups_ records when making report
   */
  GenericProfileListener(Profiler *profiler, std::vector<int> events = {},
                         int warmups_ = 0) :
    ProfileListener(profiler, events),
    warmups(warmups_) {
    for (auto &event : events) {
      reset(event);
    }
  }

  /**
   * @brief Destroy the Generic Profile Listener object
   *
   */
  virtual ~GenericProfileListener() = default;

  /**
   * @copydoc ProfileListener::onNotify(const int event, const
   * std::chrono::milliseconds &value)
   */
  virtual void onNotify(const int event,
                        const std::chrono::milliseconds &value) override;

  /**
   * @copydoc ProfileListener::reset(const int event)
   */
  virtual void reset(const int event) override;

  /**
   * @copydoc ProfileListener::result(const int event)
   */
  virtual const std::chrono::milliseconds result(const int event) override;

  /**
   * @copydoc ProfileListener::report(std::ostream &out)
   */
  virtual void report(std::ostream &out) const override;

private:
  unsigned int warmups;

  static constexpr int CUR = 0;
  static constexpr int MIN = 1;
  static constexpr int MAX = 2;
  static constexpr int SUM = 3;
  static constexpr int CNT = 4;

  std::unordered_map<int, std::tuple<std::chrono::milliseconds, /** CUR */
                                     std::chrono::milliseconds, /** MIN */
                                     std::chrono::milliseconds, /** MAX */
                                     std::chrono::milliseconds, /** SUM */
                                     unsigned int /** CNT */>>
    time_taken;

  decltype(time_taken)::iterator time_iter; /**< iterator for the time_taken */
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

class Profiler {
public:
  Profiler() {}

  Profiler(const Profiler &) = delete;

  Profiler &operator=(const Profiler &) = delete;

  /**
   *
   * @brief Get Global app context.
   *
   * @return AppContext&
   */
  static Profiler &Global();

  /**
   * @brief start profile
   *
   * @param key to record the profile result. Either designated key from enum
   * or arbitrary key can be used
   */
  void start(const int &key);

  /**
   * @brief end profile and notify to the listeners
   *
   * @param key to record the profile result. Either designated key from enum
   * or arbitrary key can be used
   */
  void end(const int &key);

  /**
   * @brief subscribe a listener to the profiler
   *
   * @param listener listener to register, listener must call unsubscribe on
   * destruction
   * @param events event listeners are subscribing, if empty listener subscribes
   * to all events
   * @throw std::invalid_argument if listener is already registered
   */
  void subscribe(ProfileListener *listener,
                 const std::vector<int> &events = {});

  /**
   * @brief unsubscribe a listener from the profiler
   *
   * @param listener listener to unsubscribe
   */
  void unsubscribe(ProfileListener *listener);

private:
  /**
   * @brief notify the result
   *
   * @param event event to notify
   * @param value measured value from the profiler
   */
  void notify(const int &event, const std::chrono::milliseconds &value);

  std::unordered_set<ProfileListener *>
    all_registered_listeners; /**< prevent registering listener twice */

  std::unordered_set<ProfileListener *>
    all_event_listeners; /**< listeners listen to every events */

  std::unordered_map<int, std::unordered_set<ProfileListener *>>
    event_listeners; /**< listeners for an events */

  std::unordered_map<int, std::chrono::time_point<std::chrono::steady_clock>>
    start_time; /**< start_time of the clock */

  std::mutex subscription_mutex; /**< protect sub/unsub routine to
                                             gaurantee invarient */
};

} // namespace profile
} // namespace nntrainer

#endif /** __PROFILER_H__ */
