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

namespace nntrainer {
namespace profile {
typedef enum {
  NN_FORWARD = 0 /**< Neuralnet single inference without loss calculation */,
  TEMP = 999 /**< Temporary event */
} EVENT;
}
} // namespace nntrainer
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
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace nntrainer {
namespace profile {
/**
 * @brief get string representation of event
 *
 * @return std::string name
 */
std::string event_to_str(const EVENT event);

/**
 * @brief Generic profile listener class to attach to a profiler,
 * this can be inherited to create a custom profile listener
 */
class ProfileListener {
public:
  /**
   * @brief Destroy the Base Profile Listener object
   *
   */
  virtual ~ProfileListener() = default;

  /**
   * @brief A callback function to be called from a profiler
   *
   * @param event event key to store the result
   * @param value time value from the profiler
   */
  virtual void onNotify(const EVENT event,
                        const std::chrono::milliseconds &value) = 0;

  /**
   * @brief resets the listener to the inital state for a particular key
   *
   * @param event event which profiler should notice
   */
  virtual void reset(const EVENT event) = 0;

  /**
   * @brief get the latest result of a event
   *
   * @param event event to query the result
   * @return const std::chrono::milliseconds
   */
  virtual const std::chrono::milliseconds result(const EVENT event) = 0;

  /**
   * @brief report the result
   *
   * @param out outstream object to make a report
   */
  virtual void report(std::ostream &out) const = 0;
};

class GenericProfileListener : public ProfileListener {
public:
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
                        const std::chrono::milliseconds &value);

  /**
   * @copydoc ProfileListener::reset(const int event)
   */
  virtual void reset(const int event);

  /**
   * @copydoc ProfileListener::result(const int event)
   */
  virtual const std::chrono::milliseconds result(const int event);

  /**
   * @copydoc ProfileListener::report(std::ostream &out)
   */
  virtual void report(std::ostream &out) const;

private:
  std::unordered_map<int, std::chrono::milliseconds> time_taken;
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
  void start(const EVENT &key);

  /**
   * @brief end profile and notify to the listeners
   *
   * @param key to record the profile result. Either designated key from enum
   * or arbitrary key can be used
   */
  void end(const EVENT &key);

  /**
   * @brief subscribe a listner to the profiler
   *
   * @param listener listener to register, listener must outlive lifetime of
   * profiler
   * @param events event listeners are subscribing, if empty listener subscribes
   * to all events
   */
  void subscribe(ProfileListener *listener,
                 const std::vector<EVENT> &events = {});

private:
  /**
   * @brief notify the result
   *
   * @param event event to notify
   * @param value measured value from the profiler
   */
  void notify(const EVENT &event, const std::chrono::milliseconds &value);

  std::vector<ProfileListener *>
    all_event_listeners; /**< listeners subscribed to all events */

  std::unordered_map<EVENT, std::vector<ProfileListener *>>
    event_listeners; /**< listeners for an events */

  std::unordered_map<EVENT, std::chrono::time_point<std::chrono::steady_clock>>
    start_time; /**< start_time of the clock */
};

} // namespace profile
} // namespace nntrainer

#endif /** __PROFILER_H__ */
