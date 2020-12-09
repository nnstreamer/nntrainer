// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   profiler.cpp
 * @date   09 December 2020
 * @brief  Profiler related codes to be used to benchmark things
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */
#include <sstream>

#include <profiler.h>

namespace nntrainer {
namespace profile {

Profiler &Profiler::Global() {
  static Profiler instance;
  return instance;
}

std::string event_to_str(const EVENT event) {
  switch (event) {
  case EVENT::NN_FORWARD:
    return "nn_forward";
  case EVENT::TEMP:
    return "temp";
  }

  std::stringstream ss;
  ss << "undefined key - " << event;
  return ss.str();
}

void Profiler::start(const EVENT &event) {
  /// @todo: consider race condition
  auto iter = start_time.find(event);

  if (iter != start_time.end()) {
    throw std::invalid_argument("profiler has already started");
  }

  start_time[event] = std::chrono::steady_clock::now();
}

void Profiler::end(const EVENT &event) {
  /// @todo: consider race condition
  auto end = std::chrono::steady_clock::now();
  auto iter = start_time.find(event);

  if (iter == start_time.end()) {
    throw std::invalid_argument("profiler hasn't started with the event");
  }

  auto duration =
    std::chrono::duration_cast<std::chrono::milliseconds>(end - iter->second);
  notify(event, duration);

  start_time.erase(iter);
}

void Profiler::notify(const EVENT &event,
                      const std::chrono::milliseconds &value) {
  for (auto listener : all_event_listeners) {
    listener->onNotify(event, value);
  }
  auto items = event_listeners[event];

  for (auto listner : items) {
    listner->onNotify(event, value);
  }
}

void Profiler::subscribe(ProfileListener *listener,
                         const std::vector<EVENT> &events) {

  if (listener == nullptr) {
    throw std::invalid_argument("listener is null!");
  }

  if (events.empty()) {
    all_event_listeners.push_back(listener);
    return;
  }

  for (auto event : events) {
    auto iter = event_listeners.find(event);
    if (iter == event_listeners.end()) {
      event_listeners[event] = std::vector<ProfileListener *>{};
    }
    event_listeners[event].push_back(listener);
  }
}

} // namespace profile

} // namespace nntrainer
