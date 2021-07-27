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
#include <algorithm>
#include <iomanip>
#include <map>
#include <numeric>
#include <sstream>
#include <tuple>

#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <profiler.h>

namespace nntrainer {
namespace profile {

ProfileListener::ProfileListener(Profiler *profiler_, std::vector<int> events) :
  profiler(profiler_) {
  if (profiler != nullptr) {
    profiler->subscribe(this, events);
  }
}

ProfileListener::~ProfileListener() noexcept {
  if (profiler != nullptr) {
    try {
      profiler->unsubscribe(this);
    } catch (...) {
      ml_logw("unsubscribing profiler failed may cause undefined error");
    }
  }
}

void GenericProfileListener::onNotify(const int event,
                                      const std::chrono::microseconds &value) {

  time_iter = time_taken.find(event);

  if (time_iter == time_taken.end()) {
    reset(event); // this sets time_iter to current, note that this is a side
                  // effect of reset()
  }
  auto &cnt_ = std::get<GenericProfileListener::CNT>(time_iter->second);
  cnt_++;

  if (warmups >= cnt_) {
    return;
  }

  auto &cur_ = std::get<GenericProfileListener::CUR>(time_iter->second);
  auto &min_ = std::get<GenericProfileListener::MIN>(time_iter->second);
  auto &max_ = std::get<GenericProfileListener::MAX>(time_iter->second);
  auto &sum_ = std::get<GenericProfileListener::SUM>(time_iter->second);

  cur_ = value;
  min_ = std::min(min_, value);
  max_ = std::max(max_, value);
  sum_ += value;
}

void GenericProfileListener::reset(const int event) {
  time_iter =
    time_taken
      .insert({event, std::make_tuple(std::chrono::microseconds{0},
                                      std::chrono::microseconds::max(),
                                      std::chrono::microseconds::min(),
                                      std::chrono::microseconds{0}, int{0})})
      .first;
}

const std::chrono::microseconds
GenericProfileListener::result(const int event) {
  auto iter = time_taken.find(event);

  if (iter == time_taken.end() ||
      std::get<GenericProfileListener::CNT>(iter->second) == 0) {
    std::stringstream ss;
    ss << "event has never recorded" << profiler->eventToStr(event);
    throw std::invalid_argument("event has never recorded");
  }

  return std::get<GenericProfileListener::CUR>(iter->second);
}

void GenericProfileListener::report(std::ostream &out) const {
  std::vector<unsigned int> column_size = {10, 23, 23, 23, 23};

  for (auto &entry : time_taken) {
    auto title = profiler->eventToStr(entry.first);
    column_size[0] =
      std::max(column_size[0], static_cast<unsigned int>(title.size()));
  }
  auto total_col_size =
    std::accumulate(column_size.begin(), column_size.end(), 0);

  if (warmups != 0) {
    out << "warm up: " << warmups << '\n';
  }

  /// creating header
  // clang-format off
  out << std::setw(column_size[0]) << "key"
      << std::setw(column_size[1]) << "avg"
      << std::setw(column_size[2]) << "min"
      << std::setw(column_size[3]) << "max"
      << std::setw(column_size[4]) << "sum" << '\n';
  // clang-format on

  // seperator
  out << std::string(total_col_size, '=') << '\n';

  std::map<int, std::function<void(std::ostream & out)>> ordered_report;

  /// calculate metrics while skipping warmups
  for (auto &entry : time_taken) {
    auto func = [&](std::ostream &out_) {
      auto &cnt_ = std::get<GenericProfileListener::CNT>(entry.second);
      auto &min_ = std::get<GenericProfileListener::MIN>(entry.second);
      auto &max_ = std::get<GenericProfileListener::MAX>(entry.second);
      auto &sum_ = std::get<GenericProfileListener::SUM>(entry.second);

      auto title = profiler->eventToStr(entry.first);

      if (warmups >= cnt_) {
        out_ << std::left << std::setw(total_col_size) << title
             << "less data then warmup\n";
        out_
          << std::right; // Restore outputstream adjustflag to standard stream
        return;
      }

      // clang-format off
    out_ << std::setw(column_size[0]) << title
        << std::setw(column_size[1]) << sum_.count() / (cnt_ - warmups)
        << std::setw(column_size[2]) << min_.count()
        << std::setw(column_size[3]) << max_.count()
        << std::setw(column_size[4]) << sum_.count() << '\n';
      // clang-format on
    };
    ordered_report[-entry.first] = func;
  }

  for (auto &entry : ordered_report) {
    entry.second(out);
  }
}

Profiler &Profiler::Global() {
  static Profiler instance;
  return instance;
}

void Profiler::start(const int &event) {
#ifdef DEBUG
  /// @todo: consider race condition
  auto iter = start_time.find(event);
  if (iter != start_time.end()) {
    throw std::invalid_argument("profiler has already started");
  }
#endif

  start_time[event] = std::chrono::steady_clock::now();
}

void Profiler::end(const int &event) {
  /// @todo: consider race condition
  auto end = std::chrono::steady_clock::now();
  auto iter = start_time.find(event);

#ifdef DEBUG
  if (iter == start_time.end()) {
    throw std::invalid_argument("profiler hasn't started with the event");
  }
#endif

  auto duration =
    std::chrono::duration_cast<std::chrono::microseconds>(end - iter->second);
  notify(event, duration);

#ifdef DEBUG
  start_time.erase(iter);
#endif
}

void Profiler::notify(const int &event,
                      const std::chrono::microseconds &value) {
  std::lock_guard<std::mutex> lk(subscription_mutex);
  for (auto &listener : all_event_listeners) {
    listener->onNotify(event, value);
  }

  auto &items = event_listeners[event];

  for (auto &listner : items) {
    listner->onNotify(event, value);
  }
}

void Profiler::subscribe(ProfileListener *listener,
                         const std::vector<int> &events) {
  if (listener == nullptr) {
    throw std::invalid_argument("listener is null!");
  }

  {
    std::lock_guard<std::mutex> lk(subscription_mutex);
    if (all_registered_listeners.count(listener) == 1) {
      throw std::invalid_argument(
        "listener is already registered, please unsubscribe before subscribe ");
    }

    all_registered_listeners.insert(listener);

    if (events.empty()) {
      all_event_listeners.insert(listener);
      return;
    }

    for (auto event : events) {
      auto iter = event_listeners.find(event);
      if (iter == event_listeners.end()) {
        event_listeners[event] = std::unordered_set<ProfileListener *>{};
      }
      event_listeners[event].insert(listener);
    }
  }
}

void Profiler::unsubscribe(ProfileListener *listener) {
  std::lock_guard<std::mutex> lk(subscription_mutex);
  all_registered_listeners.erase(listener);
  all_event_listeners.erase(listener);

  for (auto &it : event_listeners) {
    it.second.erase(listener);
  }
}

std::string Profiler::eventToStr(const int event) {

  /// reserved event
  if (event >= 0) {
    switch (event) {
    case EVENT::NN_FORWARD:
      return "nn_forward";
    }
  }
  /// custom events
  else if (event < 0) {
    std::lock_guard<std::mutex> lk(event_registration_mutex);
    int actual_idx = -event - 1;

    if (actual_idx > static_cast<int>(custom_events.size()) - 1) {
      std::stringstream ss;
      ss << "undef(" << event << ')';
      return ss.str();
    }
    return custom_events[actual_idx];
  }

  throw std::logic_error("control should not reach here");
}

int Profiler::registerEvent(const std::string &name) {
  std::lock_guard<std::mutex> lk(event_registration_mutex);
  custom_events.push_back(name);
  int key = -custom_events.size();

  ml_logd("[Profiler] Event registered, event name: %s event key: %d",
          name.c_str(), key);

  return key;
}

} // namespace profile

} // namespace nntrainer
