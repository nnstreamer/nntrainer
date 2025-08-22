// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Michal Wlasiuk <testmailsmtp12345@gmail.com>
 *
 * @file	timer.h
 * @date	22 Aug 2025
 * @brief	Timer class wrapping std::chrono...
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Michal Wlasiuk <testmailsmtp12345@gmail.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __TIMER_H__
#define __TIMER_H__

#include <chrono>

namespace TimerConstants {
static constexpr float kNanoSecondsToNanoSeconds = 1.0f / 1.0f;
static constexpr float kNanoSecondsToMicroSeconds = 1.0f / 1'000.0f;
static constexpr float kNanoSecondsToMilliSeconds = 1.0f / 1'000'000.0f;
static constexpr float kNanoSecondsToSeconds = 1.0f / 1'000'000'000.0f;
} // namespace TimerConstants

class Timer {
private:
  std::chrono::time_point<std::chrono::high_resolution_clock>
    _begin_timestamp{};

public:
  Timer() : _begin_timestamp(std::chrono::high_resolution_clock::now()) {}

  ~Timer() {}

  inline float GetElapsedNanoseconds() const {
    return static_cast<float>(
             std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::high_resolution_clock::now() - _begin_timestamp)
               .count()) *
           TimerConstants::kNanoSecondsToNanoSeconds;
  }

  inline float GetElapsedMicroseconds() const {
    return GetElapsedNanoseconds() * TimerConstants::kNanoSecondsToMicroSeconds;
  }

  inline float GetElapsedMilliseconds() const {
    return GetElapsedNanoseconds() * TimerConstants::kNanoSecondsToMilliSeconds;
  }

  inline float GetElapsedSeconds() const {
    return GetElapsedNanoseconds() * TimerConstants::kNanoSecondsToSeconds;
  }

  inline void Reset() {
    _begin_timestamp = std::chrono::high_resolution_clock::now();
  }
};

#endif
