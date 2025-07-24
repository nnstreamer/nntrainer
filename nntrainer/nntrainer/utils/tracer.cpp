// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2022 Jiho Chu <jjiho.chu@samsung.com>
 *
 * @file   trace.cpp
 * @date   23 December 2022
 * @brief  trace class
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jiho Chu <jiho.chu@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include "tracer.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <functional>
#include <iterator>
#include <list>
#include <sys/types.h>
#include <tuple>
#include <unistd.h>

namespace {

const std::string memory_trace_tag = "memory_trace";
const std::string time_trace_tag = "time_trace";

auto outputFileName = [](std::string name) -> std::string {
  return name + "_" + std::to_string(static_cast<int>(getpid())) + ".log";
};

unsigned long getMemoryUsage(void) {
  std::ifstream ifs("/proc/self/smaps", std::ios::in);
  std::istream_iterator<std::string> it(ifs);
  unsigned long total = 0;

  while (it != std::istream_iterator<std::string>()) {
    if (*it == "Rss:") {
      ++it;
      total += std::stoul(*it);
    }
    ++it;
  }

  return total;
}

unsigned long getTimeStamp(void) {
  static auto start = std::chrono::system_clock::now().time_since_epoch();
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::system_clock::now().time_since_epoch() - start)
              .count();

  return ms;
}

} // namespace

namespace nntrainer {

std::unique_ptr<MemoryTracer> &MemoryTracer::getInstance() {
  static std::unique_ptr<MemoryTracer> instance;
  static std::once_flag flag;

  std::call_once(flag,
                 []() { instance.reset(new MemoryTracer(memory_trace_tag)); });
  return instance;
}

MemoryTracer::MemoryTracer(const std::string &name, bool flush) :
  Tracer(name),
  flush_(flush) {
  std::ofstream ifs(outputFileName(name), std::fstream::trunc);
  trace_info_.emplace_back(getMemoryUsage(), "start");
  if (flush_)
    writeToFile(outputFileName(name), trace_info_);
}

MemoryTracer::~MemoryTracer() {
  writeToFile(outputFileName(name_), trace_info_);
}

Tracer &MemoryTracer::tracePoint(const std::string &msg) {
  trace_info_.emplace_back(getMemoryUsage(), msg);
  if (flush_)
    writeToFile(outputFileName(name_), trace_info_);

  return (*this);
}

std::unique_ptr<TimeTracer> &TimeTracer::getInstance() {
  static std::unique_ptr<TimeTracer> instance;
  static std::once_flag flag;

  std::call_once(flag,
                 []() { instance.reset(new TimeTracer(time_trace_tag)); });
  return instance;
}

TimeTracer::TimeTracer(const std::string &name, bool flush_) :
  Tracer(name),
  flush_(flush_) {
  std::ofstream ifs(outputFileName(name), std::fstream::trunc);
  trace_info_.emplace_back(getTimeStamp(), "start");
  if (flush_)
    writeToFile(outputFileName(name), trace_info_);
}

TimeTracer::~TimeTracer() { writeToFile(outputFileName(name_), trace_info_); }

Tracer &TimeTracer::tracePoint(const std::string &msg) {
  trace_info_.emplace_back(getTimeStamp(), msg);
  if (flush_)
    writeToFile(outputFileName(name_), trace_info_);

  return (*this);
}

} // namespace nntrainer
