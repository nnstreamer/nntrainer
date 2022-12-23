// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2022 Jiho Chu <jiho.chu@samsung.com>
 *
 * @file   tracer.h
 * @date   23 December 2022
 * @brief  Trace abstract class
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jiho Chu <jiho.chu@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __TRACER_H__
#define __TRACER_H__

#include <exception>
#include <fstream>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <type_traits>

#ifdef TRACE

namespace nntrainer {

/**
 * @class   Tracer
 * @brief   Tracer class
 */
class Tracer {
public:
  /**
   * @brief     Constructor of Tracer
   */
  Tracer(const std::string &name) : name_(name) {}

  /**
   * @brief     Destructor of Tracer
   */
  virtual ~Tracer() = default;

  /**
   * @brief     Start tracing
   */
  virtual Tracer &traceStart(const std::string &tag,
                             const std::string &msg) = 0;

  /**
   * @brief     End tracing
   */
  virtual Tracer &traceEnd(const std::string &tag) = 0;

  /**
   * @brief     Trace point
   */
  virtual Tracer &tracePoint(const std::string &msg) = 0;

protected:
  /**
   * @brief     Write trace message (final tuple)
   */
  template <std::size_t I = 0, typename... Tp>
  inline typename std::enable_if<I == sizeof...(Tp), void>::type
  write(std::ofstream &out, std::tuple<Tp...> &t) {}

  /**
   * @brief     Write trace message
   * @param[in] out output stream
   * @param[in] t tuple to write
   */
  template <std::size_t I = 0, typename... Tp>
    inline typename std::enable_if <
    I<sizeof...(Tp), void>::type write(std::ofstream &out,
                                       std::tuple<Tp...> &t) {
    out << std::get<I>(t) << "\t";
    write<I + 1, Tp...>(out, t);
  }

  /**
   * @brief     Write trace information to file
   * @param[in] filename file name to write
   * @param[in] trace_info trace information to write
   */
  template <typename T>
  void writeToFile(std::string filename, std::list<T> &trace_info) {
    std::ofstream file(filename, std::fstream::app);
    if (!file.is_open()) {
      throw std::runtime_error("Cannot open file: " + filename);
    }

    for (auto &i : trace_info) {
      write(file, i);
      file << std::endl;
    }

    trace_info.clear();
  }

  const std::string name_; /**< name of the tracer */
};

/**
 * @class   MemoryTracer
 * @brief   Memory Tracer class
 */
class MemoryTracer : public Tracer {
public:
  /**
   * @brief     Destructor of Tracer
   */
  virtual ~MemoryTracer();

  /**
   * @brief     Get instance of MemoryTracer
   */
  static std::unique_ptr<MemoryTracer> &getInstance();

  /**
   * @brief     Start tracing
   */
  Tracer &traceStart(const std::string &tag, const std::string &msg) override {
    return (*this);
  }

  /**
   * @brief     End tracing
   */
  Tracer &traceEnd(const std::string &tag) override { return (*this); }

  /**
   * @brief     Trace point
   */
  Tracer &tracePoint(const std::string &msg) override;

  /**
   * @brief     Operator overload for << to trace point
   */
  Tracer &operator<<(const std::string &msg) { return tracePoint(msg); }

private:
  /**
   * @brief     Constructor of MemoryTracer
   */
  MemoryTracer(const std::string &name, bool flush = false);

  std::list<std::tuple<unsigned long, std::string>>
    trace_info_; /**< memory usage, msg */
  bool flush_;   /**< flush to file */
};

/**
 * @class   TiemTracer
 * @brief   Time Tracer class
 */
class TimeTracer : public Tracer {
public:
  /**
   * @brief     Destructor of Tracer
   */
  virtual ~TimeTracer();

  /**
   * @brief     Get instance of TimeTracer
   */
  static std::unique_ptr<TimeTracer> &getInstance();

  /**
   * @brief     Start tracing
   */
  Tracer &traceStart(const std::string &tag, const std::string &msg) override {
    return (*this);
  }

  /**
   * @brief     End tracing
   */
  Tracer &traceEnd(const std::string &tag) override { return (*this); }

  /**
   * @brief     Trace point
   */
  Tracer &tracePoint(const std::string &msg) override;

  /**
   * @brief     Operator overload for << to trace point
   */
  Tracer &operator<<(const std::string &msg) { return tracePoint(msg); }

private:
  /**
   * @brief     Constructor of TimeTracer
   */
  TimeTracer(const std::string &name, bool flush = false);

  std::list<std::tuple<unsigned long, std::string>>
    trace_info_; /**< time point (ms), msg */
  bool flush_;   /**< flush to file */
};

} // namespace nntrainer

#define TRACE_MEMORY_POINT(msg) \
  nntrainer::MemoryTracer::getInstance()->tracePoint(msg)
#define TRACE_MEMORY() *(nntrainer::MemoryTracer::getInstance())
#define TRACE_TIME_POINT(msg) \
  nntrainer::TimeTracer::getInstance()->tracePoint(msg)
#define TRACE_TIME() *(nntrainer::TimeTracer::getInstance())

#else

#define TRACE_MEMORY_POINT(msg)
#define TRACE_MEMORY() std::ostream(nullptr)
#define TRACE_TIME_POINT(msg)
#define TRACE_TIME() std::ostream(nullptr)

#endif

#endif /* __TRACER_H__ */
