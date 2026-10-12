// SPDX-License-Identifier: Apache-2.0
/** @file moe_lookahead_trace.h
 * @brief Opt-in FP32 snapshots for offline MoE lookahead evaluation.
 */
#ifndef NNTR_MOE_LOOKAHEAD_TRACE_H
#define NNTR_MOE_LOOKAHEAD_TRACE_H

#include <cstdlib>
#include <fstream>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>

namespace causallm {

/** The caller creates this directory. Diagnostic I/O invalidates timings. */
inline const char *moeLookaheadDirectory() {
  static const char *path = std::getenv("NNTR_MOE_LOOKAHEAD_DUMP");
  return path && *path ? path : nullptr;
}

/** Write one tensor, optionally only once per process. */
inline void moeLookaheadDump(const std::string &name, const float *data,
                             size_t count, bool once = false) {
  const char *dir = moeLookaheadDirectory();
  if (!dir)
    return;
  static std::mutex mutex;
  static std::set<std::string> written;
  std::lock_guard<std::mutex> lock(mutex);
  const std::string path = std::string(dir) + "/" + name + ".f32";
  if (once && written.count(path))
    return;
  std::ofstream out(path, std::ios::binary);
  if (!out.write(reinterpret_cast<const char *>(data), count * sizeof(float)))
    throw std::runtime_error("Cannot write MoE lookahead snapshot: " + path);
  if (once)
    written.insert(path);
}

} // namespace causallm
#endif
