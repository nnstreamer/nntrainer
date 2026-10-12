// SPDX-License-Identifier: Apache-2.0
/** @file htp_qs2_mapped_file.h
 * @brief Optional read-only mappings of immutable QS2 model files.
 */
#ifndef __NNTRAINER_HTP_QS2_MAPPED_FILE_H__
#define __NNTRAINER_HTP_QS2_MAPPED_FILE_H__

#if defined(__linux__) || defined(__ANDROID__)
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <sys/mman.h>
#include <sys/stat.h>
#include <tuple>

namespace nntrainer {
/** The model file must remain immutable. Callers join all readers before
 * destroying this cache. Mappings are keyed by file identity, not reusable
 * descriptor numbers, and add no private copy of the model's page cache.
 */
class HtpQs2MappedFiles {
  struct Mapping {
    void *data = MAP_FAILED;
    size_t size = 0;
    ~Mapping() {
      if (data != MAP_FAILED)
        munmap(data, size);
    }
  };
  std::mutex mutex_;
  std::map<std::tuple<dev_t, ino_t, off_t>, std::unique_ptr<Mapping>> maps_;

public:
  /** Same return convention as preadAll: zero, errno, or -1 at EOF. */
  int read(int fd, void *dst, size_t bytes, uint64_t offset) {
    struct stat st;
    if (fstat(fd, &st) != 0)
      return errno;
    if (!S_ISREG(st.st_mode) || st.st_size < 0 ||
        static_cast<uint64_t>(st.st_size) > std::numeric_limits<size_t>::max())
      return EINVAL;
    const size_t size = static_cast<size_t>(st.st_size);
    if (offset > size || bytes > size - offset)
      return -1;
    if (!bytes)
      return 0;
    if (!dst)
      return EINVAL;
    const Mapping *mapping;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      const auto key = std::make_tuple(st.st_dev, st.st_ino, st.st_size);
      auto found = maps_.find(key);
      if (found == maps_.end()) {
        auto created = std::make_unique<Mapping>();
        created->size = size;
        created->data = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
        if (created->data == MAP_FAILED)
          return errno;
        found = maps_.emplace(key, std::move(created)).first;
      }
      mapping = found->second.get();
    }
    std::memcpy(dst, static_cast<const uint8_t *>(mapping->data) + offset,
                bytes);
    return 0;
  }
};
} // namespace nntrainer
#endif
#endif
