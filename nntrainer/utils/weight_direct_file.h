// SPDX-License-Identifier: Apache-2.0
/** @file weight_direct_file.h
 * @brief Opt-in, fail-closed direct I/O for immutable model weights.
 */
#ifndef __NNTRAINER_WEIGHT_DIRECT_FILE_H__
#define __NNTRAINER_WEIGHT_DIRECT_FILE_H__

#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace nntrainer {
/** This is storage O_DIRECT, unlike QS2_DIRECT_READ (copy avoidance).
 * Only model weights are affected; executable and tokenizer I/O is unchanged.
 */
inline bool weightDirectIO() {
  static const bool enabled = [] {
    const char *value = std::getenv("NNTR_WEIGHT_IO");
    if (!value || std::strcmp(value, "buffered") == 0)
      return false;
    if (std::strcmp(value, "direct") != 0)
      throw std::invalid_argument("NNTR_WEIGHT_IO must be buffered or direct");
#if defined(__linux__) || defined(__ANDROID__)
    return true;
#else
    throw std::runtime_error("NNTR_WEIGHT_IO=direct requires Linux/Android");
#endif
  }();
  return enabled;
}
} // namespace nntrainer

#if defined(__linux__) || defined(__ANDROID__)
#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <fcntl.h>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <streambuf>
#include <string>
#include <sys/stat.h>
#include <system_error>
#include <tuple>
#include <unistd.h>

namespace nntrainer {
/** A separate open description, so the caller's offset and flags never change.
 * Files must remain immutable. Unsupported O_DIRECT or alignment fails without
 * falling back to buffered I/O. 4 KiB alignment is verified on the target FS;
 * larger alignment requirements fail at pread rather than being hidden.
 */
class WeightDirectFile {
  int fd_ = -1;
  uint64_t size_ = 0;
  static constexpr size_t alignment = 4096, capacity = 1024 * 1024;

public:
  explicit WeightDirectFile(int source) {
    struct stat before {
    }, after{};
    if (fstat(source, &before))
      throw std::system_error(errno, std::generic_category(), "weight fstat");
    if (!S_ISREG(before.st_mode) || before.st_size < 0)
      throw std::invalid_argument("Direct weights require a regular file");
    const auto path = "/proc/self/fd/" + std::to_string(source);
    fd_ = open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
    if (fd_ < 0)
      throw std::system_error(errno, std::generic_category(),
                              "weight O_DIRECT");
    const int rc = fstat(fd_, &after);
    const int error = errno;
    if (rc || before.st_dev != after.st_dev || before.st_ino != after.st_ino ||
        before.st_size != after.st_size) {
      close(fd_);
      fd_ = -1;
      throw std::system_error(rc ? error : ESTALE, std::generic_category(),
                              "weight identity changed");
    }
    size_ = static_cast<uint64_t>(after.st_size);
  }
  ~WeightDirectFile() { close(fd_); }
  WeightDirectFile(const WeightDirectFile &) = delete;
  WeightDirectFile &operator=(const WeightDirectFile &) = delete;
  uint64_t size() const { return size_; }

  /** 0 on success, errno on error, -1 on EOF. Arbitrary offsets/destinations
   * use a bounded aligned bounce buffer, including unaligned file tails.
   * Each thread owns scratch only (no retained weight cache).
   */
  int read(void *dst, size_t bytes, uint64_t offset) const {
    if (offset > size_ || bytes > size_ - offset)
      return -1;
    if (!bytes)
      return 0;
    if (!dst)
      return EINVAL;
    struct Scratch {
      void *data = nullptr;
      ~Scratch() { std::free(data); }
    };
    thread_local Scratch scratch;
    if (!scratch.data) {
      const int rc = posix_memalign(&scratch.data, alignment, capacity);
      if (rc)
        return rc;
    }
    auto *out = static_cast<uint8_t *>(dst);
    while (bytes) {
      const size_t skip = offset % alignment;
      const size_t take = std::min(bytes, capacity - skip);
      const size_t request =
        (skip + take + alignment - 1) / alignment * alignment;
      ssize_t got;
      do {
        got =
          pread(fd_, scratch.data, request, static_cast<off_t>(offset - skip));
      } while (got < 0 && errno == EINTR);
      if (got < 0)
        return errno;
      if (static_cast<size_t>(got) < skip + take)
        return -1;
      std::memcpy(out, static_cast<const uint8_t *>(scratch.data) + skip, take);
      out += take;
      offset += take;
      bytes -= take;
    }
    return 0;
  }
};

/** Thread-safe reader cache keyed by file identity, never by reusable fd alone.
 * Descriptors are retained; file data is not. Join readers before destruction.
 */
class WeightDirectFiles {
  std::mutex mutex_;
  std::map<std::tuple<dev_t, ino_t, off_t>, std::unique_ptr<WeightDirectFile>>
    files_;

public:
  int read(int fd, void *dst, size_t bytes, uint64_t offset) {
    struct stat st {};
    if (fstat(fd, &st))
      return errno;
    const WeightDirectFile *file;
    try {
      std::lock_guard<std::mutex> lock(mutex_);
      const auto key = std::make_tuple(st.st_dev, st.st_ino, st.st_size);
      auto it = files_.find(key);
      if (it == files_.end())
        it = files_.emplace(key, std::make_unique<WeightDirectFile>(fd)).first;
      file = it->second.get();
    } catch (const std::system_error &error) {
      return error.code().value();
    } catch (const std::invalid_argument &) {
      return EINVAL;
    }
    return file->read(dst, bytes, offset);
  }
};

/** Adapter for existing ifstream tensor readers without changing ReadSource
 * ABI. Supports binary block reads and seeks; no buffered model file is opened.
 */
class WeightDirectStream {
  class Buffer : public std::streambuf {
    WeightDirectFile file_;
    uint64_t position_ = 0;

  protected:
    std::streamsize xsgetn(char *dst, std::streamsize count) override {
      if (count < 0)
        return 0;
      const int rc = file_.read(dst, static_cast<size_t>(count), position_);
      if (rc)
        throw std::system_error(rc < 0 ? EIO : rc, std::generic_category(),
                                "direct weight read");
      position_ += static_cast<uint64_t>(count);
      return count;
    }
    pos_type seekoff(off_type off, std::ios_base::seekdir dir,
                     std::ios_base::openmode mode) override {
      if (!(mode & std::ios_base::in))
        return pos_type(off_type(-1));
      const uint64_t base = dir == std::ios_base::beg   ? 0
                            : dir == std::ios_base::cur ? position_
                                                        : file_.size();
      if (base > static_cast<uint64_t>(std::numeric_limits<off_type>::max()) ||
          (off >= 0 && static_cast<uint64_t>(off) > file_.size() - base) ||
          (off < 0 && static_cast<uint64_t>(-(off + 1)) + 1 > base))
        return pos_type(off_type(-1));
      position_ = static_cast<off_type>(base) + off;
      return pos_type(position_);
    }
    pos_type seekpos(pos_type pos, std::ios_base::openmode mode) override {
      return seekoff(off_type(pos), std::ios_base::beg, mode);
    }

  public:
    explicit Buffer(int fd) : file_(fd) {}
  } buffer_;
  std::ifstream stream_;

public:
  explicit WeightDirectStream(int fd) : buffer_(fd) {
    static_cast<std::ios &>(stream_).rdbuf(&buffer_);
  }
  ~WeightDirectStream() { static_cast<std::ios &>(stream_).rdbuf(nullptr); }
  std::ifstream &stream() { return stream_; }
};
} // namespace nntrainer
#endif
#endif
