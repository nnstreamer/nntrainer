// SPDX-License-Identifier: Apache-2.0
/** @file htp_activation_reuse.h
 * @brief Single-consumer activation staging lease, guarded by invoke_mutex_.
 */
#ifndef __HTP_ACTIVATION_REUSE_H__
#define __HTP_ACTIVATION_REUSE_H__
#include <cstdint>
#include <thread>

namespace nntrainer {
class HtpActivationReuse {
public:
  uint64_t begin(const float *input, unsigned int rows, unsigned int cols) {
    invalidate();
    if (++serial_ == 0)
      ++serial_;
    token_ = serial_;
    owner_ = std::this_thread::get_id();
    input_ = input;
    rows_ = rows;
    cols_ = cols;
    return token_;
  }
  void end(uint64_t token) {
    if (token && token == token_) {
      invalidate();
      token_ = 0;
    }
  }
  void invalidate() { staged_ = nullptr; }
  void remember(const float *input, unsigned int rows, unsigned int cols,
                void *staged) {
    if (matches(input, rows, cols))
      staged_ = staged;
  }
  void *consume(const float *input, unsigned int rows, unsigned int cols) {
    void *result = matches(input, rows, cols) ? staged_ : nullptr;
    invalidate();
    return result;
  }

private:
  bool matches(const float *input, unsigned int rows, unsigned int cols) const {
    return token_ && owner_ == std::this_thread::get_id() && input == input_ &&
           rows == rows_ && cols == cols_;
  }
  uint64_t serial_ = 0, token_ = 0;
  std::thread::id owner_;
  const float *input_ = nullptr;
  unsigned int rows_ = 0, cols_ = 0;
  void *staged_ = nullptr;
};
} // namespace nntrainer
#endif
