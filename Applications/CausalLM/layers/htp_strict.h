// SPDX-License-Identifier: Apache-2.0
/** @file htp_strict.h
 * @brief Explicit HTP LFM2 inference with an optional CPU LM head.
 */
#ifndef __CAUSALLM_HTP_STRICT_H__
#define __CAUSALLM_HTP_STRICT_H__

#include <compute_ops.h>
#include <cstdlib>
#include <stdexcept>

namespace causallm {
inline bool htpStrict() {
  const char *value = std::getenv("NNTR_HTP_STRICT");
  return value && std::atoi(value) != 0;
}

// Keep the unfused strict path available for numerical and latency comparisons.
inline bool htpStrictFusion() {
  const char *value = std::getenv("NNTR_HTP_STRICT_FUSION");
  return htpStrict() && (!value || std::atoi(value) != 0);
}

// Explicit CPU exception for the tied GEMV; all other strict math stays HTP.
inline bool htpCpuLmHead() {
  const char *value = std::getenv("NNTR_HTP_LMHEAD_CPU");
  return value && std::atoi(value) != 0;
}

inline bool htpAddNormFusion() {
  const char *value = std::getenv("NNTR_HTP_ADD_NORM_FUSION");
  return htpStrict() && (!value || std::atoi(value) != 0);
}

inline nntrainer::ComputeOps *htpStrictOps() {
#ifdef ENABLE_HEXKL
  return nntrainer::get_htp_ops();
#else
  throw std::runtime_error("NNTR_HTP_STRICT requires an HTP build");
#endif
}

inline void htpStrictMath(nntrainer::ComputeOps::InferenceMath kind,
                          unsigned rows, unsigned width, unsigned chunk,
                          float eps, float scale, const float *x,
                          const float *aux, const float *param, void *out) {
  if (!htpStrictOps()->inference_math_fp32(kind, rows, width, chunk, eps, scale,
                                           x, aux, param, out))
    throw std::runtime_error("NNTR_HTP_STRICT: unsupported inference math");
}
} // namespace causallm
#endif
