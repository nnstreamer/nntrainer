#include <nntr_ggml_impl.h>

#include <assert.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <unordered_map>

typedef uint16_t ggml_half;
typedef uint32_t ggml_half2;

typedef uint16_t ggml_fp16_t;

typedef void (*ggml_to_float_t)(const void *__restrict x, float *__restrict y,
                                int64_t k);
typedef void (*ggml_from_float_t)(const float *__restrict x, void *__restrict y,
                                  int64_t k);

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

#define GROUP_MAX_EPS 1e-15f

#define QK_K 256
#define K_SCALE_SIZE 12

#define QK4_0 32
typedef struct {
  ggml_half d;           // delta
  uint8_t qs[QK4_0 / 2]; // nibbles / quants
} block_q4_0;

static_assert(sizeof(block_q4_0) == sizeof(ggml_half) + QK4_0 / 2,
              "wrong q4_0 block size/padding");

#define QK8_0 32
typedef struct {
  ggml_half d;      // delta
  int8_t qs[QK8_0]; // quants
} block_q8_0;

static_assert(sizeof(block_q8_0) == sizeof(ggml_half) + QK8_0,
              "wrong q8_0 block size/padding");

typedef struct {
  union {
    struct {
      ggml_half d;    // super-block scale for quantized scales
      ggml_half dmin; // super-block scale for quantized mins
    };
    ggml_half2 dm;
  };
  uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
  uint8_t qs[QK_K / 2];         // 4--bit quants
} block_q4_K;

static_assert(sizeof(block_q4_K) ==
                2 * sizeof(ggml_half) + K_SCALE_SIZE + QK_K / 2,
              "wrong q4_K block size/padding");

// 6-bit quantization
// weight is represented as x = a * q
// 16 blocks of 16 elements each
// Effectively 6.5625 bits per weight
typedef struct {
  uint8_t ql[QK_K / 2];     // quants, lower 4 bits
  uint8_t qh[QK_K / 4];     // quants, upper 2 bits
  int8_t scales[QK_K / 16]; // scales, quantized with 8 bits
  ggml_half d;              // super-block scale
} block_q6_K;

static_assert(sizeof(block_q6_K) ==
                sizeof(ggml_half) + QK_K / 16 + 3 * QK_K / 4,
              "wrong q6_K block size/padding");

typedef struct {
  float d;                  // delta
  int8_t qs[QK_K];          // quants
  int16_t bsums[QK_K / 16]; // sum of quants in groups of 16
} block_q8_K;

static_assert(sizeof(block_q8_K) ==
                sizeof(float) + QK_K + QK_K / 16 * sizeof(int16_t),
              "wrong q8_K block size/padding");

enum ggml_type {
  GGML_TYPE_F32 = 0,
  GGML_TYPE_F16 = 1,
  GGML_TYPE_Q4_0 = 2,
  GGML_TYPE_Q4_1 = 3,
  GGML_TYPE_Q8_0 = 8,
  GGML_TYPE_Q4_K = 12,
  GGML_TYPE_Q6_K = 14,
  GGML_TYPE_Q8_K = 15,
  GGML_TYPE_COUNT = 39,
};

struct ggml_type_traits {
  const char *type_name;
  int64_t blck_size;
  int64_t blck_size_interleave; // interleave elements in blocks
  size_t type_size;
  bool is_quantized;
  ggml_to_float_t to_float;
  ggml_from_float_t from_float_ref;
};

static const std::unordered_map<ggml_type, ggml_type_traits> type_traits = {
  {GGML_TYPE_Q4_0,
   ggml_type_traits{
     "q4_0",
     QK4_0,
     sizeof(block_q4_0),
     true,
   }},
  {GGML_TYPE_Q8_0,
   ggml_type_traits{
     "q4_0",
     QK8_0,
     sizeof(block_q8_0),
     true,
   }},
  {GGML_TYPE_Q4_K,
   ggml_type_traits{
     "q4_K",
     QK_K,
     sizeof(block_q4_K),
     true,
   }},
  {GGML_TYPE_Q6_K,
   ggml_type_traits{
     "q6_K",
     QK_K,
     sizeof(block_q6_K),
     true,
   }},
  {GGML_TYPE_Q8_K,
   ggml_type_traits{
     "q8_K",
     QK_K,
     sizeof(block_q8_K),
     true,
   }},
};

//
// ===================== Helper functions
//
static inline int nearest_int(float fval) {
  assert(fabsf(fval) <= 4194303.f);
  float val = fval + 12582912.f;
  int i;
  memcpy(&i, &val, sizeof(int));
  return (i & 0x007fffff) - 0x00400000;
}

static float make_qx_quants(int n, int nmax, const float *__restrict x,
                            int8_t *__restrict L, int rmse_type,
                            const float *__restrict qw) {
  float max = 0;
  float amax = 0;
  for (int i = 0; i < n; ++i) {
    float ax = fabsf(x[i]);
    if (ax > amax) {
      amax = ax;
      max = x[i];
    }
  }
  if (amax < GROUP_MAX_EPS) { // all zero
    for (int i = 0; i < n; ++i) {
      L[i] = 0;
    }
    return 0.f;
  }
  float iscale = -nmax / max;
  if (rmse_type == 0) {
    for (int i = 0; i < n; ++i) {
      int l = nearest_int(iscale * x[i]);
      L[i] = nmax + MAX(-nmax, MIN(nmax - 1, l));
    }
    return 1 / iscale;
  }
  bool return_early = false;
  if (rmse_type < 0) {
    rmse_type = -rmse_type;
    return_early = true;
  }
  float sumlx = 0;
  float suml2 = 0;
#ifdef HAVE_BUGGY_APPLE_LINKER
  // use 'volatile' to prevent unroll and work around a bug in Apple ld64 1015.7
  for (volatile int i = 0; i < n; ++i) {
#else
  for (int i = 0; i < n; ++i) {
#endif
    int l = nearest_int(iscale * x[i]);
    l = MAX(-nmax, MIN(nmax - 1, l));
    L[i] = l + nmax;
    float w = qw               ? qw[i]
              : rmse_type == 1 ? x[i] * x[i]
              : rmse_type == 2 ? 1
              : rmse_type == 3 ? fabsf(x[i])
                               : sqrtf(fabsf(x[i]));
    sumlx += w * x[i] * l;
    suml2 += w * l * l;
  }
  float scale = suml2 ? sumlx / suml2 : 0.0f;
  if (return_early)
    return suml2 > 0 ? 0.5f * (scale + 1 / iscale) : 1 / iscale;
  float best = scale * sumlx;
  for (int is = -9; is <= 9; ++is) {
    if (is == 0) {
      continue;
    }
    iscale = -(nmax + 0.1f * is) / max;
    sumlx = suml2 = 0;
    for (int i = 0; i < n; ++i) {
      int l = nearest_int(iscale * x[i]);
      l = MAX(-nmax, MIN(nmax - 1, l));
      float w = qw               ? qw[i]
                : rmse_type == 1 ? x[i] * x[i]
                : rmse_type == 2 ? 1
                : rmse_type == 3 ? fabsf(x[i])
                                 : sqrtf(fabsf(x[i]));
      sumlx += w * x[i] * l;
      suml2 += w * l * l;
    }
    if (suml2 > 0 && sumlx * sumlx > best * suml2) {
      for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale * x[i]);
        L[i] = nmax + MAX(-nmax, MIN(nmax - 1, l));
      }
      scale = sumlx / suml2;
      best = scale * sumlx;
    }
  }
  return scale;
}

static float make_q3_quants(int n, int nmax, const float *__restrict x,
                            int8_t *__restrict L, bool do_rmse) {
  float max = 0;
  float amax = 0;
  for (int i = 0; i < n; ++i) {
    float ax = fabsf(x[i]);
    if (ax > amax) {
      amax = ax;
      max = x[i];
    }
  }
  if (amax < GROUP_MAX_EPS) { // all zero
    for (int i = 0; i < n; ++i) {
      L[i] = 0;
    }
    return 0.f;
  }
  float iscale = -nmax / max;
  if (do_rmse) {
    float sumlx = 0;
    float suml2 = 0;
    for (int i = 0; i < n; ++i) {
      int l = nearest_int(iscale * x[i]);
      l = MAX(-nmax, MIN(nmax - 1, l));
      L[i] = l;
      float w = x[i] * x[i];
      sumlx += w * x[i] * l;
      suml2 += w * l * l;
    }
    for (int itry = 0; itry < 5; ++itry) {
      int n_changed = 0;
      for (int i = 0; i < n; ++i) {
        float w = x[i] * x[i];
        float slx = sumlx - w * x[i] * L[i];
        if (slx > 0) {
          float sl2 = suml2 - w * L[i] * L[i];
          int new_l = nearest_int(x[i] * sl2 / slx);
          new_l = MAX(-nmax, MIN(nmax - 1, new_l));
          if (new_l != L[i]) {
            slx += w * x[i] * new_l;
            sl2 += w * new_l * new_l;
            if (sl2 > 0 && slx * slx * suml2 > sumlx * sumlx * sl2) {
              L[i] = new_l;
              sumlx = slx;
              suml2 = sl2;
              ++n_changed;
            }
          }
        }
      }
      if (!n_changed) {
        break;
      }
    }
    for (int i = 0; i < n; ++i) {
      L[i] += nmax;
    }
    return sumlx / suml2;
  }
  for (int i = 0; i < n; ++i) {
    int l = nearest_int(iscale * x[i]);
    l = MAX(-nmax, MIN(nmax - 1, l));
    L[i] = l + nmax;
  }
  return 1 / iscale;
}

static float make_qkx1_quants(int n, int nmax, const float *__restrict x,
                              uint8_t *__restrict L, float *__restrict the_min,
                              int ntry, float alpha) {
  float min = x[0];
  float max = x[0];
  for (int i = 1; i < n; ++i) {
    if (x[i] < min)
      min = x[i];
    if (x[i] > max)
      max = x[i];
  }
  if (max == min) {
    for (int i = 0; i < n; ++i)
      L[i] = 0;
    *the_min = 0;
    return 0.f;
  }
  if (min > 0)
    min = 0;
  float iscale = nmax / (max - min);
  float scale = 1 / iscale;
  for (int itry = 0; itry < ntry; ++itry) {
    float sumlx = 0;
    int suml2 = 0;
    bool did_change = false;
    for (int i = 0; i < n; ++i) {
      int l = nearest_int(iscale * (x[i] - min));
      l = MAX(0, MIN(nmax, l));
      if (l != L[i]) {
        L[i] = l;
        did_change = true;
      }
      sumlx += (x[i] - min) * l;
      suml2 += l * l;
    }
    scale = sumlx / suml2;
    float sum = 0;
    for (int i = 0; i < n; ++i) {
      sum += x[i] - scale * L[i];
    }
    min = alpha * min + (1 - alpha) * sum / n;
    if (min > 0)
      min = 0;
    iscale = 1 / scale;
    if (!did_change)
      break;
  }
  *the_min = -min;
  return scale;
}

static float make_qkx2_quants(int n, int nmax, const float *__restrict x,
                              const float *__restrict weights,
                              uint8_t *__restrict L, float *__restrict the_min,
                              uint8_t *__restrict Laux, float rmin,
                              float rdelta, int nstep, bool use_mad) {
  float min = x[0];
  float max = x[0];
  float sum_w = weights[0];
  float sum_x = sum_w * x[0];
#ifdef HAVE_BUGGY_APPLE_LINKER
  // use 'volatile' to prevent unroll and work around a bug in Apple ld64 1015.7
  for (volatile int i = 1; i < n; ++i) {
#else
  for (int i = 1; i < n; ++i) {
#endif
    if (x[i] < min)
      min = x[i];
    if (x[i] > max)
      max = x[i];
    float w = weights[i];
    sum_w += w;
    sum_x += w * x[i];
  }
  if (min > 0)
    min = 0;
  if (max == min) {
    for (int i = 0; i < n; ++i)
      L[i] = 0;
    *the_min = -min;
    return 0.f;
  }
  float iscale = nmax / (max - min);
  float scale = 1 / iscale;
  float best_mad = 0;
  for (int i = 0; i < n; ++i) {
    int l = nearest_int(iscale * (x[i] - min));
    L[i] = MAX(0, MIN(nmax, l));
    float diff = scale * L[i] + min - x[i];
    diff = use_mad ? fabsf(diff) : diff * diff;
    float w = weights[i];
    best_mad += w * diff;
  }
  if (nstep < 1) {
    *the_min = -min;
    return scale;
  }
  for (int is = 0; is <= nstep; ++is) {
    iscale = (rmin + rdelta * is + nmax) / (max - min);
    float sum_l = 0, sum_l2 = 0, sum_xl = 0;
    for (int i = 0; i < n; ++i) {
      int l = nearest_int(iscale * (x[i] - min));
      l = MAX(0, MIN(nmax, l));
      Laux[i] = l;
      float w = weights[i];
      sum_l += w * l;
      sum_l2 += w * l * l;
      sum_xl += w * l * x[i];
    }
    float D = sum_w * sum_l2 - sum_l * sum_l;
    if (D > 0) {
      float this_scale = (sum_w * sum_xl - sum_x * sum_l) / D;
      float this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D;
      if (this_min > 0) {
        this_min = 0;
        this_scale = sum_xl / sum_l2;
      }
      float mad = 0;
      for (int i = 0; i < n; ++i) {
        float diff = this_scale * Laux[i] + this_min - x[i];
        diff = use_mad ? fabsf(diff) : diff * diff;
        float w = weights[i];
        mad += w * diff;
      }
      if (mad < best_mad) {
        for (int i = 0; i < n; ++i) {
          L[i] = Laux[i];
        }
        best_mad = mad;
        scale = this_scale;
        min = this_min;
      }
    }
  }
  *the_min = -min;
  return scale;
}

static float make_qkx3_quants(int n, int nmax, const float *__restrict x,
                              const float *__restrict weights,
                              uint8_t *__restrict L, float *__restrict the_min,
                              uint8_t *__restrict Laux, float rmin,
                              float rdelta, int nstep, bool use_mad) {
  float min = x[0];
  float max = x[0];
  float sum_w = weights ? weights[0] : x[0] * x[0];
  float sum_x = sum_w * x[0];
#ifdef HAVE_BUGGY_APPLE_LINKER
  // use 'volatile' to prevent unroll and work around a bug in Apple ld64 1015.7
  for (volatile int i = 1; i < n; ++i) {
#else
  for (int i = 1; i < n; ++i) {
#endif
    if (x[i] < min)
      min = x[i];
    if (x[i] > max)
      max = x[i];
    float w = weights ? weights[i] : x[i] * x[i];
    sum_w += w;
    sum_x += w * x[i];
  }
  if (min > 0) {
    min = 0;
  }
  if (max <= min) {
    memset(L, 0, n);
    *the_min = -min;
    return 0.f;
  }
  float iscale = nmax / (max - min);
  float scale = 1 / iscale;
  float best_mad = 0;
  for (int i = 0; i < n; ++i) {
    int l = nearest_int(iscale * (x[i] - min));
    L[i] = MAX(0, MIN(nmax, l));
    float diff = scale * L[i] + min - x[i];
    diff = use_mad ? fabsf(diff) : diff * diff;
    float w = weights ? weights[i] : x[i] * x[i];
    best_mad += w * diff;
  }
  if (nstep < 1) {
    *the_min = -min;
    return scale;
  }
  for (int is = 0; is <= nstep; ++is) {
    iscale = (rmin + rdelta * is + nmax) / (max - min);
    float sum_l = 0, sum_l2 = 0, sum_xl = 0;
    for (int i = 0; i < n; ++i) {
      int l = nearest_int(iscale * (x[i] - min));
      l = MAX(0, MIN(nmax, l));
      Laux[i] = l;
      float w = weights ? weights[i] : x[i] * x[i];
      sum_l += w * l;
      sum_l2 += w * l * l;
      sum_xl += w * l * x[i];
    }
    float D = sum_w * sum_l2 - sum_l * sum_l;
    if (D > 0) {
      float this_scale = (sum_w * sum_xl - sum_x * sum_l) / D;
      float this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D;
      if (this_min > 0) {
        this_min = 0;
        this_scale = sum_xl / sum_l2;
      }
      float mad = 0;
      for (int i = 0; i < n; ++i) {
        float diff = this_scale * Laux[i] + this_min - x[i];
        diff = use_mad ? fabsf(diff) : diff * diff;
        float w = weights ? weights[i] : x[i] * x[i];
        mad += w * diff;
      }
      if (mad < best_mad) {
        for (int i = 0; i < n; ++i) {
          L[i] = Laux[i];
        }
        best_mad = mad;
        scale = this_scale;
        min = this_min;
      }
    }
  }
  *the_min = -min;
  return scale;
}

static float make_qp_quants(int n, int nmax, const float *__restrict x,
                            uint8_t *__restrict L, const float *quant_weights) {
  float max = 0;
  for (int i = 0; i < n; ++i) {
    max = MAX(max, x[i]);
  }
  if (!max) { // all zero
    for (int i = 0; i < n; ++i) {
      L[i] = 0;
    }
    return 0.f;
  }
  float iscale = nmax / max;
  for (int i = 0; i < n; ++i) {
    L[i] = nearest_int(iscale * x[i]);
  }
  float scale = 1 / iscale;
  float best_mse = 0;
  for (int i = 0; i < n; ++i) {
    float diff = x[i] - scale * L[i];
    float w = quant_weights[i];
    best_mse += w * diff * diff;
  }
  for (int is = -4; is <= 4; ++is) {
    if (is == 0)
      continue;
    float iscale_is = (0.1f * is + nmax) / max;
    float scale_is = 1 / iscale_is;
    float mse = 0;
    for (int i = 0; i < n; ++i) {
      int l = nearest_int(iscale_is * x[i]);
      l = MIN(nmax, l);
      float diff = x[i] - scale_is * l;
      float w = quant_weights[i];
      mse += w * diff * diff;
    }
    if (mse < best_mse) {
      best_mse = mse;
      iscale = iscale_is;
    }
  }
  float sumlx = 0;
  float suml2 = 0;
  for (int i = 0; i < n; ++i) {
    int l = nearest_int(iscale * x[i]);
    l = MIN(nmax, l);
    L[i] = l;
    float w = quant_weights[i];
    sumlx += w * x[i] * l;
    suml2 += w * l * l;
  }
  for (int itry = 0; itry < 5; ++itry) {
    int n_changed = 0;
    for (int i = 0; i < n; ++i) {
      float w = quant_weights[i];
      float slx = sumlx - w * x[i] * L[i];
      float sl2 = suml2 - w * L[i] * L[i];
      if (slx > 0 && sl2 > 0) {
        int new_l = nearest_int(x[i] * sl2 / slx);
        new_l = MIN(nmax, new_l);
        if (new_l != L[i]) {
          slx += w * x[i] * new_l;
          sl2 += w * new_l * new_l;
          if (slx * slx * suml2 > sumlx * sumlx * sl2) {
            L[i] = new_l;
            sumlx = slx;
            suml2 = sl2;
            ++n_changed;
          }
        }
      }
    }
    if (!n_changed) {
      break;
    }
  }
  return sumlx / suml2;
}

static inline void get_scale_min_k4(int j, const uint8_t *__restrict q,
                                    uint8_t *__restrict d,
                                    uint8_t *__restrict m) {
  if (j < 4) {
    *d = q[j] & 63;
    *m = q[j + 4] & 63;
  } else {
    *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
  }
}

static inline float fp32_from_bits(uint32_t w) {
  union {
    uint32_t as_bits;
    float as_value;
  } fp32;
  fp32.as_bits = w;
  return fp32.as_value;
}

static inline uint32_t fp32_to_bits(float f) {
  union {
    float as_value;
    uint32_t as_bits;
  } fp32;
  fp32.as_value = f;
  return fp32.as_bits;
}

static inline float ggml_compute_fp16_to_fp32(ggml_fp16_t h) {
  const uint32_t w = (uint32_t)h << 16;
  const uint32_t sign = w & UINT32_C(0x80000000);
  const uint32_t two_w = w + w;

  const uint32_t exp_offset = UINT32_C(0xE0) << 23;
#if (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) ||             \
     defined(__GNUC__) && !defined(__STRICT_ANSI__)) &&                        \
  (!defined(__cplusplus) || __cplusplus >= 201703L)
  const float exp_scale = 0x1.0p-112f;
#else
  const float exp_scale = fp32_from_bits(UINT32_C(0x7800000));
#endif
  const float normalized_value =
    fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

  const uint32_t magic_mask = UINT32_C(126) << 23;
  const float magic_bias = 0.5f;
  const float denormalized_value =
    fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

  const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
  const uint32_t result =
    sign | (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value)
                                        : fp32_to_bits(normalized_value));
  return fp32_from_bits(result);
}

static inline ggml_fp16_t ggml_compute_fp32_to_fp16(float f) {
#if (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) ||             \
     defined(__GNUC__) && !defined(__STRICT_ANSI__)) &&                        \
  (!defined(__cplusplus) || __cplusplus >= 201703L)
  const float scale_to_inf = 0x1.0p+112f;
  const float scale_to_zero = 0x1.0p-110f;
#else
  const float scale_to_inf = fp32_from_bits(UINT32_C(0x77800000));
  const float scale_to_zero = fp32_from_bits(UINT32_C(0x08800000));
#endif
  float base = (fabsf(f) * scale_to_inf) * scale_to_zero;

  const uint32_t w = fp32_to_bits(f);
  const uint32_t shl1_w = w + w;
  const uint32_t sign = w & UINT32_C(0x80000000);
  uint32_t bias = shl1_w & UINT32_C(0xFF000000);
  if (bias < UINT32_C(0x71000000)) {
    bias = UINT32_C(0x71000000);
  }

  base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
  const uint32_t bits = fp32_to_bits(base);
  const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
  const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
  const uint32_t nonsign = exp_bits + mantissa_bits;
  return (sign >> 16) |
         (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign);
}

#define GGML_COMPUTE_FP16_TO_FP32(x) ggml_compute_fp16_to_fp32(x)
#define GGML_COMPUTE_FP32_TO_FP16(x) ggml_compute_fp32_to_fp16(x)
#define GGML_FP16_TO_FP32(x) GGML_COMPUTE_FP16_TO_FP32(x)
#define GGML_FP32_TO_FP16(x) GGML_COMPUTE_FP32_TO_FP16(x)

int64_t ggml_blck_size(enum ggml_type type) {
  return type_traits.at(type).blck_size;
}

size_t ggml_type_size(enum ggml_type type) {
  return type_traits.at(type).type_size;
}

size_t ggml_row_size(enum ggml_type type, int64_t ne) {
  assert(ne % ggml_blck_size(type) == 0);
  return ggml_type_size(type) * ne / ggml_blck_size(type);
}

// -------- Q4_0 -----------------

void quantize_row_q4_0_ref(const float *__restrict x, block_q4_0 *__restrict y,
                           int64_t k) {
  static const int qk = QK4_0;

  assert(k % qk == 0);

  const int nb = k / qk;

  for (int i = 0; i < nb; i++) {
    float amax = 0.0f; // absolute max
    float max = 0.0f;

    for (int j = 0; j < qk; j++) {
      const float v = x[i * qk + j];
      if (amax < fabsf(v)) {
        amax = fabsf(v);
        max = v;
      }
    }

    const float d = max / -8;
    const float id = d ? 1.0f / d : 0.0f;

    y[i].d = GGML_FP32_TO_FP16(d);

    for (int j = 0; j < qk / 2; ++j) {
      const float x0 = x[i * qk + 0 + j] * id;
      const float x1 = x[i * qk + qk / 2 + j] * id;

      const uint8_t xi0 = MIN(15, (int8_t)(x0 + 8.5f));
      const uint8_t xi1 = MIN(15, (int8_t)(x1 + 8.5f));

      y[i].qs[j] = xi0;
      y[i].qs[j] |= xi1 << 4;
    }
  }
}

static void quantize_row_q4_0_impl(const float *__restrict x,
                                   block_q4_0 *__restrict y, int64_t n_per_row,
                                   const float *quant_weights) {
  static_assert(QK4_0 == 32, "QK4_0 must be 32");

  if (!quant_weights) {
    quantize_row_q4_0_ref(x, y, n_per_row);
    return;
  }

  float weight[QK4_0];
  int8_t L[QK4_0];

  float sum_x2 = 0;
  for (int j = 0; j < n_per_row; ++j)
    sum_x2 += x[j] * x[j];
  float sigma2 = sum_x2 / n_per_row;

  const int64_t nb = n_per_row / QK4_0;
  for (int ib = 0; ib < nb; ++ib) {
    const float *xb = x + QK4_0 * ib;
    const float *qw = quant_weights + QK4_0 * ib;
    for (int j = 0; j < QK4_0; ++j)
      weight[j] = qw[j] * sqrtf(sigma2 + xb[j] * xb[j]);
    float d = make_qx_quants(QK4_0, 8, xb, L, 1, weight);
    y[ib].d = GGML_FP32_TO_FP16(d);
    for (int j = 0; j < 16; ++j) {
      y[ib].qs[j] = L[j] | (L[j + 16] << 4);
    }
  }
}

size_t nntr_quantize_q4_0(const float *__restrict src, void *__restrict dst,
                          int64_t nrow, int64_t n_per_row,
                          const float *quant_weights) {
  if (!quant_weights) {
    quantize_row_q4_0_ref(src, (block_q4_0 *)dst, (int64_t)nrow * n_per_row);
    return nrow * ggml_row_size(GGML_TYPE_Q4_0, n_per_row);
  }
  size_t row_size = ggml_row_size(GGML_TYPE_Q4_0, n_per_row);
  char *qrow = (char *)dst;
  for (int64_t row = 0; row < nrow; ++row) {
    quantize_row_q4_0_impl(src, (block_q4_0 *)qrow, n_per_row, quant_weights);
    src += n_per_row;
    qrow += row_size;
  }
  return nrow * row_size;
}

// -------- Q8_0 -----------------

void quantize_row_q8_0_ref(const float *__restrict x, block_q8_0 *__restrict y,
                           int64_t k) {
  assert(k % QK8_0 == 0);
  const int nb = k / QK8_0;

  for (int i = 0; i < nb; i++) {
    float amax = 0.0f; // absolute max

    for (int j = 0; j < QK8_0; j++) {
      const float v = x[i * QK8_0 + j];
      amax = MAX(amax, fabsf(v));
    }

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f / d : 0.0f;

    y[i].d = GGML_FP32_TO_FP16(d);

    for (int j = 0; j < QK8_0; ++j) {
      const float x0 = x[i * QK8_0 + j] * id;

      y[i].qs[j] = roundf(x0);
    }
  }
}

size_t quantize_q8_0(const float *__restrict src, void *__restrict dst,
                     int64_t nrow, int64_t n_per_row,
                     const float *quant_weights) {
  (void)quant_weights; // not used
  const size_t row_size = ggml_row_size(GGML_TYPE_Q8_0, n_per_row);
  quantize_row_q8_0_ref(src, (block_q8_0 *)dst, (int64_t)nrow * n_per_row);
  return nrow * row_size;
}

// -------- Q4_K -----------------

void quantize_row_q4_K_ref(const float *__restrict x, block_q4_K *__restrict y,
                           int64_t k) {
  assert(k % QK_K == 0);
  const int nb = k / QK_K;

  uint8_t L[QK_K];
  uint8_t Laux[32];
  float weights[32];
  float mins[QK_K / 32];
  float scales[QK_K / 32];

  for (int i = 0; i < nb; i++) {
    float max_scale =
      0; // as we are deducting the min, scales are always positive
    float max_min = 0;
    for (int j = 0; j < QK_K / 32; ++j) {
      // scales[j] = make_qkx1_quants(32, 15, x + 32*j, L + 32*j, &mins[j], 9,
      // 0.5f);
      float sum_x2 = 0;
      for (int l = 0; l < 32; ++l)
        sum_x2 += x[32 * j + l] * x[32 * j + l];
      float av_x = sqrtf(sum_x2 / 32);
      for (int l = 0; l < 32; ++l)
        weights[l] = av_x + fabsf(x[32 * j + l]);
      scales[j] = make_qkx2_quants(32, 15, x + 32 * j, weights, L + 32 * j,
                                   &mins[j], Laux, -1.f, 0.1f, 20, false);
      float scale = scales[j];
      if (scale > max_scale) {
        max_scale = scale;
      }
      float min = mins[j];
      if (min > max_min) {
        max_min = min;
      }
    }

    float inv_scale = max_scale > 0 ? 63.f / max_scale : 0.f;
    float inv_min = max_min > 0 ? 63.f / max_min : 0.f;
    for (int j = 0; j < QK_K / 32; ++j) {
      uint8_t ls = nearest_int(inv_scale * scales[j]);
      uint8_t lm = nearest_int(inv_min * mins[j]);
      ls = MIN(63, ls);
      lm = MIN(63, lm);
      if (j < 4) {
        y[i].scales[j] = ls;
        y[i].scales[j + 4] = lm;
      } else {
        y[i].scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
        y[i].scales[j - 4] |= ((ls >> 4) << 6);
        y[i].scales[j - 0] |= ((lm >> 4) << 6);
      }
    }
    y[i].d = GGML_FP32_TO_FP16(max_scale / 63.f);
    y[i].dmin = GGML_FP32_TO_FP16(max_min / 63.f);

    uint8_t sc, m;
    for (int j = 0; j < QK_K / 32; ++j) {
      get_scale_min_k4(j, y[i].scales, &sc, &m);
      const float d = GGML_FP16_TO_FP32(y[i].d) * sc;
      if (!d)
        continue;
      const float dm = GGML_FP16_TO_FP32(y[i].dmin) * m;
      for (int ii = 0; ii < 32; ++ii) {
        int l = nearest_int((x[32 * j + ii] + dm) / d);
        l = MAX(0, MIN(15, l));
        L[32 * j + ii] = l;
      }
    }

    uint8_t *q = y[i].qs;
    for (int j = 0; j < QK_K; j += 64) {
      for (int l = 0; l < 32; ++l)
        q[l] = L[j + l] | (L[j + l + 32] << 4);
      q += 32;
    }

    x += QK_K;
  }
}

static void quantize_row_q4_K_impl(const float *__restrict x,
                                   block_q4_K *__restrict y, int64_t n_per_row,
                                   const float *quant_weights) {
  assert(n_per_row % QK_K == 0);
  const int64_t nb = n_per_row / QK_K;

  uint8_t L[QK_K];
  uint8_t Laux[32];
  uint8_t Ls[QK_K / 32];
  uint8_t Lm[QK_K / 32];
  float weights[32];
  float sw[QK_K / 32];
  float mins[QK_K / 32];
  float scales[QK_K / 32];

  for (int i = 0; i < nb; i++) {

    float sum_x2 = 0;
    for (int l = 0; l < QK_K; ++l)
      sum_x2 += x[l] * x[l];
    float sigma2 = 2 * sum_x2 / QK_K;
    float av_x = sqrtf(sigma2);

    for (int j = 0; j < QK_K / 32; ++j) {
      if (quant_weights) {
        const float *qw = quant_weights + QK_K * i + 32 * j;
        for (int l = 0; l < 32; ++l)
          weights[l] = qw[l] * sqrtf(sigma2 + x[32 * j + l] * x[32 * j + l]);
      } else {
        for (int l = 0; l < 32; ++l)
          weights[l] = av_x + fabsf(x[32 * j + l]);
      }
      float sumw = 0;
      for (int l = 0; l < 32; ++l)
        sumw += weights[l];
      sw[j] = sumw;
      scales[j] = make_qkx3_quants(32, 15, x + 32 * j, weights, L + 32 * j,
                                   &mins[j], Laux, -0.9f, 0.05f, 36, false);
    }

    float d_block = make_qp_quants(QK_K / 32, 63, scales, Ls, sw);
    float m_block = make_qp_quants(QK_K / 32, 63, mins, Lm, sw);
    for (int j = 0; j < QK_K / 32; ++j) {
      uint8_t ls = Ls[j];
      uint8_t lm = Lm[j];
      if (j < 4) {
        y[i].scales[j] = ls;
        y[i].scales[j + 4] = lm;
      } else {
        y[i].scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
        y[i].scales[j - 4] |= ((ls >> 4) << 6);
        y[i].scales[j - 0] |= ((lm >> 4) << 6);
      }
    }
    y[i].d = GGML_FP32_TO_FP16(d_block);
    y[i].dmin = GGML_FP32_TO_FP16(m_block);

    uint8_t sc, m;
    for (int j = 0; j < QK_K / 32; ++j) {
      get_scale_min_k4(j, y[i].scales, &sc, &m);
      const float d = GGML_FP16_TO_FP32(y[i].d) * sc;
      if (!d)
        continue;
      const float dm = GGML_FP16_TO_FP32(y[i].dmin) * m;
      for (int ii = 0; ii < 32; ++ii) {
        int l = nearest_int((x[32 * j + ii] + dm) / d);
        l = MAX(0, MIN(15, l));
        L[32 * j + ii] = l;
      }
    }
    uint8_t *q = y[i].qs;
    for (int j = 0; j < QK_K; j += 64) {
      for (int l = 0; l < 32; ++l)
        q[l] = L[j + l] | (L[j + l + 32] << 4);
      q += 32;
    }

    x += QK_K;
  }
}

size_t nntr_quantize_q4_K(const float *__restrict src, void *__restrict dst,
                          int64_t nrow, int64_t n_per_row,
                          const float *quant_weights) {
  size_t row_size = ggml_row_size(GGML_TYPE_Q4_K, n_per_row);
  if (!quant_weights) {
    quantize_row_q4_K_ref(src, (block_q4_K *)dst, (int64_t)nrow * n_per_row);
  } else {
    char *qrow = (char *)dst;
    for (int64_t row = 0; row < nrow; ++row) {
      quantize_row_q4_K_impl(src, (block_q4_K *)qrow, n_per_row, quant_weights);
      src += n_per_row;
      qrow += row_size;
    }
  }
  return nrow * row_size;
}

// --------- Q6_K --------------

void quantize_row_q6_K_ref(const float *__restrict x, block_q6_K *__restrict y,
                           int64_t k) {
  assert(k % QK_K == 0);
  const int64_t nb = k / QK_K;

  int8_t L[QK_K];
  float scales[QK_K / 16];

  for (int i = 0; i < nb; i++) {

    float max_scale = 0;
    float max_abs_scale = 0;

    for (int ib = 0; ib < QK_K / 16; ++ib) {

      const float scale =
        make_qx_quants(16, 32, x + 16 * ib, L + 16 * ib, 1, NULL);
      scales[ib] = scale;

      const float abs_scale = fabsf(scale);
      if (abs_scale > max_abs_scale) {
        max_abs_scale = abs_scale;
        max_scale = scale;
      }
    }

    if (max_abs_scale < GROUP_MAX_EPS) {
      memset(&y[i], 0, sizeof(block_q6_K));
      y[i].d = GGML_FP32_TO_FP16(0.f);
      x += QK_K;
      continue;
    }

    float iscale = -128.f / max_scale;
    y[i].d = GGML_FP32_TO_FP16(1 / iscale);
    for (int ib = 0; ib < QK_K / 16; ++ib) {
      y[i].scales[ib] = MIN(127, nearest_int(iscale * scales[ib]));
    }

    for (int j = 0; j < QK_K / 16; ++j) {
      float d = GGML_FP16_TO_FP32(y[i].d) * y[i].scales[j];
      if (!d) {
        continue;
      }
      for (int ii = 0; ii < 16; ++ii) {
        int l = nearest_int(x[16 * j + ii] / d);
        l = MAX(-32, MIN(31, l));
        L[16 * j + ii] = l + 32;
      }
    }

    uint8_t *__restrict ql = y[i].ql;
    uint8_t *__restrict qh = y[i].qh;
    for (int j = 0; j < QK_K; j += 128) {
      for (int l = 0; l < 32; ++l) {
        const uint8_t q1 = L[j + l + 0] & 0xF;
        const uint8_t q2 = L[j + l + 32] & 0xF;
        const uint8_t q3 = L[j + l + 64] & 0xF;
        const uint8_t q4 = L[j + l + 96] & 0xF;
        ql[l + 0] = q1 | (q3 << 4);
        ql[l + 32] = q2 | (q4 << 4);
        qh[l] = (L[j + l] >> 4) | ((L[j + l + 32] >> 4) << 2) |
                ((L[j + l + 64] >> 4) << 4) | ((L[j + l + 96] >> 4) << 6);
      }
      ql += 64;
      qh += 32;
    }

    x += QK_K;
  }
}

static void quantize_row_q6_K_impl(const float *__restrict x,
                                   block_q6_K *__restrict y, int64_t n_per_row,
                                   const float *quant_weights) {
  assert(n_per_row % QK_K == 0);
  const int64_t nb = n_per_row / QK_K;

  int8_t L[QK_K];
  float scales[QK_K / 16];
  // float   weights[16];

  for (int i = 0; i < nb; i++) {

    // float sum_x2 = 0;
    // for (int j = 0; j < QK_K; ++j) sum_x2 += x[j]*x[j];
    // float sigma2 = sum_x2/QK_K;

    float max_scale = 0;
    float max_abs_scale = 0;

    for (int ib = 0; ib < QK_K / 16; ++ib) {

      float scale;
      if (quant_weights) {
        const float *qw = quant_weights + QK_K * i + 16 * ib;
        // for (int j = 0; j < 16; ++j) weights[j] = qw[j] * sqrtf(sigma2 +
        // x[16*ib + j]*x[16*ib + j]); scale = make_qx_quants(16, 32, x + 16*ib,
        // L + 16*ib, 1, weights);
        scale = make_qx_quants(16, 32, x + 16 * ib, L + 16 * ib, 1, qw);
      } else {
        scale = make_qx_quants(16, 32, x + 16 * ib, L + 16 * ib, 1, NULL);
      }
      scales[ib] = scale;

      const float abs_scale = fabsf(scale);
      if (abs_scale > max_abs_scale) {
        max_abs_scale = abs_scale;
        max_scale = scale;
      }
    }

    if (max_abs_scale < GROUP_MAX_EPS) {
      memset(&y[i], 0, sizeof(block_q6_K));
      y[i].d = GGML_FP32_TO_FP16(0.f);
      x += QK_K;
      continue;
    }

    float iscale = -128.f / max_scale;
    y[i].d = GGML_FP32_TO_FP16(1 / iscale);
    for (int ib = 0; ib < QK_K / 16; ++ib) {
      y[i].scales[ib] = MIN(127, nearest_int(iscale * scales[ib]));
    }

    for (int j = 0; j < QK_K / 16; ++j) {
      float d = GGML_FP16_TO_FP32(y[i].d) * y[i].scales[j];
      if (!d) {
        continue;
      }
      for (int ii = 0; ii < 16; ++ii) {
        int l = nearest_int(x[16 * j + ii] / d);
        l = MAX(-32, MIN(31, l));
        L[16 * j + ii] = l + 32;
      }
    }

    uint8_t *__restrict ql = y[i].ql;
    uint8_t *__restrict qh = y[i].qh;
    for (int j = 0; j < QK_K; j += 128) {
      for (int l = 0; l < 32; ++l) {
        const uint8_t q1 = L[j + l + 0] & 0xF;
        const uint8_t q2 = L[j + l + 32] & 0xF;
        const uint8_t q3 = L[j + l + 64] & 0xF;
        const uint8_t q4 = L[j + l + 96] & 0xF;
        ql[l + 0] = q1 | (q3 << 4);
        ql[l + 32] = q2 | (q4 << 4);
        qh[l] = (L[j + l] >> 4) | ((L[j + l + 32] >> 4) << 2) |
                ((L[j + l + 64] >> 4) << 4) | ((L[j + l + 96] >> 4) << 6);
      }
      ql += 64;
      qh += 32;
    }

    x += QK_K;
  }
}

size_t nntr_quantize_q6_K(const float *__restrict src, void *__restrict dst,
                          int64_t nrow, int64_t n_per_row,
                          const float *quant_weights) {
  size_t row_size = ggml_row_size(GGML_TYPE_Q6_K, n_per_row);
  if (!quant_weights) {
    quantize_row_q6_K_ref(src, (block_q6_K *)dst, (int64_t)nrow * n_per_row);
  } else {
    char *qrow = (char *)dst;
    for (int64_t row = 0; row < nrow; ++row) {
      quantize_row_q6_K_impl(src, (block_q6_K *)qrow, n_per_row, quant_weights);
      src += n_per_row;
      qrow += row_size;
    }
  }
  return nrow * row_size;
}

// --------- Q8_K --------------

void quantize_row_q8_K_ref(const float *__restrict x, block_q8_K *__restrict y,
                           int64_t k) {
  assert(k % QK_K == 0);
  const int64_t nb = k / QK_K;

  for (int i = 0; i < nb; i++) {

    float max = 0;
    float amax = 0;
    for (int j = 0; j < QK_K; ++j) {
      float ax = fabsf(x[j]);
      if (ax > amax) {
        amax = ax;
        max = x[j];
      }
    }
    if (!amax) {
      y[i].d = 0;
      memset(y[i].qs, 0, QK_K);
      x += QK_K;
      continue;
    }
    // const float iscale = -128.f/max;
    //  We need this change for IQ2_XXS, else the AVX implementation becomes
    //  very awkward
    const float iscale = -127.f / max;
    for (int j = 0; j < QK_K; ++j) {
      int v = nearest_int(iscale * x[j]);
      y[i].qs[j] = MIN(127, v);
    }
    for (int j = 0; j < QK_K / 16; ++j) {
      int sum = 0;
      for (int ii = 0; ii < 16; ++ii) {
        sum += y[i].qs[j * 16 + ii];
      }
      y[i].bsums[j] = sum;
    }
    y[i].d = 1 / iscale;
    x += QK_K;
  }
}

void nntr_quantize_row_q8_K(const float *__restrict x, void *__restrict y,
                            int64_t k) {
#ifdef __wasm_simd128__
  assert(k % QK_K == 0);
  const int64_t nb = k / QK_K;
  block_q8_K *__restrict yc = y; // Cast to proper type

  for (int i = 0; i < nb; i++) {
    const float *x_block = x + i * QK_K;

    v128_t min_vec = wasm_v128_load(x_block);
    v128_t max_vec = min_vec;

    for (int j = 4; j < QK_K; j += 4) {
      v128_t x_vec = wasm_v128_load(x_block + j);
      max_vec = wasm_f32x4_pmax(max_vec, x_vec);
      min_vec = wasm_f32x4_pmin(min_vec, x_vec);
    }
    max_vec = wasm_f32x4_pmax(max_vec,
                              wasm_i32x4_shuffle(max_vec, max_vec, 2, 3, 0, 1));
    max_vec = wasm_f32x4_pmax(max_vec,
                              wasm_i32x4_shuffle(max_vec, max_vec, 1, 0, 3, 2));
    min_vec = wasm_f32x4_pmin(min_vec,
                              wasm_i32x4_shuffle(min_vec, min_vec, 2, 3, 0, 1));
    min_vec = wasm_f32x4_pmin(min_vec,
                              wasm_i32x4_shuffle(min_vec, min_vec, 1, 0, 3, 2));
    float max = wasm_f32x4_extract_lane(max_vec, 0);
    float min = wasm_f32x4_extract_lane(min_vec, 0);
    float amax = -min > max ? min : max;

    if (amax == 0.0f) {
      yc[i].d = 0.0f;
      const v128_t zero = wasm_i8x16_splat(0);
      for (int j = 0; j < QK_K; j += 16) {
        wasm_v128_store(yc[i].qs + j, zero);
      }
      continue;
    }

    const float iscale = -127.0f / amax;
    const v128_t scale_vec = wasm_f32x4_splat(iscale);

    // Process 16 elements per iteration
    for (int j = 0, jb = 0; j < QK_K; j += 16, jb++) {
      // Load and quantize 16 floats
      v128_t x0 = wasm_v128_load(x_block + j);
      v128_t x1 = wasm_v128_load(x_block + j + 4);
      v128_t x2 = wasm_v128_load(x_block + j + 8);
      v128_t x3 = wasm_v128_load(x_block + j + 12);

      v128_t q0 = wasm_f32x4_nearest(wasm_f32x4_mul(x0, scale_vec));
      v128_t q1 = wasm_f32x4_nearest(wasm_f32x4_mul(x1, scale_vec));
      v128_t q2 = wasm_f32x4_nearest(wasm_f32x4_mul(x2, scale_vec));
      v128_t q3 = wasm_f32x4_nearest(wasm_f32x4_mul(x3, scale_vec));

      // Convert to i32 with saturation
      v128_t i0 = wasm_i32x4_trunc_sat_f32x4(q0);
      v128_t i1 = wasm_i32x4_trunc_sat_f32x4(q1);
      v128_t i2 = wasm_i32x4_trunc_sat_f32x4(q2);
      v128_t i3 = wasm_i32x4_trunc_sat_f32x4(q3);

      // Pack into 16 i8 values
      v128_t i8 = wasm_i8x16_narrow_i16x8(wasm_i16x8_narrow_i32x4(i0, i1),
                                          wasm_i16x8_narrow_i32x4(i2, i3));
      wasm_v128_store(yc[i].qs + j, i8);

      // Calculate bsums using SIMD
      v128_t sum16 = wasm_i16x8_add(wasm_i16x8_extend_low_i8x16(i8),
                                    wasm_i16x8_extend_high_i8x16(i8));
      v128_t sum32 = wasm_i32x4_add(wasm_i32x4_extend_low_i16x8(sum16),
                                    wasm_i32x4_extend_high_i16x8(sum16));
      sum32 =
        wasm_i32x4_add(sum32, wasm_i32x4_shuffle(sum32, sum32, 2, 3, 0, 1));
      sum32 =
        wasm_i32x4_add(sum32, wasm_i32x4_shuffle(sum32, sum32, 1, 0, 3, 2));
      yc[i].bsums[jb] = wasm_i32x4_extract_lane(sum32, 0);
    }

    yc[i].d = 1.0f / iscale;
  }
#else
  quantize_row_q8_K_ref(x, (block_q8_K *)y, k);
#endif
}
