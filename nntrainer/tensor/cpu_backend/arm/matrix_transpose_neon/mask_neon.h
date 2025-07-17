// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   mask_neon.h
 * @date   09 May 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is mask shaped filters to distill certain indices using SIMD
 *
 */

#include <cstdint>

// clang-format off
alignas(16) static const int16_t neon_16bit_masks[9][8] = {
  {  0,  0,  0,  0,  0,  0,  0,  0,  },
  { -1,  0,  0,  0,  0,  0,  0,  0,  },
  { -1, -1,  0,  0,  0,  0,  0,  0,  },
  { -1, -1, -1,  0,  0,  0,  0,  0,  },
  { -1, -1, -1, -1,  0,  0,  0,  0,  },
  { -1, -1, -1, -1, -1,  0,  0,  0,  },
  { -1, -1, -1, -1, -1, -1,  0,  0,  },
  { -1, -1, -1, -1, -1, -1, -1,  0,  },
  { -1, -1, -1, -1, -1, -1, -1, -1,  },
};

alignas(16) static const int16_t masks[5][4] = {
  {  0,  0,  0,  0, },
  { -1,  0,  0,  0, },
  { -1, -1,  0,  0, },
  { -1, -1, -1,  0, },
  { -1, -1, -1, -1, },
};

alignas(16) static const int16_t shuffle_masks[8] = {
    -1, -1, 0, 0, -1, -1,  0,  0, 
};
// clang-format on
