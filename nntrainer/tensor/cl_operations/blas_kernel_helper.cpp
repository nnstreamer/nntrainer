#include "blas_kernel_helper.h"

#include <omp.h>

#include <cstring>

namespace nntrainer {

struct block_q4_0x8 {
  unsigned short d[8];
  unsigned char qs[128];
};

void convert_st(const void *src, unsigned short *d, unsigned char *qs, size_t N) {
  block_q4_0x8 *x = (block_q4_0x8 *)src;
  for (int i = 0; i < N; ++i) {
    std::memcpy(&d[i * 8], x[i].d, sizeof(unsigned short) * 8);
    std::memcpy(&qs[i * 128], x[i].qs, sizeof(unsigned char) * 128);
  }
}

void convert_omp(const void *src, unsigned short *d, unsigned char *qs, size_t N) {
  block_q4_0x8 *x = (block_q4_0x8 *)src;
#pragma omp parallel for
  for (int i = 0; i < N; ++i) {
    std::memcpy(&d[i * 8], x[i].d, sizeof(unsigned short) * 8);
    std::memcpy(&qs[i * 128], x[i].qs, sizeof(unsigned char) * 128);
  }
}

} // namespace nntrainer