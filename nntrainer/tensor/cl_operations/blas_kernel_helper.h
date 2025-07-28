#ifndef __BLAS_KERNELS_HELPER_H__
#define __BLAS_KERNELS_HELPER_H__

namespace nntrainer {

/**
 * @brief Convert array of block_q4_0x8 into two arrays for scale, qs (st)
 * @param[in] x Array of struct block_q4_0x8
 * @param[out] d Array of unsigned short
 * @param[out] qs Array of unsigned char
 * @param N size of @param x
 */
void convert_st(const void *x, unsigned short *d, unsigned char *qs, size_t N);

/**
 * @brief Convert array of block_q4_0x8 into two arrays for scale, qs (mt)
 * @param[in] x Array of struct block_q4_0x8
 * @param[out] d Array of unsigned short
 * @param[out] qs Array of unsigned char
 * @param N size of @param x
 */
void convert_omp(const void *x, unsigned short *d, unsigned char *qs, size_t N);

} // namespace nntrainer

#endif /* __BLAS_KERNELS_HELPER_H__ */
