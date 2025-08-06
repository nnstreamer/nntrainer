// Original code :
// https://github.com/ggml-org/ggml/blob/master/src/ggml-opencl/kernels/rms_norm.cl
// Original code :
// https://github.com/ggml-org/ggml/blob/master/src/ggml-opencl/kernels/add.cl
// Original code :
// https://github.com/ggml-org/ggml/blob/master/src/ggml-opencl/kernels/glu.cl

#ifndef __GGML_KERNELS_H__
#define __GGML_KERNELS_H__

#include <cl_buffer_manager.h>
#include <cl_context.h>
#include <engine.h>
#include <opencl_buffer.h>
#include <opencl_kernel.h>

namespace nntrainer {
// GLU
void ggml_swiglu_cl(void *src0, unsigned long offset0, void *src1,
                    unsigned long offset1, void *dst, unsigned long offsetd,
                    unsigned long nb01, unsigned long nb11, int32_t ne0,
                    unsigned long nb1, int32_t ne00_off, int32_t ne10_off);

void ggml_swiglu_f16_cl(void *src0, unsigned long offset0, void *src1,
                        unsigned long offset1, void *dst, unsigned long offsetd,
                        unsigned long nb01, unsigned long nb11, int32_t ne0,
                        unsigned long nb1, int32_t ne00_off, int32_t ne10_off);

// ADD
void ggml_add_cl(void *src0, unsigned long offset0, void *src1,
                 unsigned long offset1, void *dst, unsigned long offsetd,
                 int32_t ne00, int32_t ne01, int32_t ne02, int32_t ne03,
                 unsigned long nb00, unsigned long nb01, unsigned long nb02,
                 unsigned long nb03, int32_t ne10, int32_t ne11, int32_t ne12,
                 int32_t ne13, unsigned long nb10, unsigned long nb11,
                 unsigned long nb12, unsigned long nb13, int32_t ne0,
                 int32_t ne1, int32_t ne2, int32_t ne3, unsigned long nb0,
                 unsigned long nb1, unsigned long nb2, unsigned long nb3);

void ggml_add_f16_cl(void *src0, unsigned long offset0, void *src1,
                     unsigned long offset1, void *dst, unsigned long offsetd,
                     int32_t ne00, int32_t ne01, int32_t ne02, int32_t ne03,
                     unsigned long nb00, unsigned long nb01, unsigned long nb02,
                     unsigned long nb03, int32_t ne10, int32_t ne11,
                     int32_t ne12, int32_t ne13, unsigned long nb10,
                     unsigned long nb11, unsigned long nb12, unsigned long nb13,
                     int32_t ne0, int32_t ne1, int32_t ne2, int32_t ne3,
                     unsigned long nb0, unsigned long nb1, unsigned long nb2,
                     unsigned long nb3);

// RMS NORM
void ggml_rms_norm_cl(void *src0, unsigned long offset0, void *dst,
                      unsigned long offsetd, int32_t ne00, int32_t ne01,
                      int32_t ne02, int32_t ne03, unsigned long nb01,
                      unsigned long nb02, unsigned long nb03, float eps);

void ggml_rms_norm_mul_cl(void *src0, unsigned long offset0, void *src1,
                          unsigned long offset1, void *dst,
                          unsigned long offsetd, int32_t ne00, int32_t ne01,
                          int32_t ne02, int32_t ne03, unsigned long nb01,
                          unsigned long nb02, unsigned long nb03, int32_t ne10,
                          int32_t ne11, int32_t ne12, int32_t ne13,
                          unsigned long nb11, unsigned long nb12,
                          unsigned long nb13, unsigned long nb1,
                          unsigned long nb2, unsigned long nb3, float eps);
} // namespace nntrainer
#endif
