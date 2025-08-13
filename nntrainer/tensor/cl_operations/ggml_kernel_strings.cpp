// Original code :
// https://github.com/ggml-org/ggml/blob/master/src/ggml-opencl/kernels/rms_norm.cl
// Original code :
// https://github.com/ggml-org/ggml/blob/master/src/ggml-opencl/kernels/add.cl
// Original code :
// https://github.com/ggml-org/ggml/blob/master/src/ggml-opencl/kernels/glu.cl

#include "ggml_kernel_strings.h"

namespace nntrainer {

// glu.cl
const std::string &getGegluKernel() {
  static std::string kernel_geglu =
    R"(
#define GELU_COEF_A     0.044715f
#define GELU_QUICK_COEF -1.702f
#define SQRT_2_OVER_PI  0.79788456080286535587989211986876f
#define SQRT_2_INV      0.70710678118654752440084436210484f

kernel void kernel_geglu(
    global char * src0,
    ulong  offset0,
    global char * src1,
    ulong  offset1,
    global char * dst,
    ulong  offsetd,
    ulong nb01,
    ulong nb11,
    int ne0,
    ulong nb1,
    int ne00_off,
    int ne10_off
) {
    src0 = (global char*)((global char*)src0 + offset0);
    src1 = (global char*)((global char*)src1 + offset1);
    dst  = (global char*)((global char*)dst  + offsetd);

    global float * src0_row = (global float *) ((global char *) src0 + get_group_id(0)*nb01) + ne00_off;
    global float * src1_row = (global float *) ((global char *) src1 + get_group_id(0)*nb11) + ne10_off;
    global float * dst_row  = (global float *) ((global char *) dst  + get_group_id(0)*nb1);

    for (int i0 = get_local_id(0); i0 < ne0; i0 += get_local_size(0)) {
        const float x0 = src0_row[i0];
        const float x1 = src1_row[i0];

        const float gelu = 0.5f*x0*(1.0f + tanh(SQRT_2_OVER_PI*x0*(1.0f + GELU_COEF_A*x0*x0)));

        dst_row[i0] = gelu*x1;
    }
}
    )";
  return kernel_geglu;
}

const std::string &getGegluF16Kernel() {
  static std::string kernel_geglu_f16 =
    R"(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define GELU_COEF_A     0.044715f
#define GELU_QUICK_COEF -1.702f
#define SQRT_2_OVER_PI  0.79788456080286535587989211986876f
#define SQRT_2_INV      0.70710678118654752440084436210484f

kernel void kernel_geglu_f16(
    global char * src0,
    ulong  offset0,
    global char * src1,
    ulong  offset1,
    global char * dst,
    ulong  offsetd,
    ulong nb01,
    ulong nb11,
    int ne0,
    ulong nb1,
    int ne00_off,
    int ne10_off
) {
    src0 = (global char*)((global char*)src0 + offset0);
    src1 = (global char*)((global char*)src1 + offset1);
    dst  = (global char*)((global char*)dst  + offsetd);

    global half * src0_row = (global half *) ((global char *) src0 + get_group_id(0)*nb01) + ne00_off;
    global half * src1_row = (global half *) ((global char *) src1 + get_group_id(0)*nb11) + ne10_off;
    global half * dst_row  = (global half *) ((global char *) dst  + get_group_id(0)*nb1);

    for (int i0 = get_local_id(0); i0 < ne0; i0 += get_local_size(0)) {
        const half x0 = src0_row[i0];
        const half x1 = src1_row[i0];

        const half gelu = 0.5f*x0*(1.0f + tanh(SQRT_2_OVER_PI*x0*(1.0f + GELU_COEF_A*x0*x0)));

        dst_row[i0] = gelu*x1;
    }
}
    )";
  return kernel_geglu_f16;
}

const std::string &getRegluKernel() {
  static std::string kernel_reglu =
    R"(
#define GELU_COEF_A     0.044715f
#define GELU_QUICK_COEF -1.702f
#define SQRT_2_OVER_PI  0.79788456080286535587989211986876f
#define SQRT_2_INV      0.70710678118654752440084436210484f

kernel void kernel_reglu(
    global char * src0,
    ulong  offset0,
    global char * src1,
    ulong  offset1,
    global char * dst,
    ulong  offsetd,
    ulong nb01,
    ulong nb11,
    int ne0,
    ulong nb1,
    int ne00_off,
    int ne10_off
) {
    src0 = (global char*)((global char*)src0 + offset0);
    src1 = (global char*)((global char*)src1 + offset1);
    dst  = (global char*)((global char*)dst  + offsetd);

    global float * src0_row = (global float *) ((global char *) src0 + get_group_id(0)*nb01) + ne00_off;
    global float * src1_row = (global float *) ((global char *) src1 + get_group_id(0)*nb11) + ne10_off;
    global float * dst_row  = (global float *) ((global char *) dst  + get_group_id(0)*nb1);

    for (int i0 = get_local_id(0); i0 < ne0; i0 += get_local_size(0)) {
        const float x0 = src0_row[i0];
        const float x1 = src1_row[i0];

        dst_row[i0] = x0*x1*(x0 > 0.0f);
    }
}
    )";
  return kernel_reglu;
}

const std::string &getRegluF16Kernel() {
  static std::string kernel_reglu_f16 =
    R"(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define GELU_COEF_A     0.044715f
#define GELU_QUICK_COEF -1.702f
#define SQRT_2_OVER_PI  0.79788456080286535587989211986876f
#define SQRT_2_INV      0.70710678118654752440084436210484f

kernel void kernel_reglu_f16(
    global char * src0,
    ulong  offset0,
    global char * src1,
    ulong  offset1,
    global char * dst,
    ulong  offsetd,
    ulong nb01,
    ulong nb11,
    int ne0,
    ulong nb1,
    int ne00_off,
    int ne10_off
) {
    src0 = (global char*)((global char*)src0 + offset0);
    src1 = (global char*)((global char*)src1 + offset1);
    dst  = (global char*)((global char*)dst  + offsetd);

    global half * src0_row = (global half *) ((global char *) src0 + get_group_id(0)*nb01) + ne00_off;
    global half * src1_row = (global half *) ((global char *) src1 + get_group_id(0)*nb11) + ne10_off;
    global half * dst_row  = (global half *) ((global char *) dst  + get_group_id(0)*nb1);

    for (int i0 = get_local_id(0); i0 < ne0; i0 += get_local_size(0)) {
        const half x0 = src0_row[i0];
        const half x1 = src1_row[i0];

        dst_row[i0] = x0*x1*(x0 > 0.0f);
    }
}
    )";
  return kernel_reglu_f16;
}

const std::string &getSwigluKernel() {
  static std::string kernel_swiglu =
    R"(
#define GELU_COEF_A     0.044715f
#define GELU_QUICK_COEF -1.702f
#define SQRT_2_OVER_PI  0.79788456080286535587989211986876f
#define SQRT_2_INV      0.70710678118654752440084436210484f

kernel void kernel_swiglu(
    global char * src0,
    ulong  offset0,
    global char * src1,
    ulong  offset1,
    global char * dst,
    ulong  offsetd,
    ulong nb01,
    ulong nb11,
    int ne0,
    ulong nb1,
    int ne00_off,
    int ne10_off
) {
    src0 = (global char*)((global char*)src0 + offset0);
    src1 = (global char*)((global char*)src1 + offset1);
    dst  = (global char*)((global char*)dst  + offsetd);

    global float * src0_row = (global float *) ((global char *) src0 + get_group_id(0)*nb01) + ne00_off;
    global float * src1_row = (global float *) ((global char *) src1 + get_group_id(0)*nb11) + ne10_off;
    global float * dst_row  = (global float *) ((global char *) dst  + get_group_id(0)*nb1);

    for (int i0 = get_local_id(0); i0 < ne0; i0 += get_local_size(0)) {
        const float x0 = src0_row[i0];
        const float x1 = src1_row[i0];

        const float silu = x0 / (1.0f + exp(-x0));

        dst_row[i0] = silu*x1;
    }
}
    )";
  return kernel_swiglu;
}

const std::string &getSwigluF16Kernel() {
  static std::string kernel_swiglu_f16 =
    R"(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define GELU_COEF_A     0.044715f
#define GELU_QUICK_COEF -1.702f
#define SQRT_2_OVER_PI  0.79788456080286535587989211986876f
#define SQRT_2_INV      0.70710678118654752440084436210484f

kernel void kernel_swiglu_f16(
    global char * src0,
    ulong  offset0,
    global char * src1,
    ulong  offset1,
    global char * dst,
    ulong  offsetd,
    ulong nb01,
    ulong nb11,
    int ne0,
    ulong nb1,
    int ne00_off,
    int ne10_off
) {
    src0 = (global char*)((global char*)src0 + offset0);
    src1 = (global char*)((global char*)src1 + offset1);
    dst  = (global char*)((global char*)dst  + offsetd);

    global half * src0_row = (global half *) ((global char *) src0 + get_group_id(0)*nb01) + ne00_off;
    global half * src1_row = (global half *) ((global char *) src1 + get_group_id(0)*nb11) + ne10_off;
    global half * dst_row  = (global half *) ((global char *) dst  + get_group_id(0)*nb1);

    for (int i0 = get_local_id(0); i0 < ne0; i0 += get_local_size(0)) {
        const half x0 = src0_row[i0];
        const half x1 = src1_row[i0];

        const half silu = x0 / (1.0f + exp(-x0));

        dst_row[i0] = silu*x1;
    }
}
    )";
  return kernel_swiglu_f16;
}

const std::string &getGegluErfKernel() {
  static std::string kernel_geglu_erf =
    R"(
#define GELU_COEF_A     0.044715f
#define GELU_QUICK_COEF -1.702f
#define SQRT_2_OVER_PI  0.79788456080286535587989211986876f
#define SQRT_2_INV      0.70710678118654752440084436210484f

kernel void kernel_geglu_erf(
    global char * src0,
    ulong  offset0,
    global char * src1,
    ulong  offset1,
    global char * dst,
    ulong  offsetd,
    ulong nb01,
    ulong nb11,
    int ne0,
    ulong nb1,
    int ne00_off,
    int ne10_off
) {
    src0 = (global char*)((global char*)src0 + offset0);
    src1 = (global char*)((global char*)src1 + offset1);
    dst  = (global char*)((global char*)dst  + offsetd);

    global float * src0_row = (global float *) ((global char *) src0 + get_group_id(0)*nb01) + ne00_off;
    global float * src1_row = (global float *) ((global char *) src1 + get_group_id(0)*nb11) + ne10_off;
    global float * dst_row  = (global float *) ((global char *) dst  + get_group_id(0)*nb1);

    for (int i0 = get_local_id(0); i0 < ne0; i0 += get_local_size(0)) {
        const float x0 = src0_row[i0];
        const float x1 = src1_row[i0];

        const float gelu_erf = 0.5f*x0*(1.0f + erf(x0*SQRT_2_INV));

        dst_row[i0] = gelu_erf*x1;
    }
}
    )";
  return kernel_geglu_erf;
}

const std::string &getGegluErfF16Kernel() {
  static std::string kernel_geglu_erf_f16 =
    R"(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define GELU_COEF_A     0.044715f
#define GELU_QUICK_COEF -1.702f
#define SQRT_2_OVER_PI  0.79788456080286535587989211986876f
#define SQRT_2_INV      0.70710678118654752440084436210484f

kernel void kernel_geglu_erf_f16(
    global char * src0,
    ulong  offset0,
    global char * src1,
    ulong  offset1,
    global char * dst,
    ulong  offsetd,
    ulong nb01,
    ulong nb11,
    int ne0,
    ulong nb1,
    int ne00_off,
    int ne10_off
) {
    src0 = (global char*)((global char*)src0 + offset0);
    src1 = (global char*)((global char*)src1 + offset1);
    dst  = (global char*)((global char*)dst  + offsetd);

    global half * src0_row = (global half *) ((global char *) src0 + get_group_id(0)*nb01) + ne00_off;
    global half * src1_row = (global half *) ((global char *) src1 + get_group_id(0)*nb11) + ne10_off;
    global half * dst_row  = (global half *) ((global char *) dst  + get_group_id(0)*nb1);

    for (int i0 = get_local_id(0); i0 < ne0; i0 += get_local_size(0)) {
        const half x0 = src0_row[i0];
        const half x1 = src1_row[i0];

        const half gelu_erf = 0.5f*x0*(1.0f + erf(x0*SQRT_2_INV));

        dst_row[i0] = gelu_erf*x1;
    }
}
    )";
  return kernel_geglu_erf_f16;
}

const std::string &getGegluQuickKernel() {
  static std::string kernel_geglu_quick =
    R"(
#define GELU_COEF_A     0.044715f
#define GELU_QUICK_COEF -1.702f
#define SQRT_2_OVER_PI  0.79788456080286535587989211986876f
#define SQRT_2_INV      0.70710678118654752440084436210484f

kernel void kernel_geglu_quick(
    global char * src0,
    ulong  offset0,
    global char * src1,
    ulong  offset1,
    global char * dst,
    ulong  offsetd,
    ulong nb01,
    ulong nb11,
    int ne0,
    ulong nb1,
    int ne00_off,
    int ne10_off
) {
    src0 = (global char*)((global char*)src0 + offset0);
    src1 = (global char*)((global char*)src1 + offset1);
    dst  = (global char*)((global char*)dst  + offsetd);

    global float * src0_row = (global float *) ((global char *) src0 + get_group_id(0)*nb01) + ne00_off;
    global float * src1_row = (global float *) ((global char *) src1 + get_group_id(0)*nb11) + ne10_off;
    global float * dst_row  = (global float *) ((global char *) dst  + get_group_id(0)*nb1);

    for (int i0 = get_local_id(0); i0 < ne0; i0 += get_local_size(0)) {
        const float x0 = src0_row[i0];
        const float x1 = src1_row[i0];

        const float gelu_quick = x0*(1.0f/(1.0f + exp(GELU_QUICK_COEF*x0)));

        dst_row[i0] = gelu_quick*x1;
    }
}
    )";
  return kernel_geglu_quick;
}

const std::string &getGegluQuickF16Kernel() {
  static std::string kernel_geglu_quick_f16 =
    R"(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define GELU_COEF_A     0.044715f
#define GELU_QUICK_COEF -1.702f
#define SQRT_2_OVER_PI  0.79788456080286535587989211986876f
#define SQRT_2_INV      0.70710678118654752440084436210484f

kernel void kernel_geglu_quick_f16(
    global char * src0,
    ulong  offset0,
    global char * src1,
    ulong  offset1,
    global char * dst,
    ulong  offsetd,
    ulong nb01,
    ulong nb11,
    int ne0,
    ulong nb1,
    int ne00_off,
    int ne10_off
) {
    src0 = (global char*)((global char*)src0 + offset0);
    src1 = (global char*)((global char*)src1 + offset1);
    dst  = (global char*)((global char*)dst  + offsetd);

    global half * src0_row = (global half *) ((global char *) src0 + get_group_id(0)*nb01) + ne00_off;
    global half * src1_row = (global half *) ((global char *) src1 + get_group_id(0)*nb11) + ne10_off;
    global half * dst_row  = (global half *) ((global char *) dst  + get_group_id(0)*nb1);

    for (int i0 = get_local_id(0); i0 < ne0; i0 += get_local_size(0)) {
        const half x0 = src0_row[i0];
        const half x1 = src1_row[i0];

        const half gelu_quick = x0*(1.0f/(1.0f + exp(GELU_QUICK_COEF*x0)));

        dst_row[i0] = gelu_quick*x1;
    }
}
    )";
  return kernel_geglu_quick_f16;
}

// add.cl
const std::string &getAddKernel() {
  static std::string kernel_add =
    R"(
// general-purpose kernel for addition of two tensors
// pros: works for non-contiguous tensors, supports broadcast across dims 1, 2 and 3
// cons: not very efficient
kernel void kernel_add(
        global char * src0,
        ulong  offset0,
        global char * src1,
        ulong  offset1,
        global char * dst,
        ulong  offsetd,
        int   ne00,
        int   ne01,
        int   ne02,
        int   ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int   ne10,
        int   ne11,
        int   ne12,
        int   ne13,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        int   ne0,
        int   ne1,
        int   ne2,
        int   ne3,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst = dst + offsetd;

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0);

    int i13 = i03 % ne13;
    int i12 = i02 % ne12;
    int i11 = i01 % ne11;

    global char * src0_ptr = src0 + i03*nb03 + i02*nb02 + i01*nb01;
    global char * src1_ptr = src1 + i13*nb13 + i12*nb12 + i11*nb11;
    global char * dst_ptr  = dst  + i03*nb3  + i02*nb2  + i01*nb1;

    for (int i0 = get_local_id(0); i0 < ne0; i0 += get_local_size(0)) {
        const int i10 = i0 % ne10;
        *((global float *)(dst_ptr + i0*nb0)) = *((global float *)(src0_ptr + i0*nb00)) + *((global float *)(src1_ptr + i10*nb10));
    }
}
    )";
  return kernel_add;
}

const std::string &getAddRowKernel() {
  static std::string kernel_add_row =
    R"(
// assumption: src1 is a row
// broadcast src1 into src0
kernel void kernel_add_row(
        global float4 * src0,
        ulong  offset0,
        global float4 * src1,
        ulong  offset1,
        global float4 * dst,
        ulong  offsetd,
        int ne
) {
    src0 = (global float4*)((global char*)src0 + offset0);
    src1 = (global float4*)((global char*)src1 + offset1);
    dst = (global float4*)((global char*)dst + offsetd);

    // This performs better than using %.
    uint gid = get_global_id(0);
    uint idx1 = gid - (gid/ne)*ne; // get_global_id(0) % ne
    dst[gid] = src0[gid] + src1[idx1];
}  
    )";
  return kernel_add_row;
}

const std::string &getAddF16Kernel() {
  static std::string kernel_add_f16 =
    R"(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

kernel void kernel_add_f16(
        global char * src0,
        ulong  offset0,
        global char * src1,
        ulong  offset1,
        global char * dst,
        ulong  offsetd,
        int   ne00,
        int   ne01,
        int   ne02,
        int   ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int   ne10,
        int   ne11,
        int   ne12,
        int   ne13,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        int   ne0,
        int   ne1,
        int   ne2,
        int   ne3,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst = dst + offsetd;

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0);

    int i13 = i03 % ne13;
    int i12 = i02 % ne12;
    int i11 = i01 % ne11;

    global char * src0_ptr = src0 + i03*nb03 + i02*nb02 + i01*nb01;
    global char * src1_ptr = src1 + i13*nb13 + i12*nb12 + i11*nb11;
    global char * dst_ptr  = dst  + i03*nb3  + i02*nb2  + i01*nb1;

    for (int i0 = get_local_id(0); i0 < ne0; i0 += get_local_size(0)) {
        const int i10 = i0 % ne10;
        *((global half *)(dst_ptr + i0*nb0)) = *((global half *)(src0_ptr + i0*nb00)) + *((global half *)(src1_ptr + i10*nb10));
    }
}
    )";
  return kernel_add_f16;
}

const std::string &getAddRowF16Kernel() {
  static std::string kernel_add_row_f16 =
    R"(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

kernel void kernel_add_row_f16(
        global half4 * src0,
        ulong  offset0,
        global half4 * src1,
        ulong  offset1,
        global half4 * dst,
        ulong  offsetd,
        int ne
) {
    src0 = (global half4*)((global char*)src0 + offset0);
    src1 = (global half4*)((global char*)src1 + offset1);
    dst = (global half4*)((global char*)dst + offsetd);

    // This performs better than using %.
    uint gid = get_global_id(0);
    uint idx1 = gid - (gid/ne)*ne; // get_global_id(0) % ne
    dst[gid] = src0[gid] + src1[idx1];
}
    )";
  return kernel_add_row_f16;
}

// rms_norm.cl
const std::string &getRmsNormKernel() {
  static std::string kernel_rms_norm =
    R"(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_intel_subgroups
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#else
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif

#ifdef cl_intel_required_subgroup_size
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#define INTEL_GPU 1
#define REQD_SUBGROUP_SIZE_16 __attribute__((intel_reqd_sub_group_size(16)))
#define REQD_SUBGROUP_SIZE_32 __attribute__((intel_reqd_sub_group_size(32)))
#elif defined(cl_qcom_reqd_sub_group_size)
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64  __attribute__((qcom_reqd_sub_group_size("half")))
#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
#endif

//------------------------------------------------------------------------------
// rms_norm
//------------------------------------------------------------------------------
// This kernel depends on subgroup size.
#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_32
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_rms_norm(
        global void * src0,
        ulong offset0,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        float eps,
        local float * sum // Note, the size depends on number of subgroups
) {
    src0 = (global void*)((global char*)src0 + offset0);
    dst = (global float*)((global char*)dst + offsetd);

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0);

    global float4 * x = (global float4 *) ((global char *) src0 + i03*nb03 + i02*nb02 + i01*nb01);
    global float * x_scalar = (global float *) x;
    float4 sumf = 0;
    float all_sum = 0;

    // parallel sum
    for (int i00 = get_local_id(0); i00 < ne00/4; i00 += get_local_size(0)) {
        sumf += x[i00] * x[i00];
    }
    all_sum = sumf.s0 + sumf.s1 + sumf.s2 + sumf.s3;
    all_sum = sub_group_reduce_add(all_sum);
    if (get_sub_group_local_id() == 0) {
        sum[get_sub_group_id()] = all_sum;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    // broadcast
    for (uint i = get_local_size(0) / get_max_sub_group_size() / 2; i > 0; i /= 2) {
       if (get_local_id(0) < i) {
           sum[get_local_id(0)] += sum[get_local_id(0) + i];
       }
    }
    if (get_local_id(0) == 0) {
        for (int i = 4 * (ne00 / 4); i < ne00; i++) {
            sum[0] += x_scalar[i];
        }
        sum[0] /= ne00;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    const float mean  = sum[0];
    const float scale = 1.0f/sqrt(mean + eps);

    global float4 * y = (global float4 *) (dst + i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00);
    global float * y_scalar = (global float *) y;
    for (int i00 = get_local_id(0); i00 < ne00/4; i00 += get_local_size(0)) {
        y[i00] = x[i00] * scale;
    }
    if (get_local_id(0) == 0) {
        for (int i00 = 4 * (ne00 / 4); i00 < ne00; i00++) {
            y_scalar[i00] = x_scalar[i00] * scale;
        }
    }
}
    )";
  return kernel_rms_norm;
}

const std::string &getRmsNormMulKernel() {
  static std::string kernel_rms_norm_mul =
    R"(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_intel_subgroups
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#else
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif

#ifdef cl_intel_required_subgroup_size
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#define INTEL_GPU 1
#define REQD_SUBGROUP_SIZE_16 __attribute__((intel_reqd_sub_group_size(16)))
#define REQD_SUBGROUP_SIZE_32 __attribute__((intel_reqd_sub_group_size(32)))
#elif defined(cl_qcom_reqd_sub_group_size)
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64  __attribute__((qcom_reqd_sub_group_size("half")))
#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
#endif

//------------------------------------------------------------------------------
// rms_norm_mul
//------------------------------------------------------------------------------
#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_32
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_rms_norm_mul(
        global char * src0,
        ulong offset0,
        global char * src1,
        ulong offset1,
        global char * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        int ne11,
        int ne12,
        int ne13,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        ulong nb1,
        ulong nb2,
        ulong nb3,
        float eps,
        local float * sum
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst  = dst  + offsetd;

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0);

    global float4 * x = (global float4 *) (src0 + i03*nb03 + i02*nb02 + i01*nb01);
    global float4 * f = (global float4 *) (src1 + (i03%ne13)*nb13 + (i02%ne12)*nb12 + (i01%ne11)*nb11);

    float sumf = 0;

    // parallel sum
    for (int i00 = get_local_id(0); i00 < ne00/4; i00 += get_local_size(0)) {
        sumf += dot(x[i00], x[i00]);
    }
    sumf = sub_group_reduce_add(sumf);
    if (get_sub_group_local_id() == 0) {
        sum[get_sub_group_id()] = sumf;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint i = get_local_size(0) / get_max_sub_group_size() / 2; i > 0; i /= 2) {
       if (get_local_id(0) < i) {
           sum[get_local_id(0)] += sum[get_local_id(0) + i];
       }
    }
    if (get_local_id(0) == 0) {
        sum[0] /= ne00;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    float mean  = sum[0];
    float scale = 1.0f/sqrt(mean + eps);

    global float4 * y = (global float4 *) (dst + i03*nb3 + i02*nb2 + i01*nb1);
    for (int i00 = get_local_id(0); i00 < ne00/4; i00 += get_local_size(0)) {
        y[i00] = (x[i00] * scale) * f[i00%(ne10/4)];
    }
}
    )";
  return kernel_rms_norm_mul;
}

} // namespace nntrainer
