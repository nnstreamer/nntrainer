// Original code :
// https://github.com/ggml-org/ggml/blob/master/src/ggml-opencl/kernels/rms_norm.cl
// Original code :
// https://github.com/ggml-org/ggml/blob/master/src/ggml-opencl/kernels/add.cl
// Original code :
// https://github.com/ggml-org/ggml/blob/master/src/ggml-opencl/kernels/glu.cl

#include "ggml_kernel_interface.h"

#include "ggml_kernels.h"

#include <cassert>

namespace nntrainer {
void ggml_swigluCl(const Tensor &src0, const Tensor &src1, Tensor &dst) {
  const auto src0_format = src0.getFormat();
  const auto src1_format = src1.getFormat();

  const auto src0_batch = src0.batch();
  const auto src0_width = src0.width();
  const auto src0_height = src0.height();
  const auto src0_channel = src0.channel();

  const auto src1_batch = src1.batch();
  const auto src1_width = src1.width();
  const auto src1_height = src1.height();
  const auto src1_channel = src1.channel();

  unsigned int src0_dim1 = UINT32_MAX;
  unsigned int src0_dim2 = UINT32_MAX;

  unsigned int src1_dim1 = UINT32_MAX;
  unsigned int src1_dim2 = UINT32_MAX;

  if (src0_format == Tformat::NHWC) {
    src0_dim1 = src0_batch * src0_height * src0_width;
    src0_dim2 = src0_channel;
    src1_dim1 = src1_batch * src1_height * src1_width;
    src1_dim2 = src1_channel;
  } else {
    src0_dim1 = src0_batch * src0_channel * src0_height;
    src0_dim2 = src0_width;
    src1_dim1 = src1_batch * src1_channel * src1_height;
    src1_dim2 = src1_width;
  }

  unsigned int M = UINT32_MAX;
  unsigned int N = UINT32_MAX;
  unsigned int K = UINT32_MAX;
  unsigned int lda = UINT32_MAX;
  unsigned int ldb = UINT32_MAX;
  unsigned int ldc = UINT32_MAX;

  float *data_src0 = src0.getData();
  float *data_src1 = src1.getData();

  float *data_dst = dst.getData();

  assert(data_src0 && "src0 tensor - invalid!");
  assert(data_src1 && "src1 tensor - invalid!");
  assert(data_dst && "dst tensor - invalid!");

  // Assuming no transpose
  // Assuming F32

  unsigned long offset0 = 0;
  unsigned long offset1 = 0;
  unsigned long offset = 0;

  unsigned long nb01 = 0;
  unsigned long nb11 = 0;

  int32_t ne0 = 0;

  unsigned long nb1 = 0;

  int32_t ne00_off = 0;
  int32_t ne10_off = 0;

  ggml_swiglu_cl(data_src0, offset0, data_src1, offset1, data_dst, offset, nb01,
                 nb11, ne0, nb1, ne00_off, ne10_off);
}

void ggml_addCl(const Tensor &src0, const Tensor &src1, Tensor &dst) {

  float *data_src0 = src0.getData();
  float *data_src1 = src1.getData();

  float *data_dst = dst.getData();

  const auto offset0 = 0;
  const auto offset1 = 0;
  const auto offsetd = 0;

  const auto ne00 = 0;
  const auto ne01 = 0;
  const auto ne02 = 0;
  const auto ne03 = 0;

  const auto nb00 = 0;
  const auto nb01 = 0;
  const auto nb02 = 0;
  const auto nb03 = 0;

  const auto ne10 = 0;
  const auto ne11 = 0;
  const auto ne12 = 0;
  const auto ne13 = 0;

  const auto nb10 = 0;
  const auto nb11 = 0;
  const auto nb12 = 0;
  const auto nb13 = 0;

  const auto ne0 = 0;
  const auto ne1 = 0;
  const auto ne2 = 0;
  const auto ne3 = 0;

  const auto nb0 = 0;
  const auto nb1 = 0;
  const auto nb2 = 0;
  const auto nb3 = 0;

  ggml_add_cl(data_src0, offset0, data_src1, offset1, data_dst, offsetd, ne00,
              ne01, ne02, ne03, nb00, nb01, nb02, nb03, ne10, ne11, ne12, ne13,
              nb10, nb11, nb12, nb13, ne0, ne1, ne2, ne3, nb0, nb1, nb2, nb3);
}

void ggml_rms_normCl(const Tensor &src0, Tensor &dst) {
  float *data_src0 = src0.getData();

  float *data_dst = dst.getData();

  auto offset0 = 0;
  auto offsetd = 0;

  auto ne00 = 0;
  auto ne01 = 0;
  auto ne02 = 0;
  auto ne03 = 0;

  auto nb01 = 0;
  auto nb02 = 0;
  auto nb03 = 0;

  auto eps = 0;

  ggml_rms_norm_cl(data_src0, offset0, data_dst, offsetd, ne00, ne01, ne02,
                   ne03, nb01, nb02, nb03, eps);
}

void ggml_rms_norm_mulCl(const Tensor &src0, Tensor const &src1, Tensor &dst) {
  float *data_src0 = src0.getData();
  float *data_src1 = src1.getData();

  float *data_dst = dst.getData();

  auto offset0 = 0;
  auto offset1 = 0;
  auto offsetd = 0;

  auto ne00 = 0;
  auto ne01 = 0;
  auto ne02 = 0;
  auto ne03 = 0;

  auto nb01 = 0;
  auto nb02 = 0;
  auto nb03 = 0;

  auto ne10 = 0;
  auto ne11 = 0;
  auto ne12 = 0;
  auto ne13 = 0;
  auto nb11 = 0;
  auto nb12 = 0;
  auto nb13 = 0;
  auto nb1 = 0;
  auto nb2 = 0;
  auto nb3 = 0;

  auto eps = 0;

  ggml_rms_norm_mul_cl(data_src0, offset0, data_src1, offset1, data_dst,
                       offsetd, ne00, ne01, ne02, ne03, nb01, nb02, nb03, ne10,
                       ne11, ne12, ne13, nb11, nb12, nb13, nb1, nb2, nb3, eps);
}
} // namespace nntrainer
