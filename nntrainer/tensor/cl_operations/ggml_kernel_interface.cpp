// Original code :
// https://github.com/ggml-org/ggml/blob/master/src/ggml-opencl/kernels/rms_norm.cl
// Original code :
// https://github.com/ggml-org/ggml/blob/master/src/ggml-opencl/kernels/add.cl
// Original code :
// https://github.com/ggml-org/ggml/blob/master/src/ggml-opencl/kernels/glu.cl

#include "ggml_kernel_interface.h"

#include "ggml_kernels.h"

#include <cassert>

// FROM : blas_kernels.cpp:49
//  int ne00 = M; // number of rows in matrix X
//  int ne01 = N; // number of columns in matrix X
//  int ne02 = 1; // number of channels in matrix X
//  int ne10 = M; // number of rows in vector A
//  int ne11 = 1; // number of columns in vector A
//  int ne12 = 1; // number of channels in vector A
//  int ne13 = 1; // number of channels in vector A (Need to check)
//  int ne0 = N;  // number of rows in output vector Y
//  int ne1 = 1;  // number of columns in output vector Y

//  ggml_type_size(FLOAT) = 4
//  ggml_blck_size(FLOAT) = 1

//     int64_t ne[GGML_MAX_DIMS]; // number of elements
//     size_t  nb[GGML_MAX_DIMS]; // stride in bytes:
//                                // nb[0] = ggml_type_size(type)
//                                // nb[1] = nb[0]   * (ne[0] /
//                                ggml_blck_size(type)) + padding
//                                // nb[i] = nb[i-1] * ne[i-1]

namespace nntrainer {
void ggml_swigluCl(const Tensor &src0, const Tensor &src1, Tensor &dst) {
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

  // IDK which should be width and which should be height
  unsigned long nb01 = sizeof(float) * src0.width();
  unsigned long nb11 = sizeof(float) * src1.width();

  int32_t ne0 = dst.width();

  unsigned long nb1 = sizeof(float) * dst.width();

  int32_t ne00_off = 1;
  int32_t ne10_off = 1;

  // [0]  [void *       ] - src0     [assuming SVM] [nntrainer - matAdata]
  // [1]  [unsigned long] - offset0                 [offset into src0]
  // [2]  [void *       ] - src1     [assuming SVM] [nntrainer - vecXdata]
  // [3]  [unsigned long] - offset1                 [offset into src1]
  // [4]  [void *       ] - dst      [assuming SVM] [nntraienr - vecYdata]
  // [5]  [unsigned long] - offsetd                 [offset into dst]
  // [6]  [unsigned long] - nb01                    [src0 stride in bytes in
  // dimension 1 of ggml nb[GGML_MAX_DIMS]] [7]  [unsigned long] - nb11 [src1
  // stride in bytes in dimension 1 of ggml nb[GGML_MAX_DIMS]] [8]  [int32_t ] -
  // ne0                     [dst number of elements in dimension 0] [9]
  // [unsigned long] - nb1                     [dst stride in bytes in dimension
  // 1] [10] [int32_t      ] - ne00_off                [idk - lets assume 1 and
  // hope for the best] [11] [int32_t      ] - ne10_off                [idk -
  // lets assume 1 and hope for the best]

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

  //
  //// TODO : ...
  //

  // input0
  const auto ne00 = src0.height();
  const auto ne01 = src0.width();
  const auto ne02 = src0.channel();
  const auto ne03 = 1;

  const auto nb00 = sizeof(float);
  const auto nb01 = 0;
  const auto nb02 = 0;
  const auto nb03 = 0;

  // input1
  const auto ne10 = src1.height();
  const auto ne11 = src1.width();
  const auto ne12 = src1.channel();
  const auto ne13 = 1;

  const auto nb10 = sizeof(float);
  const auto nb11 = 0;
  const auto nb12 = 0;
  const auto nb13 = 0;

  // output
  const auto ne0 = dst.height();
  const auto ne1 = dst.width();
  const auto ne2 = dst.channel();
  const auto ne3 = 1;

  const auto nb0 = sizeof(float);
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

  //
  //// TODO : ...
  //

  auto ne00 = src0.height();
  auto ne01 = src0.width();
  auto ne02 = src0.channel();
  auto ne03 = 1;

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

  //
  //// TODO : ...
  //

  // input0
  auto ne00 = src0.height();
  auto ne01 = src0.width();
  auto ne02 = src0.channel();
  auto ne03 = 1;

  auto nb01 = 0;
  auto nb02 = 0;
  auto nb03 = 0;

  // input1
  auto ne10 = src0.height();
  auto ne11 = src0.width();
  auto ne12 = src0.channel();
  auto ne13 = 1;

  auto nb11 = 0;
  auto nb12 = 0;
  auto nb13 = 0;

  // output
  auto nb1 = 0;
  auto nb2 = 0;
  auto nb3 = 0;

  auto eps = 0;

  ggml_rms_norm_mul_cl(data_src0, offset0, data_src1, offset1, data_dst,
                       offsetd, ne00, ne01, ne02, ne03, nb01, nb02, nb03, ne10,
                       ne11, ne12, ne13, nb11, nb12, nb13, nb1, nb2, nb3, eps);
}
} // namespace nntrainer
