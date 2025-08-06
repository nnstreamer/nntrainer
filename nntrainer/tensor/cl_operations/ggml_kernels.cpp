// Original code :
// https://github.com/ggml-org/ggml/blob/master/src/ggml-opencl/kernels/rms_norm.cl
// Original code :
// https://github.com/ggml-org/ggml/blob/master/src/ggml-opencl/kernels/add.cl
// Original code :
// https://github.com/ggml-org/ggml/blob/master/src/ggml-opencl/kernels/glu.cl

#include "ggml_kernels.h"

#include "ggml_kernel_strings.h"

namespace nntrainer {
// // // //
// PRIVATE
// // // //

// [0]  [void *       ] - src0     [assuming SVM]
// [1]  [unsigned long] - offset0
// [2]  [void *       ] - src1     [assuming SVM]
// [3]  [unsigned long] - offset1
// [4]  [void *       ] - dst      [assuming SVM]
// [5]  [unsigned long] - offsetd
// [6]  [unsigned long] - nb01
// [7]  [unsigned long] - nb11
// [8]  [int32_t      ] - ne0
// [9]  [unsigned long] - nb1
// [10] [int32_t      ] - ne00_off
// [11] [int32_t      ] - ne10_off
//
// TODO : Use nntrainer convention for M, N, K and calculate ggml parameters
//
static inline void ggml_swiglu_cl_kernel(
  ClContext &cl_context, ClContext::SharedPtrClKernel kernel, void *src0,
  unsigned long offset0, void *src1, unsigned long offset1, void *dst,
  unsigned long offsetd, unsigned long nb01, unsigned long nb11, int32_t ne0,
  unsigned long nb1, int32_t ne00_off, int32_t ne10_off) {
  bool unmap_status = true;
  unmap_status &= cl_context.command_queue_inst_.enqueueSVMUnmap(src0);
  unmap_status &= cl_context.command_queue_inst_.enqueueSVMUnmap(src1);

  if (!unmap_status) {
    ml_loge("Failed to unmap input SVM arguments!");
    return;
  }

  bool svm_status = true;
  svm_status &= kernel->SetKernelSVMArguments(0, src0);
  svm_status &= kernel->SetKernelSVMArguments(2, src1);
  svm_status &= kernel->SetKernelSVMArguments(4, dst);

  if (!svm_status) {
    ml_loge("Failed to set SVM arguments!");
    return;
  }

  bool arg_status = true;
  arg_status &= kernel->SetKernelArguments(1, &offset0, sizeof(unsigned long));
  arg_status &= kernel->SetKernelArguments(3, &offset1, sizeof(unsigned long));
  arg_status &= kernel->SetKernelArguments(5, &offsetd, sizeof(unsigned long));
  arg_status &= kernel->SetKernelArguments(6, &nb01, sizeof(unsigned long));
  arg_status &= kernel->SetKernelArguments(7, &nb11, sizeof(unsigned long));
  arg_status &= kernel->SetKernelArguments(8, &ne0, sizeof(int32_t));
  arg_status &= kernel->SetKernelArguments(9, &nb1, sizeof(unsigned long));
  arg_status &= kernel->SetKernelArguments(10, &ne00_off, sizeof(int32_t));
  arg_status &= kernel->SetKernelArguments(11, &ne10_off, sizeof(int32_t));

  if (!arg_status) {
    ml_loge("Failed to set arguments!");
    return;
  }

  const int32_t wgc[3] = {1, 1, 1}; // TODO : ...
  const int32_t wgs[3] = {1, 1, 1}; // TODO : ...

  if (!opencl::CommandQueueManager::Global().DispatchCommand(kernel, wgc,
                                                             wgs)) {
    ml_loge("Failed to dispatch!");
    return;
  }

  bool map_status = true;
  map_status &= cl_context.command_queue_inst_.enqueueSVMUnmap(dst);

  if (!map_status) {
    ml_loge("Failed to map output!");
    return;
  }
}

// [0]  [global char *] - src0     [assuming SVM]
// [1]  [ulong        ] - offset0
// [2]  [global char *] - src1     [assuming SVM]
// [3]  [ulong        ] - offset1
// [4]  [global char *] - dst      [assuming SVM]
// [5]  [ulong        ] - offsetd
// [6]  [int          ] - ne00
// [7]  [int          ] - ne01
// [8]  [int          ] - ne02
// [9]  [int          ] - ne03
// [10] [ulong        ] - nb00
// [11] [ulong        ] - nb01
// [12] [ulong        ] - nb02
// [13] [ulong        ] - nb03
// [14] [int          ] - ne10
// [15] [int          ] - ne11
// [16] [int          ] - ne12
// [17] [int          ] - ne13
// [18] [ulong        ] - nb10
// [19] [ulong        ] - nb11
// [20] [ulong        ] - nb12
// [21] [ulong        ] - nb13
// [22] [int          ] - ne0
// [23] [int          ] - ne1
// [24] [int          ] - ne2
// [25] [int          ] - ne3
// [26] [ulong        ] - nb0
// [27] [ulong        ] - nb1
// [28] [ulong        ] - nb2
// [29] [ulong        ] - nb3
//
// TODO : Use nntrainer convention for M, N, K and calculate ggml parameters
//
static inline void ggml_add_cl_kernel(
  ClContext &cl_context, ClContext::SharedPtrClKernel kernel, void *src0,
  unsigned long offset0, void *src1, unsigned long offset1, void *dst,
  unsigned long offsetd, int32_t ne00, int32_t ne01, int32_t ne02, int32_t ne03,
  unsigned long nb00, unsigned long nb01, unsigned long nb02,
  unsigned long nb03, int32_t ne10, int32_t ne11, int32_t ne12, int32_t ne13,
  unsigned long nb10, unsigned long nb11, unsigned long nb12,
  unsigned long nb13, int32_t ne0, int32_t ne1, int32_t ne2, int32_t ne3,
  unsigned long nb0, unsigned long nb1, unsigned long nb2, unsigned long nb3) {
  bool unmap_status = true;
  unmap_status &= cl_context.command_queue_inst_.enqueueSVMUnmap(src0);
  unmap_status &= cl_context.command_queue_inst_.enqueueSVMUnmap(src1);

  if (!unmap_status) {
    ml_loge("Failed to unmap input SVM arguments!");
    return;
  }

  bool svm_status = true;
  svm_status &= kernel->SetKernelSVMArguments(0, src0);
  svm_status &= kernel->SetKernelSVMArguments(2, src1);
  svm_status &= kernel->SetKernelSVMArguments(4, dst);

  if (!svm_status) {
    ml_loge("Failed to set SVM arguments!");
    return;
  }

  bool arg_status = true;
  arg_status &= kernel->SetKernelArguments(1, &offset0, sizeof(unsigned long));
  arg_status &= kernel->SetKernelArguments(3, &offset1, sizeof(unsigned long));
  arg_status &= kernel->SetKernelArguments(5, &offsetd, sizeof(unsigned long));
  arg_status &= kernel->SetKernelArguments(6, &ne00, sizeof(int32_t));
  arg_status &= kernel->SetKernelArguments(7, &ne01, sizeof(int32_t));
  arg_status &= kernel->SetKernelArguments(8, &ne02, sizeof(int32_t));
  arg_status &= kernel->SetKernelArguments(9, &ne03, sizeof(int32_t));
  arg_status &= kernel->SetKernelArguments(10, &nb00, sizeof(unsigned long));
  arg_status &= kernel->SetKernelArguments(11, &nb01, sizeof(unsigned long));
  arg_status &= kernel->SetKernelArguments(12, &nb02, sizeof(unsigned long));
  arg_status &= kernel->SetKernelArguments(13, &nb03, sizeof(unsigned long));
  arg_status &= kernel->SetKernelArguments(14, &ne10, sizeof(int32_t));
  arg_status &= kernel->SetKernelArguments(15, &ne11, sizeof(int32_t));
  arg_status &= kernel->SetKernelArguments(16, &ne12, sizeof(int32_t));
  arg_status &= kernel->SetKernelArguments(17, &ne13, sizeof(int32_t));
  arg_status &= kernel->SetKernelArguments(18, &nb10, sizeof(unsigned long));
  arg_status &= kernel->SetKernelArguments(19, &nb11, sizeof(unsigned long));
  arg_status &= kernel->SetKernelArguments(20, &nb12, sizeof(unsigned long));
  arg_status &= kernel->SetKernelArguments(21, &nb13, sizeof(unsigned long));
  arg_status &= kernel->SetKernelArguments(22, &ne0, sizeof(int32_t));
  arg_status &= kernel->SetKernelArguments(23, &ne1, sizeof(int32_t));
  arg_status &= kernel->SetKernelArguments(24, &ne2, sizeof(int32_t));
  arg_status &= kernel->SetKernelArguments(25, &ne3, sizeof(int32_t));
  arg_status &= kernel->SetKernelArguments(26, &nb0, sizeof(unsigned long));
  arg_status &= kernel->SetKernelArguments(27, &nb1, sizeof(unsigned long));
  arg_status &= kernel->SetKernelArguments(28, &nb2, sizeof(unsigned long));
  arg_status &= kernel->SetKernelArguments(29, &nb3, sizeof(unsigned long));

  if (!arg_status) {
    ml_loge("Failed to set arguments!");
    return;
  }

  const int32_t wgc[3] = {1, 1, 1}; // TODO : ...
  const int32_t wgs[3] = {1, 1, 1}; // TODO : ...

  if (!opencl::CommandQueueManager::Global().DispatchCommand(kernel, wgc,
                                                             wgs)) {
    ml_loge("Failed to dispatch!");
    return;
  }

  bool map_status = true;
  map_status &= cl_context.command_queue_inst_.enqueueSVMUnmap(dst);

  if (!map_status) {
    ml_loge("Failed to map output!");
    return;
  }
}

// // // //
// PUBLIC
// // // //

void ggml_swiglu_cl(void *src0, unsigned long offset0, void *src1,
                    unsigned long offset1, void *dst, unsigned long offsetd,
                    unsigned long nb01, unsigned long nb11, int32_t ne0,
                    unsigned long nb1, int32_t ne00_off, int32_t ne10_off) {
  Engine &engine = Engine::Global();
  Context &context = *engine.getRegisteredContext("gpu");

  ClContext &cl_context = static_cast<ClContext &>(context);

  ClContext::SharedPtrClKernel kernel =
    cl_context.registerClKernel(getSwigluKernel(), "kernel_swiglu");

  if (!kernel) {
    ml_loge("Kernel - invalid!");
    return;
  }

  ggml_swiglu_cl_kernel(cl_context, kernel, src0, offset0, src1, offset1, dst,
                        offsetd, nb01, nb11, ne0, nb1, ne00_off, ne10_off);
}

void ggml_swiglu_f16_cl(void *src0, unsigned long offset0, void *src1,
                        unsigned long offset1, void *dst, unsigned long offsetd,
                        unsigned long nb01, unsigned long nb11, int32_t ne0,
                        unsigned long nb1, int32_t ne00_off, int32_t ne10_off) {
  Engine &engine = Engine::Global();
  Context &context = *engine.getRegisteredContext("gpu");

  ClContext &cl_context = static_cast<ClContext &>(context);

  ClContext::SharedPtrClKernel kernel =
    cl_context.registerClKernel(getSwigluKernel(), "kernel_swiglu_f16");

  if (!kernel) {
    ml_loge("Kernel - invalid!");
    return;
  }

  ggml_swiglu_cl_kernel(cl_context, kernel, src0, offset0, src1, offset1, dst,
                        offsetd, nb01, nb11, ne0, nb1, ne00_off, ne10_off);
}

void ggml_add_cl(void *src0, unsigned long offset0, void *src1,
                 unsigned long offset1, void *dst, unsigned long offsetd,
                 int32_t ne00, int32_t ne01, int32_t ne02, int32_t ne03,
                 unsigned long nb00, unsigned long nb01, unsigned long nb02,
                 unsigned long nb03, int32_t ne10, int32_t ne11, int32_t ne12,
                 int32_t ne13, unsigned long nb10, unsigned long nb11,
                 unsigned long nb12, unsigned long nb13, int32_t ne0,
                 int32_t ne1, int32_t ne2, int32_t ne3, unsigned long nb0,
                 unsigned long nb1, unsigned long nb2, unsigned long nb3) {
  Engine &engine = Engine::Global();
  Context &context = *engine.getRegisteredContext("gpu");

  ClContext &cl_context = static_cast<ClContext &>(context);

  ClContext::SharedPtrClKernel kernel =
    cl_context.registerClKernel(getAddKernel(), "kernel_add");

  if (!kernel) {
    ml_loge("Kernel - invalid!");
    return;
  }

  ggml_add_cl_kernel(cl_context, kernel, src0, offset0, src1, offset1, dst,
                     offsetd, ne00, ne01, ne02, ne03, nb00, nb01, nb02, nb03,
                     ne10, ne11, ne12, ne13, nb10, nb11, nb12, nb13, ne0, ne1,
                     ne2, ne3, nb0, nb1, nb2, nb3);
}

void ggml_add_f16_cl(void *src0, unsigned long offset0, void *src1,
                     unsigned long offset1, void *dst, unsigned long offsetd,
                     int32_t ne00, int32_t ne01, int32_t ne02, int32_t ne03,
                     unsigned long nb00, unsigned long nb01, unsigned long nb02,
                     unsigned long nb03, int32_t ne10, int32_t ne11,
                     int32_t ne12, int32_t ne13, unsigned long nb10,
                     unsigned long nb11, unsigned long nb12, unsigned long nb13,
                     int32_t ne0, int32_t ne1, int32_t ne2, int32_t ne3,
                     unsigned long nb0, unsigned long nb1, unsigned long nb2,
                     unsigned long nb3) {
  Engine &engine = Engine::Global();
  Context &context = *engine.getRegisteredContext("gpu");

  ClContext &cl_context = static_cast<ClContext &>(context);

  ClContext::SharedPtrClKernel kernel =
    cl_context.registerClKernel(getAddF16Kernel(), "kernel_add_f16");

  if (!kernel) {
    ml_loge("Kernel - invalid!");
    return;
  }

  ggml_add_cl_kernel(cl_context, kernel, src0, offset0, src1, offset1, dst,
                     offsetd, ne00, ne01, ne02, ne03, nb00, nb01, nb02, nb03,
                     ne10, ne11, ne12, ne13, nb10, nb11, nb12, nb13, ne0, ne1,
                     ne2, ne3, nb0, nb1, nb2, nb3);
}

// [0]  [global void * ]   src0     - [assuming SVM]
// [1]  [ulong         ]   offset0
// [2]  [global float *]   dst      - [assuming SVM]
// [3]  [ulong         ]   offsetd
// [4]  [int           ]   ne00
// [5]  [int           ]   ne01
// [6]  [int           ]   ne02
// [7]  [int           ]   ne03
// [8]  [ulong         ]   nb01
// [9]  [ulong         ]   nb02
// [10] [ulong         ]   nb03
// [11] [float         ]   eps
// [12] [local float * ]   su       - [assuming local]
//
// TODO : Use nntrainer convention for M, N, K and calculate ggml parameters
//
void ggml_rms_norm_cl(void *src0, unsigned long offset0, void *dst,
                      unsigned long offsetd, int32_t ne00, int32_t ne01,
                      int32_t ne02, int32_t ne03, unsigned long nb01,
                      unsigned long nb02, unsigned long nb03, float eps) {
  Engine &engine = Engine::Global();
  Context &context = *engine.getRegisteredContext("gpu");

  ClContext &cl_context = static_cast<ClContext &>(context);

  ClContext::SharedPtrClKernel kernel =
    cl_context.registerClKernel(getRmsNormKernel(), "kernel_rms_norm");

  if (!kernel) {
    ml_loge("Kernel - invalid!");
    return;
  }

  bool unmap_status = true;
  unmap_status &= cl_context.command_queue_inst_.enqueueSVMUnmap(src0);

  if (!unmap_status) {
    ml_loge("Failed to unmap input SVM arguments!");
    return;
  }

  bool svm_status = true;
  svm_status &= kernel->SetKernelSVMArguments(0, src0);
  svm_status &= kernel->SetKernelSVMArguments(2, dst);

  if (!svm_status) {
    ml_loge("Failed to set SVM arguments!");
    return;
  }

  bool arg_status = true;
  arg_status &= kernel->SetKernelArguments(1, &offset0, sizeof(unsigned long));
  arg_status &= kernel->SetKernelArguments(3, &offsetd, sizeof(unsigned long));
  arg_status &= kernel->SetKernelArguments(4, &ne00, sizeof(int));
  arg_status &= kernel->SetKernelArguments(5, &ne01, sizeof(int));
  arg_status &= kernel->SetKernelArguments(6, &ne02, sizeof(int));
  arg_status &= kernel->SetKernelArguments(7, &ne03, sizeof(int));
  arg_status &= kernel->SetKernelArguments(8, &nb01, sizeof(unsigned long));
  arg_status &= kernel->SetKernelArguments(9, &nb02, sizeof(unsigned long));
  arg_status &= kernel->SetKernelArguments(10, &nb03, sizeof(unsigned long));
  arg_status &= kernel->SetKernelArguments(11, &eps, sizeof(float));
  arg_status &= kernel->SetKernelArguments(12, nullptr /*sum*/, 1024 * 1024);

  if (!arg_status) {
    ml_loge("Failed to set arguments!");
    return;
  }

  const int32_t wgc[3] = {1, 1, 1}; // TODO : ...
  const int32_t wgs[3] = {1, 1, 1}; // TODO : ...

  if (!opencl::CommandQueueManager::Global().DispatchCommand(kernel, wgc,
                                                             wgs)) {
    ml_loge("Failed to dispatch!");
    return;
  }

  bool map_status = true;
  map_status &= cl_context.command_queue_inst_.enqueueSVMUnmap(dst);

  if (!map_status) {
    ml_loge("Failed to map output!");
    return;
  }
}

// [0]   [global char *]  -  src0
// [1]   [ulong        ]  -  offset0
// [2]   [global char *]  -  src1
// [3]   [ulong        ]  -  offset1
// [4]   [global char *]  -  dst
// [5]   [ulong        ]  -  offsetd
// [6]   [int          ]  -  ne00
// [7]   [int          ]  -  ne01
// [8]   [int          ]  -  ne02
// [9]   [int          ]  -  ne03
// [10]  [ulong        ]  -  nb01
// [11]  [ulong        ]  -  nb02
// [12]  [ulong        ]  -  nb03
// [13]  [int          ]  -  ne10
// [14]  [int          ]  -  ne11
// [15]  [int          ]  -  ne12
// [16]  [int          ]  -  ne13
// [17]  [ulong        ]  -  nb11
// [18]  [ulong        ]  -  nb12
// [19]  [ulong        ]  -  nb13
// [20]  [ulong        ]  -  nb1
// [21]  [ulong        ]  -  nb2
// [22]  [ulong        ]  -  nb3
// [23]  [float        ]  -  eps
// [24]  [local float *]  -  sum
//
// TODO : Use nntrainer convention for M, N, K and calculate ggml parameters
//
void ggml_rms_norm_mul_cl(void *src0, unsigned long offset0, void *src1,
                          unsigned long offset1, void *dst,
                          unsigned long offsetd, int32_t ne00, int32_t ne01,
                          int32_t ne02, int32_t ne03, unsigned long nb01,
                          unsigned long nb02, unsigned long nb03, int32_t ne10,
                          int32_t ne11, int32_t ne12, int32_t ne13,
                          unsigned long nb11, unsigned long nb12,
                          unsigned long nb13, unsigned long nb1,
                          unsigned long nb2, unsigned long nb3, float eps) {
  Engine &engine = Engine::Global();
  Context &context = *engine.getRegisteredContext("gpu");

  ClContext &cl_context = static_cast<ClContext &>(context);

  ClContext::SharedPtrClKernel kernel =
    cl_context.registerClKernel(getRmsNormKernel(), "kernel_rms_norm_mul");

  if (!kernel) {
    ml_loge("Kernel - invalid!");
    return;
  }

  bool unmap_status = true;
  unmap_status &= cl_context.command_queue_inst_.enqueueSVMUnmap(src0);
  unmap_status &= cl_context.command_queue_inst_.enqueueSVMUnmap(src1);

  if (!unmap_status) {
    ml_loge("Failed to unmap input SVM arguments!");
    return;
  }

  bool svm_status = true;
  svm_status &= kernel->SetKernelSVMArguments(0, src0);
  svm_status &= kernel->SetKernelSVMArguments(2, src1);
  svm_status &= kernel->SetKernelSVMArguments(4, dst);

  if (!svm_status) {
    ml_loge("Failed to set SVM arguments!");
    return;
  }

  bool arg_status = true;
  arg_status &= kernel->SetKernelArguments(1, &offset0, sizeof(unsigned long));
  arg_status &= kernel->SetKernelArguments(3, &offset1, sizeof(unsigned long));
  arg_status &= kernel->SetKernelArguments(5, &offsetd, sizeof(unsigned long));
  arg_status &= kernel->SetKernelArguments(6, &ne00, sizeof(int32_t));
  arg_status &= kernel->SetKernelArguments(7, &ne01, sizeof(int32_t));
  arg_status &= kernel->SetKernelArguments(8, &ne02, sizeof(int32_t));
  arg_status &= kernel->SetKernelArguments(9, &ne03, sizeof(int32_t));
  arg_status &= kernel->SetKernelArguments(10, &nb01, sizeof(unsigned long));
  arg_status &= kernel->SetKernelArguments(11, &nb02, sizeof(unsigned long));
  arg_status &= kernel->SetKernelArguments(12, &nb03, sizeof(unsigned long));
  arg_status &= kernel->SetKernelArguments(13, &ne10, sizeof(int32_t));
  arg_status &= kernel->SetKernelArguments(14, &ne11, sizeof(int32_t));
  arg_status &= kernel->SetKernelArguments(15, &ne12, sizeof(int32_t));
  arg_status &= kernel->SetKernelArguments(16, &ne13, sizeof(int32_t));
  arg_status &= kernel->SetKernelArguments(17, &nb11, sizeof(unsigned long));
  arg_status &= kernel->SetKernelArguments(18, &nb12, sizeof(unsigned long));
  arg_status &= kernel->SetKernelArguments(19, &nb13, sizeof(unsigned long));
  arg_status &= kernel->SetKernelArguments(20, &nb1, sizeof(unsigned long));
  arg_status &= kernel->SetKernelArguments(21, &nb2, sizeof(unsigned long));
  arg_status &= kernel->SetKernelArguments(22, &nb3, sizeof(unsigned long));
  arg_status &= kernel->SetKernelArguments(23, &eps, sizeof(float));
  arg_status &= kernel->SetKernelArguments(24, nullptr /*sum*/, 1024 * 1024);

  if (!arg_status) {
    ml_loge("Failed to set arguments!");
    return;
  }

  const int32_t wgc[3] = {1, 1, 1}; // TODO : ...
  const int32_t wgs[3] = {1, 1, 1}; // TODO : ...

  if (!opencl::CommandQueueManager::Global().DispatchCommand(kernel, wgc,
                                                             wgs)) {
    ml_loge("Failed to dispatch!");
    return;
  }

  bool map_status = true;
  map_status &= cl_context.command_queue_inst_.enqueueSVMUnmap(dst);

  if (!map_status) {
    ml_loge("Failed to map output!");
    return;
  }
}

} // namespace nntrainer
