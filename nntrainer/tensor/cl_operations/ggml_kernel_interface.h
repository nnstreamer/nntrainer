// Original code :
// https://github.com/ggml-org/ggml/blob/master/src/ggml-opencl/kernels/rms_norm.cl
// Original code :
// https://github.com/ggml-org/ggml/blob/master/src/ggml-opencl/kernels/add.cl
// Original code :
// https://github.com/ggml-org/ggml/blob/master/src/ggml-opencl/kernels/glu.cl

#ifndef __GGML_KERNEL_INTERFACE_H__
#define __GGML_KERNEL_INTERFACE_H__

#include <string>
#include <tensor.h>

namespace nntrainer {
void ggml_swigluCl(const Tensor &src0, const Tensor &src1, Tensor &dst);
void ggml_addCl(const Tensor &src0, const Tensor &src1, Tensor &dst);
void ggml_rms_normCl(const Tensor &src0, Tensor &dst);
void ggml_rms_norm_mulCl(const Tensor &src0, Tensor const &src1, Tensor &dst);
} // namespace nntrainer
#endif
