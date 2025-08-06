// Original code :
// https://github.com/ggml-org/ggml/blob/master/src/ggml-opencl/kernels/rms_norm.cl
// Original code :
// https://github.com/ggml-org/ggml/blob/master/src/ggml-opencl/kernels/add.cl
// Original code :
// https://github.com/ggml-org/ggml/blob/master/src/ggml-opencl/kernels/glu.cl

#ifndef __GGML_KERNEL_STRINGS_H__
#define __GGML_KERNEL_STRINGS_H__

#include <string>

namespace nntrainer {

// glu.cl
const std::string &getGegluKernel();
const std::string &getGegluF16Kernel();
const std::string &getRegluKernel();
const std::string &getRegluF16Kernel();
const std::string &getSwigluKernel();
const std::string &getSwigluF16Kernel();
const std::string &getGegluErfKernel();
const std::string &getGegluErfF16Kernel();
const std::string &getGegluQuickKernel();
const std::string &getGegluQuickF16Kernel();

// add.cl
const std::string &getAddKernel();
const std::string &getAddRowKernel();
const std::string &getAddF16Kernel();
const std::string &getAddRowF16Kernel();

// rms_norm.cl
const std::string &getRmsNormKernel();
const std::string &getRmsNormMulKernel();

} // namespace nntrainer
#endif
