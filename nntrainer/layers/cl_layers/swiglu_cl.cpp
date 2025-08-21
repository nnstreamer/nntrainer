// SPDX-License-Identifier: Apache-2.0
/**
 *
 * @file   swiglu_cl.cpp
 * @date   6th June 2024
 * @brief  Implementation of SwiGLU activation function
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Niket Agarwal <niket.a@samsung.com>
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include "swiglu_cl.h"
#include "nntrainer_log.h"
#include <blas_kernel_strings.h>
#include <iostream>

namespace nntrainer {

static constexpr size_t OUT_IDX = 0;
static constexpr size_t INPUT_IDX_1 = 0;
static constexpr size_t INPUT_IDX_2 = 1;

bool SwiGLULayerCl::registerClKernels(ClContext &cl_context) {
  auto &layer_kernel_ptrs = getLayerKernelPtrs();

  // check if the kernels are already registered.
  if (!layer_kernel_ptrs.empty()) {
    ml_loge("kernels for swiglu_cl are already registered.");
    return false;
  }

  do {
    ClContext::SharedPtrClKernel kernel_swiglu_ptr = nullptr;

    kernel_swiglu_ptr =
      cl_context.registerClKernel(getSwiGluClKernel(), "swiglu_cl");

    if (!kernel_swiglu_ptr) {
      ml_loge("OpenCL Error: Fail to register swiglu_cl kernel");
      break;
    }
    layer_kernel_ptrs.emplace_back(kernel_swiglu_ptr);

#ifdef ENABLE_FP16
    kernel_swiglu_ptr =
      cl_context.registerClKernel(getSwiGluClKernelFP16(), "swiglu_cl_fp16");

    if (!kernel_swiglu_ptr) {
      ml_loge("OpenCL Error: Fail to register swiglu_cl_fp16 kernel");
      break;
    }
    layer_kernel_ptrs.emplace_back(kernel_swiglu_ptr);
#endif

    return true;
  } while (false);

  // clear all registered kernels if any error occurs during registration
  layer_kernel_ptrs.clear();

  return false;
}

void SwiGLULayerCl::finalize(nntrainer::InitLayerContext &context) {
  context.setOutputDimensions({context.getInputDimensions()[0]});
}

void SwiGLULayerCl::forwarding(RunLayerContext &context, bool training) {
  Tensor &in1 = context.getInput(INPUT_IDX_1);
  Tensor &in2 = context.getInput(INPUT_IDX_2);
  Tensor &out = context.getOutput(OUT_IDX);
  swigluProcess(in1, in2, out);
}

void SwiGLULayerCl::incremental_forwarding(RunLayerContext &context,
                                           unsigned int from, unsigned int to,
                                           bool training) {
  Tensor &in1 = context.getInput(INPUT_IDX_1);
  Tensor &in2 = context.getInput(INPUT_IDX_2);
  Tensor &out = context.getOutput(OUT_IDX);

  if (from) {
    NNTR_THROW_IF(to - from != 1, std::invalid_argument)
      << "incremental step size is not 1";
    from = 0;
    to = 1;
  }

  swigluProcess(in1, in2, out);
}

void SwiGLULayerCl::swigluProcess(Tensor const &in1, Tensor const &in2,
                                  Tensor &result) {

  unsigned int dim1, dim2;
  dim1 = in1.batch() * in1.channel() * in1.height();
  dim2 = in1.width();

  if (in1.getDataType() == ml::train::TensorDim::DataType::FP32) {
    float *data1 = in1.getData();
    float *data2 = in2.getData();
    float *rdata = result.getData();
    swiglu_cl(data1, data2, rdata, dim1, dim2);
  } else if (in1.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    _FP16 *data1 = in1.getData<_FP16>();
    _FP16 *data2 = in2.getData<_FP16>();
    _FP16 *rdata = result.getData<_FP16>();
    swiglu_cl_fp16(data1, data2, rdata, dim1, dim2);
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  }
}

void SwiGLULayerCl::swiglu_cl(float *matAdata, float *vecXdata, float *vecYdata,
                              unsigned int dim1, unsigned int dim2,
                              const bool use_svm) {
  auto cl_context =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));

  const auto &kernel = getLayerKernelPtrs()[Kernels::SWIGLU_CL];
  int dim = int(dim1 * dim2);

  if (!use_svm) {
    bool write_result = true;
    auto &clbuffInstance = ClBufferManager::Global();

    write_result &= clbuffInstance.getInBufferA()->WriteDataRegion(
      cl_context->command_queue_inst_, dim * sizeof(float), matAdata);
    write_result &= clbuffInstance.getInBufferB()->WriteDataRegion(
      cl_context->command_queue_inst_, dim * sizeof(float), vecXdata);
    if (!write_result) {
      return;
    }

    auto bufferInA = clbuffInstance.getInBufferA()->GetBuffer();
    auto bufferInB = clbuffInstance.getInBufferB()->GetBuffer();
    auto bufferOutA = clbuffInstance.getOutBufferA()->GetBuffer();

    bool set_result = true;
    set_result &= kernel->SetKernelArguments(0, &bufferInA, sizeof(cl_mem));
    set_result &= kernel->SetKernelArguments(1, &bufferInB, sizeof(cl_mem));
    set_result &= kernel->SetKernelArguments(2, &bufferOutA, sizeof(cl_mem));
    if (!set_result) {
      return;
    }
  } else {
    bool set_svm_result = true;
    set_svm_result &= kernel->SetKernelSVMArguments(0, matAdata);
    set_svm_result &= kernel->SetKernelSVMArguments(1, vecXdata);
    set_svm_result &= kernel->SetKernelSVMArguments(2, vecYdata);
    if (!set_svm_result) {
      ml_loge("Failed to set svm");
      return;
    }
  }

  std::array<size_t, 3> global_work_size = {dim, 1, 1};

  cl_event swiglu_wait;

  if (!cl_context->command_queue_inst_.enqueueKernel(
        kernel->GetKernel(), global_work_size.size(), global_work_size.data(),
        nullptr, 0, nullptr, &swiglu_wait)) {
  }

  cl_context->command_queue_inst_.waitForEvent(1, &swiglu_wait);

  if (!use_svm) {
    auto &clbuffInstance = ClBufferManager::Global();
    if (!clbuffInstance.getOutBufferA()->ReadDataRegion(
          cl_context->command_queue_inst_, dim * sizeof(float), vecYdata)) {
      return;
    }
  }
}

#ifdef ENABLE_FP16
void SwiGLULayerCl::swiglu_cl_fp16(_FP16 *matAdata, _FP16 *vecXdata,
                                   _FP16 *vecYdata, unsigned int dim1,
                                   unsigned int dim2, bool svm) {

  bool result = false;

  auto *global_cl_context =
    static_cast<ClContext *>(Engine::Global().getRegisteredContext("gpu"));
  auto &clbuffInstance = ClBufferManager::Global();

  do {
    const auto &kernel_swiglu_ptr =
      getLayerKernelPtrs()[Kernels::SWIGLU_CL_FP16];

    int dim = int(dim1 * dim2);

    result = clbuffInstance.getInBufferA()->WriteDataRegion(
      global_cl_context->command_queue_inst_, dim * sizeof(_FP16), matAdata);
    if (!result) {
      break;
    }

    result = clbuffInstance.getInBufferB()->WriteDataRegion(
      global_cl_context->command_queue_inst_, dim * sizeof(_FP16), vecXdata);
    if (!result) {
      break;
    }

    auto bufferInA = clbuffInstance.getInBufferA()->GetBuffer();
    auto bufferInB = clbuffInstance.getInBufferB()->GetBuffer();
    auto bufferOutA = clbuffInstance.getOutBufferA()->GetBuffer();

    bool set_result = true;
    set_result &=
      kernel_swiglu_ptr->SetKernelArguments(0, &bufferInA, sizeof(cl_mem));
    set_result &=
      kernel_swiglu_ptr->SetKernelArguments(1, &bufferInB, sizeof(cl_mem));
    set_result &=
      kernel_swiglu_ptr->SetKernelArguments(2, &bufferOutA, sizeof(cl_mem));
    if (!set_result) {
      break;
    }

    // NOTE(mwlasiuk) : local size can not be larger than global
    const int32_t desired_local = 64;
    const bool can_use_desired = dim >= desired_local;
    const int32_t chosen_local = can_use_desired ? desired_local : dim;

    const int work_groups_count[3] = {dim, 1, 1};
    /// @todo: create a group size by device & input
    const int work_group_size[3] = {chosen_local, 1, 1}; // test-value

    result = global_cl_context->command_queue_inst_.DispatchCommand(
      kernel_swiglu_ptr, work_groups_count, work_group_size);
    if (!result) {
      break;
    }

    result = clbuffInstance.getOutBufferA()->ReadDataRegion(
      global_cl_context->command_queue_inst_, dim * sizeof(_FP16), vecYdata);
    if (!result) {
      break;
    }

  } while (false);
}
#endif

void SwiGLULayerCl::calcDerivative(nntrainer::RunLayerContext &context) {
  std::throw_with_nested(std::runtime_error("Training is not supported yet."));
}

void SwiGLULayerCl::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, swiglu_props);
  if (!remain_props.empty()) {
    std::string msg = "[SwigluLayerCl] Unknown Layer Properties count " +
                      std::to_string(values.size());
    throw exception::not_supported(msg);
  }
}

std::vector<ClContext::SharedPtrClKernel> &SwiGLULayerCl::getLayerKernelPtrs() {
  /**< kernel list relevant with this layer */
  static std::vector<ClContext::SharedPtrClKernel> layer_kernel_ptrs;
  return layer_kernel_ptrs;
}

} // namespace nntrainer
