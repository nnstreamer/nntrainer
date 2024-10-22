// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file    cl_context.h
 * @date    23 Feb 2024
 * @see     https://github.com/nnstreamer/nntrainer
 * @author  Debadri Samaddar <s.debadri@samsung.com>
 * @author  Niket Agarwal <niket.a@samsung.com>
 * @author  Thummala Pallavi <t.pallavi@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   This file contains app context related functions and classes that
 * manages the global configuration of the current OpenCL environment. It also
 * creates the OpenCL command queue and context.
 */

#include <addition_layer_cl.h>
#include <attention_kernel_strings.h>
#include <blas_kernel_strings.h>
#include <cl_context.h>
#include <concat_cl.h>
#include <fc_layer_cl.h>
#include <reshape_cl.h>
#include <rmsnorm_layer_cl.h>
#include <swiglu_cl.h>
#include <transpose_cl.h>

namespace nntrainer {

std::mutex cl_factory_mutex;

std::once_flag global_cl_context_init_flag;

static void add_default_object(ClContext &cc) {

  cc.registerFactory(nntrainer::createLayer<FullyConnectedLayerCl>,
                     FullyConnectedLayerCl::type,
                     ml::train::LayerType::LAYER_FC);

  // cc.registerFactory(nntrainer::createLayer<AdditionLayerCL>,
  //                    AdditionLayerCL::type,
  //                    ml::train::LayerType::LAYER_ADDITION);

  cc.registerFactory(nntrainer::createLayer<SwiGLULayerCl>, SwiGLULayerCl::type,
                     ml::train::LayerType::LAYER_SWIGLU);

  cc.registerFactory(nntrainer::createLayer<ReshapeLayerCl>,
                     ReshapeLayerCl::type, ml::train::LayerType::LAYER_RESHAPE);

  cc.registerFactory(nntrainer::createLayer<RMSNormLayerCl>,
                     RMSNormLayerCl::type, ml::train::LayerType::LAYER_RMSNORM);

  cc.registerFactory(nntrainer::createLayer<ConcatLayerCl>, ConcatLayerCl::type,
                     ml::train::LayerType::LAYER_CONCAT);

  cc.registerFactory(nntrainer::createLayer<TransposeLayerCl>,
                     TransposeLayerCl::type,
                     ml::train::LayerType::LAYER_TRANSPOSE);
}

static void registerer(ClContext &cc) noexcept {
  try {
    cc.initBlasClKernels();
    add_default_object(cc);
  } catch (std::exception &e) {
    ml_loge("cl_context: registering layers failed!!, reason: %s", e.what());
  } catch (...) {
    ml_loge("cl_context: registering layer failed due to unknown reason");
  }
};

ClContext &ClContext::Global() {
  static ClContext instance;

  // initializing commandqueue and context
  bool result = instance.clInit();

  if (!result) {
    ml_loge("cl_context: opencl command queue creation failed");
  }

  /// in g++ there is a bug that hangs up if caller throws,
  /// so registerer is noexcept although it'd better not
  /// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=70298
  std::call_once(global_cl_context_init_flag, registerer, std::ref(instance));
  return instance;
}

template <typename T>
const int ClContext::registerFactory(const FactoryType<T> factory,
                                     const std::string &key,
                                     const int int_key) {
  static_assert(isSupported<T>::value,
                "cl_context: given type is not supported for current context");

  auto &index = std::get<IndexType<T>>(factory_map);
  auto &str_map = std::get<StrIndexType<T>>(index);
  auto &int_map = std::get<IntIndexType>(index);

  std::string assigned_key = key == "" ? factory({})->getType() : key;

  std::transform(assigned_key.begin(), assigned_key.end(), assigned_key.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  const std::lock_guard<std::mutex> lock(cl_factory_mutex);
  if (str_map.find(assigned_key) != str_map.end()) {
    std::stringstream ss;
    ss << "cl_context: cannot register factory with already taken key: " << key;
    throw std::invalid_argument(ss.str().c_str());
  }

  if (int_key != -1 && int_map.find(int_key) != int_map.end()) {
    std::stringstream ss;
    ss << "cl_context: cannot register factory with already taken int key: "
       << int_key;
    throw std::invalid_argument(ss.str().c_str());
  }

  int assigned_int_key = int_key == -1 ? str_map.size() + 1 : int_key;

  str_map[assigned_key] = factory;
  int_map[assigned_int_key] = assigned_key;

  ml_logd("cl_context: factory has registered with key: %s, int_key: %d",
          assigned_key.c_str(), assigned_int_key);

  return assigned_int_key;
}

void ClContext::initBlasClKernels() {
  if (blas_kernels_initialized) {
    ml_logi(
      "ClContext: Default blas kernels already registered and initialized");
    return;
  }

  registerClKernel(sgemv_cl_kernel_, "sgemv_cl");
  registerClKernel(sgemv_cl_noTrans_kernel_, "sgemv_cl_noTrans");
  registerClKernel(dot_cl_kernel_, "dot_cl");
  registerClKernel(sgemm_cl_noTrans_kernel_, "sgemm_cl_noTrans");
  registerClKernel(sgemm_cl_transA_kernel_, "sgemm_cl_transA");
  registerClKernel(sgemm_cl_transB_kernel_, "sgemm_cl_transB");
  registerClKernel(sgemm_cl_transAB_kernel_, "sgemm_cl_transAB");
  registerClKernel(addition_cl_kernel_, "addition_cl");
  registerClKernel(sscal_cl_kernel_, "sscal_cl");

#ifdef ENABLE_FP16
  registerClKernel(sgemv_cl_kernel_fp16_, "sgemv_cl_fp16");
  registerClKernel(sgemv_cl_noTrans_kernel_fp16_, "sgemv_cl_noTrans_fp16");
  registerClKernel(dot_cl_kernel_fp16_, "dot_cl_fp16");
  registerClKernel(sgemm_cl_noTrans_kernel_fp16_, "sgemm_cl_noTrans_fp16");
  registerClKernel(sgemm_cl_transA_kernel_fp16_, "sgemm_cl_transA_fp16");
  registerClKernel(sgemm_cl_transB_kernel_fp16_, "sgemm_cl_transB_fp16");
  registerClKernel(sgemm_cl_transAB_kernel_fp16_, "sgemm_cl_transAB_fp16");
  registerClKernel(addition_cl_kernel_fp16_, "addition_cl_fp16");
  registerClKernel(sscal_cl_kernel_fp16_, "sscal_cl_fp16");
#endif
  blas_kernels_initialized = true;
}

void ClContext::initAttentionClKernels() {
  if (attention_kernels_initialized) {
    ml_logi("ClContext: Default attention kernels already registered and "
            "initialized");
    return;
  }

  registerClKernel(rotary_emb_cl_kernel_, "rotary_emb_cl");

#ifdef ENABLE_FP16
  registerClKernel(rotary_emb_cl_kernel_fp16_, "rotary_emb_cl_fp16");
#endif
  attention_kernels_initialized = true;
}

const ClContext::SharedPtrClKernel
ClContext::registerClKernel(std::string kernel_string,
                            std::string kernel_name) {
  // check if created before
  if (ocl_kernel_map.find(kernel_name) != ocl_kernel_map.end()) {
    ml_logi("Kernel already registered and initialized: %s",
            kernel_name.c_str());
    return ocl_kernel_map[kernel_name];
  }

  // creating shared_ptr for kernel object
  SharedPtrClKernel kernelPtr = std::make_shared<opencl::Kernel>();
  if (!clCreateKernel(kernel_string, kernel_name, kernelPtr)) {
    ml_loge("Failed to register kernel %s", kernel_name.c_str());
    return nullptr;
  }
  // add to map
  ocl_kernel_map.emplace(kernel_name, kernelPtr);
  return ocl_kernel_map[kernel_name];
}

bool ClContext::clCreateKernel(std::string &kernel_string,
                               std::string &kernel_name,
                               const SharedPtrClKernel &kernel_ptr_) {

  ml_logi("Kernel initializing: %s", kernel_name.c_str());

  bool result = false;

  do {
    opencl::Program program;

    // reading binary
    std::ifstream fs(opencl::Program::DEFAULT_KERNEL_PATH + "/" + kernel_name +
                       "_kernel.bin",
                     std::ios::binary | std::ios::in);

    if (fs.good()) {
      fs.seekg(0, std::ios::end);
      size_t binary_size = fs.tellg();
      fs.seekg(0, std::ios::beg);

      unsigned char chunk[binary_size];
      fs.read((char *)chunk, binary_size);

      result = program.CreateCLProgramWithBinary(
        context_inst_.GetContext(), context_inst_.GetDeviceId(), binary_size,
        chunk,
        opencl::Program::DEFAULT_KERNEL_PATH + "/" + kernel_name +
          "_kernel.bin",
        "");
    } else {
      result =
        program.CreateCLProgram(context_inst_.GetContext(),
                                context_inst_.GetDeviceId(), kernel_string, "");
    }

    if (!result) {
      break;
    }

    result = kernel_ptr_->CreateKernelFromProgram(program, kernel_name);
    if (!result) {
      break;
    }

  } while (false);

  return result;
}

/**
 * @copydoc const int ClContext::registerFactory
 */
template const int ClContext::registerFactory<nntrainer::Layer>(
  const FactoryType<nntrainer::Layer> factory, const std::string &key,
  const int int_key);

} // namespace nntrainer
