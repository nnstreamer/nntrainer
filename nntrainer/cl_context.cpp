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
#include <cl_context.h>
#include <cl_kernels/cl_kernels.h>
#include <concat_cl.h>
#include <fc_layer_cl.h>
#include <reshape_cl.h>
#include <rmsnorm_layer_cl.h>
#include <swiglu_cl.h>
#include <transpose_cl.h>

#include <filesystem>

#if defined(_WIN32)
#include <windows.h>
#endif

namespace nntrainer {
#if KERNEL_CACHE
static constexpr bool KERNEL_CACHE_ENABLED = true;
#else
static constexpr bool KERNEL_CACHE_ENABLED = false;
#endif
std::mutex cl_factory_mutex;

std::vector<std::byte> readBinaryFile(const std::string &path) {
  // reading binary
  std::ifstream fs(path, std::ios::binary | std::ios::in);

  if (fs.good()) {
    fs.seekg(0, std::ios::end);
    size_t binary_size = fs.tellg();
    fs.seekg(0, std::ios::beg);

    std::vector<std::byte> data(binary_size);
    fs.read(reinterpret_cast<char *>(data.data()), binary_size);
    return data;
  } else {
    return {};
  }
}

bool writeBinaryFile(const std::string &path,
                     const std::vector<std::byte> &data) {
  std::ofstream fs(path, std::ios::out | std::ios::binary);
  if (!fs) {
    ml_loge("Failed to open file for writing: %s", path.c_str());
    return false;
  }

  fs.write(reinterpret_cast<const char *>(data.data()), data.size());
  return true;
}

void ClContext::initialize() noexcept {
  try {
    if (!clInit()) {
      ml_loge("Error: ClContext::initialize() failed");
      return;
    }

    add_default_object();
    setMemAllocator(std::make_shared<MemAllocator>());
  } catch (std::exception &e) {
    ml_loge("cl_context: registering layers failed!!, reason: %s", e.what());
  } catch (...) {
    ml_loge("cl_context: registering layer failed due to unknown reason");
  }
};

void ClContext::add_default_object() {
  registerFactory(nntrainer::createLayer<FullyConnectedLayerCl>,
                  FullyConnectedLayerCl::type, ml::train::LayerType::LAYER_FC);

  registerFactory(nntrainer::createLayer<AdditionLayerCL>,
                  AdditionLayerCL::type, ml::train::LayerType::LAYER_ADDITION);

  registerFactory(nntrainer::createLayer<SwiGLULayerCl>, SwiGLULayerCl::type,
                  ml::train::LayerType::LAYER_SWIGLU);

  registerFactory(nntrainer::createLayer<ReshapeLayerCl>, ReshapeLayerCl::type,
                  ml::train::LayerType::LAYER_RESHAPE);

  registerFactory(nntrainer::createLayer<RMSNormLayerCl>, RMSNormLayerCl::type,
                  ml::train::LayerType::LAYER_RMSNORM);

  registerFactory(nntrainer::createLayer<ConcatLayerCl>, ConcatLayerCl::type,
                  ml::train::LayerType::LAYER_CONCAT);

  registerFactory(nntrainer::createLayer<TransposeLayerCl>,
                  TransposeLayerCl::type,
                  ml::train::LayerType::LAYER_TRANSPOSE);
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

void ClContext::initializeKernels() {
  NNTR_THROW_IF(cl_kernels_initialized_, std::runtime_error)
    << "OpenCL kernels already initialized ";

  registerClKernel(sgemv_kernel, "sgemv_cl");
  registerClKernel(sgemv_no_trans_kernel, "sgemv_cl_noTrans");
  registerClKernel(dot_kernel, "dot_cl");
  registerClKernel(sgemm_no_trans_kernel, "sgemm_cl_noTrans");
  registerClKernel(sgemm_trans_a_kernel, "sgemm_cl_transA");
  registerClKernel(sgemm_trans_b_kernel, "sgemm_cl_transB");
  registerClKernel(sgemm_trans_ab_kernel, "sgemm_cl_transAB");
  registerClKernel(addition_kernel, "addition_cl");
  registerClKernel(sscal_kernel, "sscal_cl");
  registerClKernel(q6_k_sgemv_kernel, "kernel_mul_mv_q6_K_f32");

  // register Q4_0 kernels
  registerClKernel(convert_block_q4_0_kernel,
                   "kernel_convert_block_q4_0_noshuffle");
  registerClKernel(restore_block_q4_0_kernel, "kernel_restore_block_q4_0");
  registerClKernel(transpose_16bit_kernel, "kernel_transpose_16");
  registerClKernel(transpose_32bit_16bit_kernel, "kernel_transpose_32_16");
  registerClKernel(q4_0_ab_bi_8x4_kernel, "kernel_mul_mat_Ab_Bi_8x4");

  // register INT4 computation kernels
  registerClKernel(int4_gemv_kernel, "fully_connected_gpu_int4_gemv");
  registerClKernel(int4_quantize_input_kernel, "quantize_input_int4");
  registerClKernel(int4_quantize_input_kernel, "quantize_input_int4_pad");

  // attention kernel
  registerClKernel(rotary_emb_kernel, "rotary_emb_cl");

#ifdef ENABLE_FP16
  registerClKernel(hgemv_kernel, "sgemv_cl_fp16");
  registerClKernel(hgemv_no_trans_kernel, "sgemv_cl_noTrans_fp16");
  registerClKernel(dot_fp16_kernel, "dot_cl_fp16");
  registerClKernel(hgemm_no_trans_kernel, "sgemm_cl_noTrans_fp16");
  registerClKernel(hgemm_trans_a_kernel, "sgemm_cl_transA_fp16");
  registerClKernel(hgemm_trans_b_kernel, "sgemm_cl_transB_fp16");
  registerClKernel(hgemm_trans_ab_kernel, "sgemm_cl_transAB_fp16");
  registerClKernel(addition_fp16_kernel, "addition_cl_fp16");
  registerClKernel(hscal_kernel, "sscal_cl_fp16");
  // attention kernel
  registerClKernel(rotary_emb_fp16_kernel, "rotary_emb_cl_fp16");
#endif
  cl_kernels_initialized_ = true;
}

const ClContext::SharedPtrClKernel
ClContext::registerClKernel(std::string kernel_string, std::string kernel_name,
                            std::string compile_options) {
  // check if created before
  if (ocl_kernel_map_.find(kernel_name + compile_options) !=
      ocl_kernel_map_.end()) {
    return ocl_kernel_map_[kernel_name + compile_options];
  }

  // creating shared_ptr for kernel object
  SharedPtrClKernel kernelPtr = std::make_shared<opencl::Kernel>();
  if (!clCreateKernel(kernel_string, kernel_name, compile_options, kernelPtr)) {
    ml_loge("Failed to register kernel %s", kernel_name.c_str());
    return nullptr;
  }
  // add to map
  ocl_kernel_map_.emplace(kernel_name + compile_options, kernelPtr);
  return ocl_kernel_map_[kernel_name + compile_options];
}

bool ClContext::clCreateKernel(std::string &kernel_string,
                               std::string &kernel_name,
                               std::string &compile_options,
                               const SharedPtrClKernel &kernel_ptr_) {

  ml_logi("Kernel initializing: %s", kernel_name.c_str());

  bool result = false;

  opencl::Program program;

  // reading binary
  std::string binary_file_path =
    std::filesystem::path(kernels_cache_path_)
      .append(
        std::to_string(program.GetKernelHash(kernel_string, compile_options)) +
        ".cl.bin")
      .string();
  auto binary_data = KERNEL_CACHE_ENABLED ? readBinaryFile(binary_file_path)
                                          : std::vector<std::byte>();

  if (KERNEL_CACHE_ENABLED && !binary_data.empty()) {
    ml_logi("Using cached version of kernel: %s at path %s",
            kernel_name.c_str(), binary_file_path.c_str());
    result = program.CreateCLProgramWithBinary(
      opencl::ContextManager::Global().GetContext(),
      opencl::ContextManager::Global().GetDeviceId(), binary_data,
      binary_file_path, "");
  } else {
    ml_logi("Binary for kernel %s not found, compiling from source...",
            kernel_name.c_str());
    result =
      program.CreateCLProgram(opencl::ContextManager::Global().GetContext(),
                              opencl::ContextManager::Global().GetDeviceId(),
                              kernel_string, compile_options);

    if (KERNEL_CACHE_ENABLED && result) {
      auto binary = program.GetProgramBinary(
        opencl::ContextManager::Global().GetDeviceId());

      if (binary.empty()) {
        ml_loge("Failed retrieving binary for kernel %s", kernel_name.c_str());
        result = false;
      } else {
        result &= writeBinaryFile(binary_file_path, binary);
      }
    }
  }

  if (!result) {
    return false;
  }

  result = kernel_ptr_->CreateKernelFromProgram(program, kernel_name);

  return result;
}

/**
 * @copydoc const int ClContext::registerFactory
 */
template const int ClContext::registerFactory<nntrainer::Layer>(
  const FactoryType<nntrainer::Layer> factory, const std::string &key,
  const int int_key);

void ClContext::setKernelsCachePath(const std::string &kernels_cache_path) {
  NNTR_THROW_IF(cl_kernels_initialized_, std::runtime_error)
    << "OpenCL kernels already initialized kernels path should be set before "
       "initialization";

  kernels_cache_path_ = kernels_cache_path;

  if (KERNEL_CACHE_ENABLED) {
    if (!std::filesystem::exists(kernels_cache_path_)) {
      std::filesystem::create_directories(kernels_cache_path_);
    }
  }
}

const std::string &ClContext::getKernelsCachePath() const {
  return kernels_cache_path_;
}

} // namespace nntrainer
