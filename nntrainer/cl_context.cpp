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
#include <custom_multi_head_attention_layer_cl.h>
#include <fc_layer_cl.h>
#include <reshape_cl.h>
#include <rmsnorm_layer_cl.h>
#include <swiglu_cl.h>

namespace nntrainer {

std::mutex cl_factory_mutex;

std::once_flag global_cl_context_init_flag;

static void add_default_object(ClContext &cc) {

  cc.registerFactory(nntrainer::createLayer<FullyConnectedLayerCl>,
                     FullyConnectedLayerCl::type,
                     ml::train::LayerType::LAYER_FC);

  cc.registerFactory(nntrainer::createLayer<AdditionLayerCL>,
                     AdditionLayerCL::type,
                     ml::train::LayerType::LAYER_ADDITION);

  cc.registerFactory(nntrainer::createLayer<SwiGLULayerCl>, SwiGLULayerCl::type,
                     ml::train::LayerType::LAYER_SWIGLU);

  cc.registerFactory(nntrainer::createLayer<ReshapeLayerCl>,
                     ReshapeLayerCl::type, ml::train::LayerType::LAYER_RESHAPE);

  cc.registerFactory(nntrainer::createLayer<RMSNormLayerCl>,
                     RMSNormLayerCl::type, ml::train::LayerType::LAYER_RMSNORM);

  cc.registerFactory(nntrainer::createLayer<CustomMultiHeadAttentionLayerCl>,
                     CustomMultiHeadAttentionLayerCl::type,
                     ml::train::LayerType::LAYER_CUSTOM_MULTI_HEAD_ATTENTION);
}

static void registerer(ClContext &cc) noexcept {
  try {
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

/**
 * @copydoc const int ClContext::registerFactory
 */
template const int ClContext::registerFactory<nntrainer::Layer>(
  const FactoryType<nntrainer::Layer> factory, const std::string &key,
  const int int_key);

} // namespace nntrainer
