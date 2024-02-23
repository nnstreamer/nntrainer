// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file    cl_context.h
 * @date    23 Feb 2024
 * @see     https://github.com/nnstreamer/nntrainer
 * @author  Debadri Samaddar <s.debadri@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   This file contains app context related functions and classes that
 * manages the global configuration of the current OpenCL environment. It also
 * creates the OpenCL command queue and context.
 */

#ifndef __CL_CONTEXT_H__
#define __CL_CONTEXT_H__

#include <algorithm>
#include <functional>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <layer.h>
#include <layer_devel.h>

#include <opencl_command_queue_manager.h>

#include <nntrainer_log.h>

namespace nntrainer {

extern std::mutex cl_factory_mutex;

/**
 * @class ClContext contains user-dependent configuration for OpenCL support
 * @brief OpenCL support for app context
 */

class ClContext {

public:
  using PropsType = std::vector<std::string>;

  template <typename T> using PtrType = std::unique_ptr<T>;

  template <typename T>
  using FactoryType = std::function<PtrType<T>(const PropsType &)>;

  template <typename T>
  using PtrFactoryType = PtrType<T> (*)(const PropsType &);

  template <typename T>
  using StrIndexType = std::unordered_map<std::string, FactoryType<T>>;

  /** integer to string key */
  using IntIndexType = std::unordered_map<int, std::string>;

  /**
   * This type contains tuple of
   * 1) integer -> string index
   * 2) string -> factory index
   */
  template <typename T>
  using IndexType = std::tuple<StrIndexType<T>, IntIndexType>;

  template <typename... Ts> using FactoryMap = std::tuple<IndexType<Ts>...>;

  /**
   * @brief   Default constructor
   */
  ClContext() = default;

  /**
   *
   * @brief Get Global cl context.
   *
   * @return ClContext&
   */
  static ClContext &Global();

  /**
   * @brief Factory register function, use this function to register custom
   * object
   *
   * @tparam T object to create. Currently Layer is supported
   * @param factory factory function that creates std::unique_ptr<T>
   * @param key key to access the factory, if key is empty, try to find key by
   * calling factory({})->getType();
   * @param int_key key to access the factory by integer, if it is -1(default),
   * the function automatically unsigned the key and return
   * @return const int unique integer value to access the current factory
   * @throw invalid argument when key and/or int_key is already taken
   */
  template <typename T>
  const int registerFactory(const PtrFactoryType<T> factory,
                            const std::string &key = "",
                            const int int_key = -1) {
    FactoryType<T> f = factory;
    return registerFactory(f, key, int_key);
  }

  /**
   * @brief Factory register function, use this function to register custom
   * object
   *
   * @tparam T object to create. Currently Layer is supported
   * @param factory factory function that creates std::unique_ptr<T>
   * @param key key to access the factory, if key is empty, try to find key by
   * calling factory({})->getType();
   * @param int_key key to access the factory by integer, if it is -1(default),
   * the function automatically unsigned the key and return
   * @return const int unique integer value to access the current factory
   * @throw invalid argument when key and/or int_key is already taken
   */
  template <typename T>
  const int registerFactory(const FactoryType<T> factory,
                            const std::string &key = "",
                            const int int_key = -1);

  /**
   * @brief Create an Object from the integer key
   *
   * @tparam T Type of Object, currently, Only Layer is supported
   * @param int_key integer key
   * @param props property
   * @return PtrType<T> unique pointer to the object
   */
  template <typename T>
  PtrType<T> createObject(const int int_key,
                          const PropsType &props = {}) const {
    static_assert(isSupported<T>::value,
                  "given type is not supported for current app context");
    auto &index = std::get<IndexType<T>>(factory_map);
    auto &int_map = std::get<IntIndexType>(index);

    const auto &entry = int_map.find(int_key);

    if (entry == int_map.end()) {
      std::stringstream ss;
      ss << "Int Key is not found for the object. Key: " << int_key;
      throw exception::not_supported(ss.str().c_str());
    }

    // entry is an object of int_map which is an unordered_map<int, std::string>
    return createObject<T>(entry->second, props);
  }

  /**
   * @brief Create an Object from the string key
   *
   * @tparam T Type of object, currently, only Layer is supported
   * @param key integer key
   * @param props property
   * @return PtrType<T> unique pointer to the object
   */
  template <typename T>
  PtrType<T> createObject(const std::string &key,
                          const PropsType &props = {}) const {
    auto &index = std::get<IndexType<T>>(factory_map);
    auto &str_map = std::get<StrIndexType<T>>(index);

    std::string lower_key;
    lower_key.resize(key.size());

    std::transform(key.begin(), key.end(), lower_key.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    const auto &entry = str_map.find(lower_key);

    if (entry == str_map.end()) {
      std::stringstream ss;
      ss << "Key is not found for the object. Key: " << lower_key;
      throw exception::not_supported(ss.str().c_str());
    }

    // entry -> object of str_map -> unordered_map<std::string, FactoryType<T>>
    return entry->second(props);
  }

  /**
   * @brief destructor to release opencl commandQueue
   */
  ~ClContext() {
    if (cl_initialized) {
      command_queue_inst_.ReleaseCommandQueue();
    }
  };

private:
  // flag to check opencl commandqueue and context inititalization
  bool cl_initialized = false;

  FactoryMap<nntrainer::Layer> factory_map;

  template <typename Args, typename T> struct isSupportedHelper;

  /**
   * @brief supportHelper to check if given type is supported within cl context
   */
  template <typename T, typename... Args>
  struct isSupportedHelper<T, ClContext::FactoryMap<Args...>> {
    static constexpr bool value =
      (std::is_same_v<std::decay_t<T>, std::decay_t<Args>> || ...);
  };

  /**
   * @brief supportHelper to check if given type is supported within cl context
   */
  template <typename T>
  struct isSupported : isSupportedHelper<T, decltype(factory_map)> {};

  // getting static instance of commandqueue
  opencl::CommandQueueManager &command_queue_inst_ =
    opencl::CommandQueueManager::GetInstance();

  /**
   * @brief Initialize opencl commandqueue and context
   * @return true if OpenCL context and command queue creation is successful,
   * false otherwise
   */

  bool clInit() {
    // if commandqueue already created
    if (cl_initialized)
      return true;

    // getContext() called inside createCommandQueue which creates clContext
    bool result = command_queue_inst_.CreateCommandQueue();
    cl_initialized = result;
    return cl_initialized;
  };
};

/**
 * @copydoc const int ClContext::registerFactory
 */
extern template const int ClContext::registerFactory<nntrainer::Layer>(
  const FactoryType<nntrainer::Layer> factory, const std::string &key,
  const int int_key);

} // namespace nntrainer

#endif /* __CL_CONTEXT_H__ */
