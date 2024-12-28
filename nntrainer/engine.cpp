// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   engine.cpp
 * @date   27 December 2024
 * @brief  This file contains engine context related functions and classes that
 * manages the engines (NPU, GPU, CPU) of the current environment
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */
#include <dirent.h>
#include <dlfcn.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <app_context.h>
#include <base_properties.h>
#include <context.h>
#include <engine.h>

#ifdef ENABLE_OPENCL
#include <cl_context.h>
#endif

static std::string solib_suffix = ".so";
static std::string contextlib_suffix = "context.so";
static const std::string func_tag = "[Engine] ";

namespace nntrainer {

std::mutex engine_mutex;

std::once_flag global_engine_init_flag;

void Engine::add_default_object(Engine &eg) {
  /// @note all layers should be added to the app_context to guarantee that
  /// createLayer/createOptimizer class is created

  nntrainer::AppContext *app_context = new nntrainer::AppContext();
  app_context->Global();

  eg.registerContext("cpu", app_context);

  // #ifdef ENALBE_OPENCL
  //   eg.registererContext("gpu",
  //                        nntrainer::ClContext(nntrainer::ClContext::Global()));
  // #endif
}

void Engine::registerer(Engine &eg) noexcept {
  try {
    add_default_object(eg);
  } catch (std::exception &e) {
    ml_loge("registering layers failed!!, reason: %s", e.what());
  } catch (...) {
    ml_loge("registering layer failed due to unknown reason");
  }
};

Engine &Engine::Global() {
  static Engine instance;
  /// in g++ there is a bug that hangs up if caller throws,
  /// so registerer is noexcept although it'd better not
  /// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=70298
  std::call_once(global_engine_init_flag, registerer, std::ref(instance));
  return instance;
}

std::string
Engine::parseComputeEngine(const std::vector<std::string> &props) const {
  for (auto &prop : props) {
    std::string key, value;
    int status = nntrainer::getKeyValue(prop, key, value);
    if (nntrainer::istrequal(key, "engine")) {
      constexpr const auto data =
        std::data(props::ComputeEngineTypeInfo::EnumList);
      for (uint i = 0; i < props::ComputeEngineTypeInfo::EnumList.size(); ++i) {
        if (nntrainer::istrequal(value.c_str(),
                                 props::ComputeEngineTypeInfo::EnumStr[i])) {
          return props::ComputeEngineTypeInfo::EnumStr[i];
        }
      }
    }
  }

  return "cpu";
}

/**
 * @brief Get the Full Path from given string
 * @details path is resolved in the following order
 * 1) if @a path is absolute, return path
 * ----------------------------------------
 * 2) if @a base == "" && @a path == "", return "."
 * 3) if @a base == "" && @a path != "", return @a path
 * 4) if @a base != "" && @a path == "", return @a base
 * 5) if @a base != "" && @a path != "", return @a base + "/" + path
 *
 * @param path path to calculate from base
 * @param base base path
 * @return const std::string
 */
const std::string getFullPath(const std::string &path,
                              const std::string &base) {
  /// if path is absolute, return path
  if (path[0] == '/') {
    return path;
  }

  if (base == std::string()) {
    return path == std::string() ? "." : path;
  }

  return path == std::string() ? base : base + "/" + path;
}

const std::string Engine::getWorkingPath(const std::string &path) {
  return getFullPath(path, working_path_base);
}

void Engine::setWorkingDirectory(const std::string &base) {
  DIR *dir = opendir(base.c_str());

  if (!dir) {
    std::stringstream ss;
    ss << func_tag << "path is not directory or has no permission: " << base;
    throw std::invalid_argument(ss.str().c_str());
  }
  closedir(dir);

  char *ret = getRealpath(base.c_str(), nullptr);

  if (ret == nullptr) {
    std::stringstream ss;
    ss << func_tag << "failed to get canonical path for the path: ";
    throw std::invalid_argument(ss.str().c_str());
  }

  working_path_base = std::string(ret);
  ml_logd("working path base has set: %s", working_path_base.c_str());
  free(ret);
}

int Engine::registerContext(const std::string &library_path,
                            const std::string &base_path) {
  const std::string full_path = getFullPath(library_path, base_path);

  void *handle = dlopen(full_path.c_str(), RTLD_LAZY | RTLD_LOCAL);
  const char *error_msg = dlerror();

  NNTR_THROW_IF(handle == nullptr, std::invalid_argument)
    << func_tag << "open plugin failed, reason: " << error_msg;

  nntrainer::ContextPluggable *pluggable =
    reinterpret_cast<nntrainer::ContextPluggable *>(
      dlsym(handle, "ml_train_context_pluggable"));

  error_msg = dlerror();
  auto close_dl = [handle] { dlclose(handle); };
  NNTR_THROW_IF_CLEANUP(error_msg != nullptr || pluggable == nullptr,
                        std::invalid_argument, close_dl)
    << func_tag << "loading symbol failed, reason: " << error_msg;

  auto context = pluggable->createfunc();
  NNTR_THROW_IF_CLEANUP(context == nullptr, std::invalid_argument, close_dl)
    << func_tag << "created pluggable context is null";
  auto type = context->getName();
  NNTR_THROW_IF_CLEANUP(type == "", std::invalid_argument, close_dl)
    << func_tag << "custom layer must specify type name, but it is empty";

  registerContext(type, context);

  return 0;
}

} // namespace nntrainer
