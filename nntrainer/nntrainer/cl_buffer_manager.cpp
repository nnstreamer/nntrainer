// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file    cl_buffer_manager.cpp
 * @date    01 Dec 2024
 * @see     https://github.com/nnstreamer/nntrainer
 * @author  Debadri Samaddar <s.debadri@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   This file contains global Buffer objects and manages them
 */

#include <cl_buffer_manager.h>

namespace nntrainer {

ClBufferManager &ClBufferManager::getInstance() {
  static ClBufferManager instance;
  return instance;
}

// to-do: Implementation to be updated with array of Buffer objects if required
// fp16 Buffer objects to be added in future
void ClBufferManager::initBuffers() {
  inBufferA = new opencl::Buffer(context_inst_, buffer_size_bytes, true);
  inBufferB = new opencl::Buffer(context_inst_, buffer_size_bytes, true);
  inBufferC = new opencl::Buffer(context_inst_, buffer_size_bytes, true);
  outBufferA = new opencl::Buffer(context_inst_, buffer_size_bytes, false);
  outBufferB = new opencl::Buffer(context_inst_, buffer_size_bytes, false);
  ml_logi("ClBufferManager: Buffers initialized");
}

ClBufferManager::~ClBufferManager() {
  delete inBufferA;
  delete inBufferB;
  delete inBufferC;
  delete outBufferA;
  delete outBufferB;
  ml_logi("ClBufferManager: Buffers destroyed");
}

} // namespace nntrainer
