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
  readBufferA = new opencl::Buffer(context_inst_, buffer_size_bytes, true);
  readBufferB = new opencl::Buffer(context_inst_, buffer_size_bytes, true);
  readBufferC = new opencl::Buffer(context_inst_, buffer_size_bytes, true);
  writeBufferA = new opencl::Buffer(context_inst_, buffer_size_bytes, false);
  writeBufferB = new opencl::Buffer(context_inst_, buffer_size_bytes, false);
  ml_logi("ClBufferManager: Buffers initialized");
}

ClBufferManager::~ClBufferManager() {
  delete readBufferA;
  delete readBufferB;
  delete readBufferC;
  delete writeBufferA;
  delete writeBufferB;
  ml_logi("ClBufferManager: Buffers destroyed");
}

} // namespace nntrainer
