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

#include <cstring>

#include <cl_buffer_manager.h>
#include <opencl_loader.h>

namespace nntrainer {

// to-do: Implementation to be updated with array of Buffer objects if required
// fp16 Buffer objects to be added in future
void ClBufferManager::initBuffers() {
  inBufferA = new opencl::Buffer(context_inst_, buffer_size_bytes, true);
  inBufferB = new opencl::Buffer(context_inst_, buffer_size_bytes, true);
  inBufferC = new opencl::Buffer(context_inst_, buffer_size_bytes, true);
  outBufferA = new opencl::Buffer(context_inst_, buffer_size_bytes, false);
  outBufferB = new opencl::Buffer(context_inst_, buffer_size_bytes, false);

  /// @todo Change to read-only if preprocess is done on CPU
  quantBuffer = new opencl::Buffer(context_inst_, quant_q4_0_size, false);
  scaleBuffer = new opencl::Buffer(context_inst_, scale_q4_0_size, false);

  // Initialize OpenCL images
  cl_image_format img_fmt_1d = {CL_RGBA, CL_FLOAT};
  cl_image_desc img_desc_1d;

  memset(&img_desc_1d, 0, sizeof(img_desc_1d));
  img_desc_1d.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
  img_desc_1d.image_width = buffer_size_bytes / 4;
  img_desc_1d.buffer = inBufferC->GetBuffer();
  input_image = opencl::clCreateImage(context_inst_.GetContext(), 0,
                                      &img_fmt_1d, &img_desc_1d, NULL, NULL);

  img_fmt_1d = {CL_RGBA, CL_HALF_FLOAT};
  memset(&img_desc_1d, 0, sizeof(img_desc_1d));
  img_desc_1d.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
  img_desc_1d.image_width = buffer_size_bytes / 4;
  img_desc_1d.buffer = outBufferB->GetBuffer();
  output_image = opencl::clCreateImage(context_inst_.GetContext(), 0,
                                       &img_fmt_1d, &img_desc_1d, NULL, NULL);

  ml_logi("ClBufferManager: Buffers & images initialized");
}

ClBufferManager::~ClBufferManager() {
  delete inBufferA;
  delete inBufferB;
  delete inBufferC;
  delete outBufferA;
  delete outBufferB;
  delete scaleBuffer;
  delete quantBuffer;
  opencl::clReleaseMemObject(input_image);
  opencl::clReleaseMemObject(output_image);
  ml_logi("ClBufferManager: Buffers destroyed");
}

} // namespace nntrainer
