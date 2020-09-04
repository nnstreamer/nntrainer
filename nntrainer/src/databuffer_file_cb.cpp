// SPDX-License-Identifier: Apache-2.0-only
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file	databuffer_file_cb.cpp
 * @date	4 September 2020
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug		No known bugs except for NYI items
 * @brief	This is databuffer class for Neural Network
 */

#include <databuffer_file_cb.h>
#include <nntrainer-api-common.h>

namespace nntrainer {

/**
 * @details This function prototype is for C-API. So, this should ideally use
 * c style struct. However, as this is being used in c++ setup,
 * this implementation uses c++ class style object for user_data.
 * @note Users referencing this implementation for C-API implementation must
 * restrict to C constructs.
 */
int file_dat_cb(float **input, float **label, bool *last, void *user_data) {
  DataBufferFileUserData *file_data = (DataBufferFileUserData *)user_data;
  *last = false;
  if (file_data == NULL) {
    ml_loge("Invalid user data passed to the data generator");
    return ML_ERROR_INVALID_PARAMETER;
  }

  if (file_data->cur_loc > file_data->size) {
    ml_loge("Reading a data exceeeded file size");
    return ML_ERROR_BAD_ADDRESS;
  }

  if (file_data->cur_loc == file_data->size) {
    *last = true;
    return ML_ERROR_NONE;
  }

  for (size_t i = 0; i < file_data->inputs_size.size(); i++)
    file_data->data_stream.read(reinterpret_cast<char *>(input[i]),
                                file_data->inputs_size[i]);
  for (size_t i = 0; i < file_data->labels_size.size(); i++)
    file_data->data_stream.read(reinterpret_cast<char *>(label[i]),
                                file_data->labels_size[i]);

  return ML_ERROR_NONE;
}

} // namespace nntrainer
