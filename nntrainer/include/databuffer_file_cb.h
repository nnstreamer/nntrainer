// SPDX-License-Identifier: Apache-2.0-only
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file	databuffer_file_cb.h
 * @date	4 September 2020
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug		No known bugs except for NYI items
 * @brief	This is databuffer class for Neural Network
 *
 */

#ifndef __DATABUFFER_FILE_CB_H__
#define __DATABUFFER_FILE_CB_H__
#ifdef __cplusplus

#include <fstream>
#include <numeric>
#include <string>
#include <vector>

#include <nntrainer_error.h>
#include <nntrainer_log.h>

namespace nntrainer {

/**
 * @class DataBufferFileUserData Data buffer with file user's data
 * @brief Data buffer with file helpers class
 */
class DataBufferFileUserData {
public:
  DataBufferFileUserData(const std::string file) :
    size(0),
    cur_loc(0),
    filename(file) {
    data_stream = std::ifstream(filename, std::ios::in | std::ios::binary);
    if (!data_stream.good())
      throw std::invalid_argument("Invalid data file given for dataset");

    data_stream.seekg(0, std::ios::end);
    long end = data_stream.tellg();
    data_stream.seekg(0, std::ios::beg);
    long begin = data_stream.tellg();
    if (end < 0 or begin < 0)
      throw std::runtime_error("Unable to determine the data filesize");
    size += end - begin;
  }

  void reset() {
    data_stream.clear();
    data_stream.seekg(0, std::ios_base::beg);
  }

  void setInputLabelSize(const std::vector<size_t> &input_size,
                         const std::vector<size_t> &label_size) {
    this->inputs_size = input_size;
    this->labels_size = label_size;
  }

  size_t getSingleDataSize() {
    return std::accumulate(inputs_size.begin(), inputs_size.end(), 0) +
           std::accumulate(labels_size.begin(), labels_size.end(), 0);
  }

  size_t size, cur_loc;
  std::ifstream data_stream;
  std::string filename;
  std::vector<size_t> inputs_size, labels_size;
};

/**
 * @brief   Dataset generator callback for handling data from dat format files
 */
int file_dat_cb(float **input, float **label, bool *last, void *user_data);

} // namespace nntrainer
#endif /* __cplusplus */
#endif /* __DATABUFFER_FILE_CB_H__ */
