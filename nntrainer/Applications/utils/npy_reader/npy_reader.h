// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Donghak Park <donghak.park@samsung.com>
 *
 * @file   npy_reader.h
 * @date   28 Feb 2024
 * @brief  reader for npy file
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Donghak Park <donghak.park@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef NPY_READER_H_
#define NPY_READER_H_

#include <string>
#include <vector>

namespace nntrainer::util {

/**
 * @brief NpyReader class for read numpy format file
 *
 */
class NpyReader {
private:
  std::string file_path;
  std::vector<int> dims;
  std::vector<float> values;

  /**
   * @brief read numpy file from file_path
   */
  void read_npy_file(const char *file_path);

public:
  /**
   * @brief Construct a new Npy Reader object
   *
   * @param file_path file path for numpy
   */
  NpyReader::NpyReader(const char *file_path) : file_path(file_path) {
    read_npy_file(file_path);
  }

  /**
   * @brief Construct a new Npy Reader object
   *
   * @param dims The dimension of the file you want to read.
   * @param values A vector containing the data you want to use.
   */
  NpyReader::NpyReader(std::vector<int> &dims, std::vector<float> &values) :
    file_path(), dims(dims), values(values) {}

  /**
   * @brief Get the dims object
   *
   * @return const std::vector<int>& dims
   */
  const std::vector<int> &get_dims() const { return dims; }

  /**
   * @brief Get the values object
   *
   * @return const std::vector<float>& values
   */
  const std::vector<float> &get_values() const { return values; };
};

} // namespace nntrainer::util

#endif // NPY_READER_H_
