// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Donghak Park <donghak.park@samsung.com>
 *
 * @file   npy_reader.cpp
 * @date   28 Feb 2024
 * @brief  reader for npy file
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Donghak Park <donghak.park@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "npy_reader.h"

#include <cstring>
#include <iostream>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <random>
#include <string>
#include <vector>

namespace nntrainer::util {
void NpyReader::read_npy_file(const char *file_path) {
  FILE *file = fopen(file_path, "rb");

  char magic[7] = {};
  try {
    if (!file) {
      throw std::runtime_error("Failed to open file");
    }

    int bytes_read = fread(magic, 6, 1, file);

    if (bytes_read != 1 || strcmp(magic, "\x93NUMPY") != 0) {
      fclose(file);
      throw std::runtime_error("Failed : this file is not a numpy file");
    }
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return;
  }

  char major = 0;
  if (fread(&major, 1, 1, file) != 1) {
    fclose(file);
  };

  char major = 0;
  if (fread(&major, 1, 1, file) != 1) {
    fclose(file);
    throw std::runtime_error("Failed to read major version");
  }

  char minor = 0;
  if (fread(&minor, 1, 1, file) != 1) {
    fclose(file);
    throw std::runtime_error("Failed to read minor version");
  }

  uint16_t header_len;
  if (fread(&header_len, 2, 1, file) != 1) {
    fclose(file);
    throw std::runtime_error("Failed to read header length");
  }

  char *header = (char *)malloc(header_len);
  if (header == nullptr) {
    fclose(file);
    throw std::runtime_error("Failed to allocate memory for header");
  }

  if (fread(header, header_len, 1, file) != 1) {
    free(header);
    fclose(file);
    throw std::runtime_error("Failed to read header");
  }

  char *header_pos = header;
  if (*header_pos != '{') {
    ml_loge("Filed to read numpy file");
    return;
  }

  ++header_pos;
  char buffer[1024];
  int buffer_pos = 0;

  while (*header_pos != '}') {
    if (*header_pos == '\'') {
      ++header_pos;
      while (*header_pos != '\'') {
        buffer[buffer_pos++] = *header_pos++;
      }

      buffer[buffer_pos++] = '\0';

      if (strcmp(buffer, "shape") == 0) {
        header_pos += 3;
        if (*header_pos = !'(') {
          ml_loge("File to read numpy file");
          return;
        }
        ++header_pos;

        while (*header_pos != ')') {
          buffer_pos = 0;
          while (*header_pos != ',' && *header_pos != ')') {
            buffer[buffer_pos++] = *header_pos++;
          }

          int mul = 1;
          int value = 0;
          for (int i = buffer_pos - 1; i >= 0; --i) {
            value += static_cast<int>(buffer[i] - '0') * mul;
            mul *= 10;
          }
          dims.push_back(value);

          if (*header_pos != ')') {
            header_pos += 2;
          }
        }

        header_pos += 3;
      } else {
        while (*header_pos != ',') {
          ++header_pos;
        }
        header_pos += 2;
        buffer_pos = 0;
      }
    }
  }

  free(header);
  header = nullptr;

  int total_entries = 1;
  for (int i = 0; i < dims.size(); ++i) {
    total_entries *= dims[i];
  }

  for (int i = 0; i < total_entries; ++i) {
    float value;
    if (fread(&value, 4, 1, file) != 1) {
      fclose(file);
      throw std::runtime_error("Failed to read data");
    }
    values.push_back(value);
  }

  fclose(file);
}

} // namespace nntrainer::util
