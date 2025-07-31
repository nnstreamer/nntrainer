/**
 * Copyright (C) 2020 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @file	util_func.cpp
 * @date	08 April 2020
 * @brief	This is collection of math functions
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifdef _WIN32
#define MAX_PATH_LENGTH 1024
#endif

#include <cmath>
#include <fstream>
#include <random>

#include <acti_func.h>
#include <nntrainer_log.h>
#include <util_func.h>

namespace nntrainer {

static std::uniform_real_distribution<float> dist(-0.5, 0.5);

double sqrtDouble(double x) { return sqrt(x); };

bool isFileExist(std::string file_name) {
  std::ifstream infile(file_name);
  return infile.good();
}

template <typename T>
static void checkFile(const T &file, const char *error_msg) {
  if (file.bad() | file.eof() | !file.good() | file.fail()) {
    throw std::runtime_error(error_msg);
  }
}

void checkedRead(std::ifstream &file, char *array, std::streamsize size,
                 const char *error_msg) {
  file.read(array, size);

  checkFile(file, error_msg);
}

void checkedWrite(std::ostream &file, const char *array, std::streamsize size,
                  const char *error_msg) {
  file.write(array, size);

  checkFile(file, error_msg);
}

std::string readString(std::ifstream &file, const char *error_msg) {
  std::string str;
  size_t size;

  checkedRead(file, (char *)&size, sizeof(size), error_msg);

  std::streamsize sz = static_cast<std::streamsize>(size);
  NNTR_THROW_IF(sz < 0, std::invalid_argument)
    << "read string size: " << sz
    << " is too big. It cannot be represented by std::streamsize";

  str.resize(size);
  checkedRead(file, (char *)&str[0], sz, error_msg);

  return str;
}

void writeString(std::ofstream &file, const std::string &str,
                 const char *error_msg) {
  size_t size = str.size();

  checkedWrite(file, (char *)&size, sizeof(size), error_msg);

  std::streamsize sz = static_cast<std::streamsize>(size);
  NNTR_THROW_IF(sz < 0, std::invalid_argument)
    << "write string size: " << size
    << " is too big. It cannot be represented by std::streamsize";

  checkedWrite(file, (char *)&str[0], sz, error_msg);
}

bool endswith(const std::string &target, const std::string &suffix) {
  if (target.size() < suffix.size()) {
    return false;
  }
  size_t spos = target.size() - suffix.size();
  return target.substr(spos) == suffix;
}

int getKeyValue(const std::string &input_str, std::string &key,
                std::string &value) {
  int status = ML_ERROR_NONE;
  auto input_trimmed = input_str;

  std::vector<std::string> list;
  static const std::regex words_regex("[^\\s=]+");
  input_trimmed.erase(
    std::remove(input_trimmed.begin(), input_trimmed.end(), ' '),
    input_trimmed.end());
  auto words_begin = std::sregex_iterator(input_trimmed.begin(),
                                          input_trimmed.end(), words_regex);
  auto words_end = std::sregex_iterator();
  int nwords = std::distance(words_begin, words_end);

  if (nwords != 2) {
    ml_loge("Error: input string must be 'key = value' format "
            "(e.g.{\"key1=value1\",\"key2=value2\"}), \"%s\" given",
            input_trimmed.c_str());
    return ML_ERROR_INVALID_PARAMETER;
  }

  for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
    list.push_back((*i).str());
  }

  key = list[0];
  value = list[1];

  return status;
}

int getValues(int n_str, std::string str, int *value) {
  int status = ML_ERROR_NONE;
  static const std::regex words_regex("[^\\s.,:;!?]+");
  str.erase(std::remove(str.begin(), str.end(), ' '), str.end());
  auto words_begin = std::sregex_iterator(str.begin(), str.end(), words_regex);
  auto words_end = std::sregex_iterator();

  int num = std::distance(words_begin, words_end);
  if (num != n_str) {
    ml_loge("Number of Data is not match");
    return ML_ERROR_INVALID_PARAMETER;
  }
  int cn = 0;
  for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
    value[cn] = std::stoi((*i).str());
    cn++;
  }
  return status;
}

std::vector<std::string> split(const std::string &s, const std::regex &reg) {
  std::vector<std::string> out;
  const int NUM_SKIP_CHAR = 3;
  char char_to_remove[NUM_SKIP_CHAR] = {' ', '[', ']'};
  std::string str = s;
  for (unsigned int i = 0; i < NUM_SKIP_CHAR; ++i) {
    str.erase(std::remove(str.begin(), str.end(), char_to_remove[i]),
              str.end());
  }

  std::regex_token_iterator<std::string::iterator> end;
  std::regex_token_iterator<std::string::iterator> iter(str.begin(), str.end(),
                                                        reg, -1);

  while (iter != end) {
    out.push_back(*iter);
    ++iter;
  }
  return out;
}

bool istrequal(const std::string &a, const std::string &b) {
  if (a.size() != b.size())
    return false;

  return std::equal(a.begin(), a.end(), b.begin(), [](char a_, char b_) {
    return tolower(a_) == tolower(b_);
  });
}

char *getRealpath(const char *name, char *resolved) {
#ifdef _WIN32
  return _fullpath(resolved, name, MAX_PATH_LENGTH);
#else
  resolved = realpath(name, nullptr);
  return resolved;
#endif
}

tm *getLocaltime(tm *tp) {
  time_t t = time(0);
#ifdef _WIN32
  localtime_s(tp, &t);
  return tp;
#else
  return localtime_r(&t, tp);
#endif
}

std::regex getRegex(const std::string &str) {
  std::regex result;

  try {
    result = std::regex(str);
  } catch (const std::regex_error &e) {
    ml_loge("regex_error caught: %s", e.what());
  }

  return result;
}

} // namespace nntrainer
