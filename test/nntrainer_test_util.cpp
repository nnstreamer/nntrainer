/**
 * Copyright (C) 2019 Samsung Electronics Co., Ltd. All Rights Reserved.
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
 *
 * @file	nntrainer_test_util.cpp
 * @date	28 April 2020
 * @brief	This is util functions for test
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include "nntrainer_test_util.h"
#include <iostream>

/**
 * @brief replace string and save it in file
 */
void replaceString(const std::string &from, const std::string &to,
                   const std::string n) {
  size_t start_pos = 0;
  std::string s;
  std::ifstream file_stream(n.c_str(), std::ifstream::in);
  if (file_stream.good()) {
    s.assign((std::istreambuf_iterator<char>(file_stream)),
             std::istreambuf_iterator<char>());
    file_stream.close();
  } else {
    s = config_str;
  }
  while ((start_pos = s.find(from, start_pos)) != std::string::npos) {
    s.replace(start_pos, from.length(), to);
    start_pos += to.length();
  }

  std::ofstream data_file(n.c_str());
  data_file << s;
  data_file.close();
}
