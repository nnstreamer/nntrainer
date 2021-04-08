// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file ini_wrapper.cpp
 * @date 08 April 2021
 * @brief NNTrainer Ini Wrapper helps to save ini
 * @note this is to be used with ini_interpreter
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <ini_wrapper.h>

#include <nntrainer_error.h>
#include <parse_util.h>
#include <regex>

namespace nntrainer {

IniSection::IniSection(const std::string &name) : section_name(name) {}

IniSection::IniSection(const std::string &section_name,
                       const std::string &entry_str) :
  IniSection(section_name) {
  setEntry(entry_str);
}

IniSection::IniSection(IniSection &from, const std::string &section_name,
                       const std::string &entry_str) :
  IniSection(from) {
  if (!section_name.empty()) {
    this->section_name = section_name;
  }
  if (!entry_str.empty()) {
    setEntry(entry_str);
  }
}

void IniSection::print(std::ostream &out) {
  out << '[' << section_name << ']' << std::endl;
  for (auto &it : entry)
    out << it.first << " = " << it.second << std::endl;
}

void IniSection::setEntry(const std::map<std::string, std::string> &entry) {
  for (auto &it : entry) {
    this->entry[it.first] = it.second;
  }
}

void IniSection::setEntry(const std::string &entry_str) {
  // setting property separated by "|"
  std::regex words_regex("[^|]+");

  auto words_begin =
    std::sregex_iterator(entry_str.begin(), entry_str.end(), words_regex);
  auto words_end = std::sregex_iterator();

  std::string key, value;
  for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
    std::string cur = (*i).str();

    if (cur[0] == '-') {
      entry.erase(cur.substr(1));
      continue;
    }

    int status = getKeyValue(cur, key, value);
    NNTR_THROW_IF(status != ML_ERROR_NONE, std::invalid_argument)
      << "getKeyValue Failed";
    entry[key] = value;
  }
}

} // namespace nntrainer
