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

#include <regex>

#include <nntrainer_error.h>
#include <parse_util.h>
#include <util_func.h>

namespace nntrainer {

IniSection::IniSection(const std::string &name) : section_name(name), entry{} {}

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

void IniSection::setEntry(const std::string &key, const std::string &value) {
  entry[key] = value;
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

    setEntry(key, value);
  }
}

void IniWrapper::updateSection(const std::string &s) {

  auto seperator_pos = s.find('/');

  NNTR_THROW_IF(seperator_pos == std::string::npos, std::invalid_argument)
    << "invalid string format is given, please "
       "pass format of sectionKey/properties format";

  auto section_key = s.substr(0, seperator_pos);
  auto properties = s.substr(seperator_pos + 1);

  IniSection target(section_key, properties);

  updateSection(target);
}

void IniWrapper::updateSection(const IniSection &s) {

  auto section = std::find_if(sections.begin(), sections.end(),
                              [&](const IniSection &section) {
                                return section.getName() == s.getName();
                              });

  NNTR_THROW_IF(section == sections.end(), std::invalid_argument)
    << "section key is not found key: " << s.getName();

  (*section) += s;
}

void IniWrapper::updateSections(const Sections &sections_) {
  for (auto &section : sections_) {
    updateSection(section);
  }
}

void IniWrapper::save_ini() { save_ini(getIniName()); }

void IniWrapper::save_ini(const std::string &ini_name) {
  auto out = checkedOpenStream<std::ofstream>(ini_name, std::ios_base::app);

  for (auto &it : sections) {
    it.print(out);
    out << std::endl;
  }
}

} // namespace nntrainer
