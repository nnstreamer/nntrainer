// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file ini_wrapper.h
 * @date 08 April 2021
 * @brief NNTrainer Ini Wrapper helps to save ini
 * @note this is to be used with ini_interpreter
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <iostream>
#include <map>
#include <string>

#ifndef __INI_WRAPPER_H__
#define __INI_WRAPPER_H__

namespace nntrainer {

/**
 * @brief IniSection class that maps to one ini section
 * @todo consider API style setEntry function
 *
 */
class IniSection {

public:
  /**
   * @brief Construct a new Ini Section object
   *
   * @param name section name
   */
  IniSection(const std::string &name);

  /**
   * @brief Construct a new Ini Section object
   *
   * @param section_name section name
   * @param entry_str entry representing string (separated by `|`)
   */
  IniSection(const std::string &section_name, const std::string &entry_str);

  /**
   * @brief Copy Construct a new Ini Section object
   * @note this function copies entry from @a from and overwrite entry and
   * section_name
   *
   * @param from Ini Section to copy from
   * @param section_name section name to override, if empty, section name is not
   * updated
   * @param entry_str entry string to override the given section name
   */
  IniSection(IniSection &from, const std::string &section_name,
             const std::string &entry_str);

  /**
   * @brief Construct a new Ini Section object
   *
   * @param from Ini Section to copy from
   * @param entry_str entry string to override the given section name
   */
  IniSection(IniSection &from, const std::string &entry_str) :
    IniSection(from, "", entry_str) {}

  IniSection() = default;
  ~IniSection() = default;

  /**
   * @brief +=operator from IniSection
   *
   * @param rhs operand to add
   * @return IniSection& this
   */
  IniSection &operator+=(const IniSection &rhs) {
    setEntry(rhs.entry);
    return *this;
  }

  /**
   * @brief + operator from IniSection
   *
   * @param rhs operand to add
   * @return IniSection new Inisection
   */
  IniSection operator+(const IniSection &rhs) const {
    return IniSection(*this) += rhs;
  }

  /**
   * @brief += operator from string
   *
   * @param s string representation to add
   * @return IniSection& this
   */
  IniSection &operator+=(const std::string &s) {
    setEntry(s);
    return *this;
  }

  /**
   * @brief + operator from string
   *
   * @param s string representation to add
   * @return IniSection Newly created section
   */
  IniSection operator+(const std::string &s) { return IniSection(*this) += s; }

  /**
   * @brief equal operator between ini section
   *
   * @param rhs ini section to compare
   * @return true two inisections are equal
   * @return false two ini sections are not equal
   */
  bool operator==(const IniSection &rhs) const {
    return section_name == rhs.section_name && entry == rhs.entry;
  }

  /**
   * @brief  not equal operator between ini section
   *
   * @param rhs ini section to compare
   * @return true two inisections are not equal
   * @return false two inisections are equal
   */
  bool operator!=(const IniSection &rhs) const { return !operator==(rhs); }

  /**
   * @brief print out a section
   *
   * @param out ostream to print
   */
  void print(std::ostream &out);

  /**
   * @brief Get the Name object
   *
   * @return std::string section name
   */
  std::string getName() { return section_name; }

private:
  /**
   * @brief Set the Entry
   *
   * @param entry set entry from a given map
   */
  void setEntry(const std::map<std::string, std::string> &entry);

  /**
   * @brief set entry from the string representation
   *
   * @param entry_str setEntry as "Type = neuralnetwork | decayrate = 0.96 |
   * -epochs = 1" will delete epochs, and overwrite type and decayrate
   */
  void setEntry(const std::string &entry_str);

  std::string section_name; /**< section name of the ini section */

  /// @note if ini_wrapper needs to be optimized, change this to unordered_map
  std::map<std::string, std::string>
    entry; /**< entry information that this ini contains */

  /**
   * @brief <<operator of a section
   *
   * @param os ostream
   * @param section section to print
   * @return std::ostream& ostream
   */
  friend std::ostream &operator<<(std::ostream &os, const IniSection &section) {
    return os << section.section_name;
  }
};

} // namespace nntrainer

#endif // __INI_WRAPPER_H__
