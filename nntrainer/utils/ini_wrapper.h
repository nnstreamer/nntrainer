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

#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <string>

#include <vector>

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

  /**
   * @brief Default constructor for the Ini Section object
   *
   */
  IniSection() : section_name(""), entry{} {};

  /**
   * @brief Default destructor for the Ini Section object
   *
   */
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
   * @retval true two inisections are equal
   * @retval false two ini sections are not equal
   */
  bool operator==(const IniSection &rhs) const {
    return section_name == rhs.section_name && entry == rhs.entry;
  }

  /**
   * @brief  not equal operator between ini section
   *
   * @param rhs ini section to compare
   * @retval true two inisections are not equal
   * @retval false two inisections are equal
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
  std::string getName() const { return section_name; }

  /**
   * @brief Set the Entry object by key and value
   *
   * @param key key to update
   * @param value value to be added
   */
  void setEntry(const std::string &key, const std::string &value);

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

/**
 * @brief IniWrapper using IniSections
 *
 */
class IniWrapper {
public:
  using Sections = std::vector<IniSection>;

  /**
   * @brief Construct a new Ini Test Wrapper object
   *
   */
  IniWrapper() = default;

  /**
   * @brief Construct a new Ini Test Wrapper object
   *
   * @param name_ name of the ini without `.ini` extension
   * @param sections_ sections that should go into ini
   */
  IniWrapper(const std::string &name_, const Sections &sections_ = {}) :
    name(name_),
    sections(sections_){};

  /**
   * @brief ini operator== to check if IniWrapper is equal
   *
   * @param rhs IniWrapper to compare
   * @retval true true if ini is equal (deeply)
   * @retval false false if ini is not equal
   */
  bool operator==(const IniWrapper &rhs) const {
    return name == rhs.name && sections == rhs.sections;
  }

  /**
   * @brief ini operator!= to check if IniWrapper is not equal
   *
   * @param rhs IniWrapper to compare
   * @retval true if not equal
   * @retval false if equal
   */
  bool operator!=(const IniWrapper &rhs) const { return !operator==(rhs); }

  /**
   * @brief update sections if section is empty, else update section by section
   * by key
   *
   * @param[in] ini IniWrapper
   * @return IniWrapper& this
   */
  IniWrapper &operator+=(const IniWrapper &ini) {
    if (sections.empty()) {
      sections = ini.sections;
    } else {
      updateSections(ini.sections);
    }

    return *this;
  }

  /**
   * @brief update sections if section is empty, else update section by section
   * by key
   *
   * @param[in] rhs IniWrapper
   * @return IniWrapper& a new instance
   */
  IniWrapper operator+(const IniWrapper &rhs) const {
    return IniWrapper(*this) += rhs;
  }

  /**
   * @brief update a single section using operator+=
   *
   * @param string format of `sectionkey / propkey=val | propkey=val| ..`
   * @return IniWrapper& ini wrapper
   */
  IniWrapper &operator+=(const std::string &s) {
    updateSection(s);
    return *this;
  }

  /**
   * @brief update a single section using operator +
   *
   * @param rhs string representatioin to merge
   * @return IniWrapper ini wrapper
   */
  IniWrapper operator+(const std::string &rhs) const {
    return IniWrapper(*this) += rhs;
  }

  /**
   * @brief Get the Ini Name object
   *
   * @return std::string ini name with extension appended
   */
  std::string getIniName() { return name + ".ini"; }

  /**
   * @brief Get the Name
   *
   * @return std::string name
   */
  std::string getName() const { return name; }

  /**
   * @brief save ini to a file, (getIniName() is used to save)
   */
  void save_ini();

  /**
   * @brief save ini by ini_name
   *
   * @param ini_name ini name to svae
   */
  void save_ini(const std::string &ini_name);
  /**
   * @brief erase ini
   *
   */
  void erase_ini() noexcept {
    if (remove(getIniName().c_str())) {
      std::cerr << "remove ini " << getIniName()
                << "failed, reason: " << strerror(errno);
    }
  }

  /**
   * @brief operator<< to print information to outstream
   *
   * @param os outstream
   * @param ini ini wrapper
   * @return std::ostream& outstream
   */
  friend std::ostream &operator<<(std::ostream &os, const IniWrapper &ini) {
    return os << ini.name;
  }

private:
  /**
   * @brief update a section from a formatted string, `sectionkey / propkey=val
   * | propkey=val`
   * @note add containered version of this, something like std::pair
   * @param string_representation "model/optimizer=SGD | ..."
   */
  void updateSection(const std::string &string_representation);

  /**
   * @brief update Section that matches section key of @a sections
   *
   * @param section section
   */
  void updateSection(const IniSection &section);

  /**
   * @brief update sections with following rule
   * if there is a section key, update entry of the section else throw
   * std::invalid_argument
   * @param sections sections to update
   */
  void updateSections(const Sections &sections_);

  std::string name;  /**< name of ini */
  Sections sections; /**< sections of ini */
};

} // namespace nntrainer

#endif // __INI_WRAPPER_H__
