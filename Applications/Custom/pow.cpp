// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   pow.cpp
 * @date   16 November 2020
 * @brief  This file contains the simple pow2 layer which squares input
 * elements.
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */
#include <iostream>
#include <regex>

#include <pow.h>

namespace custom {

const std::string PowLayer::type = "pow";
namespace PowUtil {

struct Entry {
  std::string key;
  std::string value;
};

static Entry getKeyValue(const std::string &input) {
  Entry entry;
  static const std::regex words_regex("[^\\s=]+");

  std::string input_str(input);
  input_str.erase(std::remove(input_str.begin(), input_str.end(), ' '),
                  input_str.end());
  auto words_begin =
    std::sregex_iterator(input_str.begin(), input_str.end(), words_regex);
  auto words_end = std::sregex_iterator();

  int nwords = std::distance(words_begin, words_end);
  if (nwords != 2) {
    throw std::invalid_argument("key, value is not found");
  }

  entry.key = words_begin->str();
  entry.value = (++words_begin)->str();
  return entry;
}

void pause() {
  do {
    std::cout << "Press enter key to continue...\n";
  } while (std::cin.get() != '\n');
}

} // namespace PowUtil

int PowLayer::setProperty(std::vector<std::string> values) {
  PowUtil::Entry e;

  std::vector<std::string> unhandled_values;

  for (auto &val : values) {
    try {
      e = PowUtil::getKeyValue(val);
    } catch (std::invalid_argument &e) {
      std::cerr << e.what() << std::endl;
      return -1;
    }

    if (e.key != "exponent") {
      unhandled_values.push_back(val);
      continue;
    }

    try {
      exponent = std::stoi(e.value);
    } catch (std::invalid_argument &e) {
      std::cerr << e.what() << std::endl;
      return -1;
    } catch (std::out_of_range &e) {
      std::cerr << e.what() << std::endl;
      return -1;
    }
  }

  /// unhandled values are passed to the layer_internal.h
  return nntrainer::Layer::setProperty(unhandled_values);
}

int PowLayer::initialize(nntrainer::Manager &manager) {
  // setting output dimension from input dimension
  output_dim[0] = input_dim[0];

  /// If there are weights inside the layer, it should be added here
  //  setNumWeights(1);
  //  weightAt(0) = Weight(input_dim, weight_initializer, true, "Pow::sample")

  return 0;
}

void PowLayer::forwarding(bool training) {
#ifdef DEBUG
  /// intended here to demonstrate that PowLayer::forwarding is being called
  std::cout << "pow layer forward is called\n";
#endif

  /// net hidden are used to save var,
  net_hidden[0]->getVariableRef() =
    net_input[0]->getVariableRef().pow(exponent);

#ifdef DEBUG
  std::cout << "input: " << net_input[0]->getVariable();
  std::cout << "output: " << net_hidden[0]->getVariable();
  PowUtil::pause();
#endif
}

void PowLayer::calcDerivative() {
/// intended here to demonstrate that PowLayer::backwarding is being called
#ifdef DEBUG
  std::cout << "pow layer backward is called\n";
#endif

  nntrainer::Tensor &derivative_ = net_hidden[0]->getVariableRef();
  nntrainer::Tensor &dx = net_input[0]->getVariableRef();

  dx = derivative_.multiply(exponent);

#ifdef DEBUG
  std::cout << "input: " << net_hidden[0]->getVariable();
  std::cout << "output: " << net_input[0]->getVariable();
  PowUtil::pause();
#endif
}

#ifdef PLUGGABLE

ml::train::Layer *create_pow_layer() {
  auto layer = new PowLayer();
  std::cout << "power created\n";
  return layer;
}

void destory_pow_layer(ml::train::Layer *layer) {
  std::cout << "power deleted\n";
  delete layer;
}

extern "C" {
ml::train::LayerPluggable ml_train_layer_pluggable{create_pow_layer,
                                                   destory_pow_layer};
}

#endif

} // namespace custom
