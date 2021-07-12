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

#include "pow.h"

namespace custom {

static constexpr size_t SINGLE_INOUT_IDX = 0;

namespace PowUtil {

/**
 * @brief Entry structure for handling properties
 */
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

void PowLayer::setProperty(const std::vector<std::string> &values) {
  PowUtil::Entry e;

  for (auto &val : values) {
    e = PowUtil::getKeyValue(val);

    if (e.key != "exponent") {
      std::string msg =
        "[PowLayer] Unknown Layer Property Key for value " + std::string(e.key);
      throw std::invalid_argument(msg);
    }

    exponent = std::stoi(e.value);
  }
}

void PowLayer::finalize(nntrainer::InitLayerContext &context) {
  context.setOutputDimensions(context.getInputDimensions());

  /**
   * If there are weights inside the layer, it should be added here
   * auto weight_idx = context.requestWeight(
   *      input_dim, weight_initializer, true, "Pow::sample");
   */
}

void PowLayer::forwarding(nntrainer::RunLayerContext &context, bool training) {
#ifdef DEBUG
  /// intended here to demonstrate that PowLayer::forwarding is being called
  std::cout << "pow layer forward is called\n";
#endif

  /// net hidden are used to save var,
  context.getOutput(SINGLE_INOUT_IDX) =
    context.getInput(SINGLE_INOUT_IDX).pow(exponent);

#ifdef DEBUG
  std::cout << "input: " << context.getInput(SINGLE_INOUT_IDX);
  std::cout << "output: " << context.getOutput(SINGLE_INOUT_IDX);
  PowUtil::pause();
#endif
}

void PowLayer::calcDerivative(nntrainer::RunLayerContext &context) {
/// intended here to demonstrate that PowLayer::backwarding is being called
#ifdef DEBUG
  std::cout << "pow layer backward is called\n";
#endif

  nntrainer::Tensor &derivative_ =
    context.getIncomingDerivative(SINGLE_INOUT_IDX);
  nntrainer::Tensor &dx = context.getOutgoingDerivative(SINGLE_INOUT_IDX);

  dx = derivative_.multiply(exponent);

#ifdef DEBUG
  std::cout << "input: " << context.getOutput(SINGLE_INOUT_IDX);
  std::cout << "output: " << context.getInput(SINGLE_INOUT_IDX);
  PowUtil::pause();
#endif
}

#ifdef PLUGGABLE

nntrainer::Layer *create_pow_layer() {
  auto layer = new PowLayer();
  std::cout << "power created\n";
  return layer;
}

void destory_pow_layer(nntrainer::Layer *layer) {
  std::cout << "power deleted\n";
  delete layer;
}

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{create_pow_layer,
                                                   destory_pow_layer};
}

#endif

} // namespace custom
