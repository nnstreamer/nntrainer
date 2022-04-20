// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2022 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file bn_realizer.h
 * @date 13 April 2022
 * @brief NNTrainer graph realizer which remove batch normalization layer for
 * inference
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug No known bugs except for NYI items
 */
#ifndef __BN_REALIZER_H__
#define __BN_REALIZER_H__

#include <memory>
#include <string>
#include <vector>

#include <connection.h>
#include <realizer.h>

namespace nntrainer {

/**
 * @brief Graph realizer class which removes batch normalization layer from the
 * graph
 * @note This assumes the number of input / output connection of batch
 * normalization layer == 1
 *
 */
class BnRealizer final : public GraphRealizer {
public:
  /**
   * @brief Construct a new BN Realizer object
   *
   */
  BnRealizer() = default;

  /**
   * @brief Destroy the Graph Realizer object
   *
   */
  ~BnRealizer() = default;

  /**
   * @brief graph realizer creates a shallow copied graph based on the reference
   * @note bn realizer removes batch normalization layers from
   * GraphRepresentation
   * @param reference GraphRepresenstaion to be realized
   * @throw std::invalid_argument if graph is ill formed
   *
   */
  GraphRepresentation realize(const GraphRepresentation &reference) override;
};

} // namespace nntrainer

#endif // __BN_REALIZER_H__
