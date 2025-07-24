// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2022 seongwoo <mhs4670go@naver.com>
 *
 * @file loss_realizer.h
 * @date 4 May 2022
 * @brief NNTrainer graph realizer which remove loss layer for inference
 * @see	https://github.com/nnstreamer/nntrainer
 * @author seongwoo <mhs4670go@naver.com>
 * @bug No known bugs except for NYI items
 */
#ifndef __LOSS_REALIZER_H__
#define __LOSS_REALIZER_H__

#include <vector>

#include <realizer.h>

namespace nntrainer {

/**
 * @brief Graph realizer class which removes loss layer from the graph
 * @note This assumes the number of input / output connection of loss layer == 1
 *
 */
class LossRealizer final : public GraphRealizer {
public:
  /**
   * @brief Construct a new Loss Realizer object
   *
   */
  LossRealizer() = default;

  /**
   * @brief Destroy the Graph Realizer object
   *
   */
  ~LossRealizer() = default;

  /**
   * @brief graph realizer creates a shallow copied graph based on the reference
   * @note loss realizer removes loss layers from GraphRepresentation
   * @param reference GraphRepresentation to be realized
   *
   */
  GraphRepresentation realize(const GraphRepresentation &reference) override;
};

} // namespace nntrainer

#endif // __LOSS_REALIZER_H__
