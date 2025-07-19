// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2022 seongwoo <mhs4670go@naver.com>
 *
 * @file tflite_export_realizer.h
 * @date 18 July 2025
 * @brief NNTrainer graph realizer which remove loss layer for inference
 * @see	https://github.com/nnstreamer/nntrainer
 * @author seongwoo <mhs4670go@naver.com>
 * @author donghak park <donghak.park@samsung.com>
 * @bug No known bugs except for NYI items
 */
#ifndef __TFLITE_EXPORT_REALIZER_H__
#define __TFLITE_EXPORT_REALIZER_H__

#include <vector>

#include <realizer.h>

namespace nntrainer {

/**
 * @brief Graph realizer class which removes loss layer from the graph
 * @note This assumes the number of input / output connection of loss layer == 1
 *
 */
class TfliteExportRealizer final : public GraphRealizer {
public:
  /**
   * @brief Construct a new Loss Realizer object
   *
   */
  TfliteExportRealizer() = default;

  /**
   * @brief Destroy the Graph Realizer object
   *
   */
  ~TfliteExportRealizer() = default;

  /**
   * @brief graph realizer creates a shallow copied graph based on the reference
   * @note loss realizer removes loss layers from GraphRepresentation
   * @param reference GraphRepresentation to be realized
   *
   */
  GraphRepresentation realize(const GraphRepresentation &reference) override;

  /**
     * @brief graph realizer creates a shallow copied graph based on the reference
     * @note drop_out realizer removes drop_out layers from GraphRepresentation
     * @param reference GraphRepresentation to be realized
     *
  */
  GraphRepresentation realize_dropout(const GraphRepresentation &reference);

};

} // namespace nntrainer

#endif // __TFLITE_EXPORT_REALIZER_H__
