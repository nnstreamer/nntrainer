// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file remap_realizer.h
 * @date 12 October 2021
 * @brief NNTrainer graph realizer which realizes identifier to a new identifier
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
#ifndef __REMAP_REALIZER_H__
#define __REMAP_REALIZER_H__

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <realizer.h>

namespace nntrainer {

/**
 * @brief Graph realizer class which remaps identifiers inside the graph
 * representation, remap_function will be applied for all the layers identifier
 * visible
 *
 */
class RemapRealizer final : public GraphRealizer {
public:
  /**
   * @brief Construct a new Remap Realizer object (connection mode)
   *
   * @param remap_connection_function remap connection function, with this
   * constructor, only connections will be remapped eg) if you are inserting a
   * new layer node in between, only connections need remapping
   */
  RemapRealizer(
    std::function<void(std::string & /**< identifier */,
                       unsigned & /**< index of a connection, remapping
                                     identifier should not modify this */)>
      remap_connection_function);

  /**
   * @brief Construct a new Remap Realizer object (identifier mode)
   *
   * @param remap_function remap function, with this constructor, all
   * identifiers wherever it is used will be remapped
   */
  RemapRealizer(
    std::function<void(std::string & /**< identifier */)> remap_function);

  /**
   * @brief Destroy the Graph Realizer object
   *
   */
  ~RemapRealizer();

  /**
   * @brief graph realizer creates a new graph based on the reference
   *
   */
  GraphRepresentation realize(const GraphRepresentation &reference) override;

private:
  std::function<void(std::string &)> remap_fn;
  std::function<void(std::string &, unsigned &)> remap_connection_fn;
};

} // namespace nntrainer

#endif // __REMAP_REALIZER_H__
