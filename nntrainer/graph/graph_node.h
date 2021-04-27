// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   graph_node.h
 * @date   1 April 2021
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is the graph node interface for c++ API
 */

#ifndef __GRAPH_NODE_H__
#define __GRAPH_NODE_H__

#include <memory>
#include <string>
#include <vector>

namespace nntrainer {

/**
 * @class   Layer Base class for the graph node
 * @brief   Base class for all layers
 */
class GraphNode {
public:
  /**
   * @brief     Destructor of Layer Class
   */
  virtual ~GraphNode() = default;

  /**
   * @brief     Get index of the node
   *
   */
  virtual size_t getIndex() = 0;

  /**
   * @brief     Set index of the node
   *
   */
  // virtual void setIndex(size_t) = 0;

  /**
   * @brief     Get the Name of the underlying object
   *
   * @return std::string Name of the underlying object
   * @note name of each node in the graph must be unique
   */
  virtual std::string getName() noexcept = 0;

  /**
   * @brief     Set the Name of the underlying object
   *
   * @param[in] std::string Name for the underlying object
   * @note name of each node in the graph must be unique
   *
   * @todo make it work with setProperty
   */
  // virtual int setName(std::string name) = 0;

  /**
   * @brief     Get the trainable property of the underlying object
   *
   * @return boolean true if trainable, else false
   */
  virtual bool getTrainable() noexcept = 0;

  /**
   * @brief     Set the properties for the node
   *
   * @param[in] properties properties of the node
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   *
   * @note this shouldn't be virtual, this became virtual to support custom
   * layer. should be reverted after layer.h can fully support custom layer
   */
  virtual int setProperty(std::vector<std::string> properties) = 0;

  /**
   * @brief     Get the Type of the underlying object
   *
   * @return const std::string type representation
   */
  virtual const std::string getType() const = 0;
};

} // namespace nntrainer
#endif // __GRAPH_NODE_H__
