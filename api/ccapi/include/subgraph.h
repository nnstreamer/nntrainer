// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   subgraph.h
 * @brief  This is subgraph interface for c++ API
 * @date   11 Mar 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug	   No known bugs except for NYI items
 *
 * @note This is experimental API and not stable.
 */

#ifndef __ML_TRAIN_SUBGRAPH_H__
#define __ML_TRAIN_SUBGRAPH_H__

#if __cplusplus < MIN_CPP_VERSION
#error "CPP versions c++17 or over are only supported"
#endif // __cpluscplus

#include <memory>
#include <string>
#include <tensor_dim.h>
#include <vector>

#include <common.h>
#include <layer.h>

namespace ml {
namespace train {

enum SubGraphType { SUBGRAPH_CPU };

/**
 * @class   SubGraph Base class for subgraphs
 * @brief   Base class for all subgraphs
 */
class SubGraph {
public:
  /**
   * @brief Destructor of SubGraph Class
   */
  virtual ~SubGraph() = default;

  /**
   * @brief Get the subgraph type
   * @return const std::string type representation
   */
  virtual const std::string getType() const = 0;

  /**
   * @brief     set Property of SubGraph
   * @todo      change the function signature
   * @param[in] values values of property
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   * @details   This function accepts vector of properties in the format -
   *  { std::string property_name, void * property_val, ...}
   */
  virtual void setProperty(const std::vector<std::string> &values) = 0;

  /**
   * @brief     Get name of the subgraph
   * @retval    name of the subgraph
   * @note      This name is unique to this layer in a model
   * @note      This name might be changed once this layer is added to the model
   * to keep the name unique to the model
   */
  virtual const std::string getName() const = 0;

  /**
   * @brief Create new LayerNode and add into SubGraph
   * @param[in] layer shared_ptr of Layer
   */
  virtual void addLayer(std::shared_ptr<Layer> layer) = 0;
};

/**
 * @brief Factory creator with constructor for subgraph type
 */
std::unique_ptr<SubGraph>
createSubGraph(const SubGraphType &type,
               const std::vector<std::string> &properties = {});

/**
 * @brief Factory creator with constructor for SubGraph
 */
std::unique_ptr<SubGraph>
createSubGraph(const std::string &type,
               const std::vector<std::string> &properties = {});

/**
 * @brief General SubGraph Factory function to register SubGraph
 *
 * @param props property representation
 * @return std::unique_ptr<ml::train::SubGraph> created object
 */
template <typename T,
          std::enable_if_t<std::is_base_of<SubGraph, T>::value, T> * = nullptr>
std::unique_ptr<SubGraph>
createSubGraph(const std::vector<std::string> &props = {}) {
  std::unique_ptr<SubGraph> ptr = std::make_unique<T>();
  ptr->setProperty(props);
  return ptr;
}

/**
 * Aliases for various subgraphs
 */
namespace subgraph {
/**
 * @brief Helper function to create input layer
 */
inline std::unique_ptr<SubGraph>
SubGraphCpu(const std::vector<std::string> &properties = {}) {
  return createSubGraph(SubGraphType::SUBGRAPH_CPU, properties);
}
} // namespace subgraph

} // namespace train
} // namespace ml
#endif // __ML_TRAIN_LAYER_H__
