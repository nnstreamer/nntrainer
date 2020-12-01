// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file	manager.h
 * @date	30 Nov 2020
 * @brief	This is NNtrainer manager for all weights, i/o and intermediate
 * tensors
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __MANAGER_H__
#define __MANAGER_H__
#ifdef __cplusplus

#include <vector>

#include <weight.h>

namespace nntrainer {

/**
 * @class   Manager
 * @brief   manager of nntrainer
 */
class Manager {
public:
  /**
   * @brief     Constructor of Manager
   */
  Manager() {}

  /**
   * @brief     Destructor of Manager
   */
  ~Manager() {}

  /**
   * @brief     Add weight to be tracked and updated with nntrainer
   *
   * @param w   Weight to be tracked
   */
  void trackWeight(Weight w) { weights.push_back(w); }

  /**
   * @brief     Add weights to be tracked and updated with nntrainer
   *
   * @param ws  Weights to be tracked
   */
  void trackWeights(std::vector<Weight> &ws) {
    weights.reserve(weights.size() + ws.size());
    weights.insert(weights.end(), ws.begin(), ws.end());
  }

  /**
   * @brief     Get weights tracked with nntrainer
   *
   * @retval    list of weights
   */
  std::vector<Weight> getWeights() { return weights; }

private:
  // TODO: ensure that names of these weights are unique
  std::vector<Weight> weights;
};

/**
 * @brief Helper func for weight creation which are tracked by nntrainer
 *
 * @retval create weight
 */
template <typename... Args>
Weight createWeight(Manager &manager, Args... args) {
  Weight w = Weight(args...);
  manager.trackWeight(w);
  return w;
}

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __MANAGER_H__ */
