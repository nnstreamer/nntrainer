// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   layer_factory.h
 * @date   7 October 2020
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is the layer factory.
 */

#ifndef __LAYER_FACTORY_H__
#define __LAYER_FACTORY_H__
#ifdef __cplusplus

#include <layer_devel.h>
#include <layer.h>

namespace nntrainer {

using LayerType = ml::train::LayerType;

/**
 * @brief get string representation type from integer
 *
 * @param type integer type
 * @return const std::string string represented type
 */
const std::string layerGetStrType(const LayerType &type);

/**
 * @brief Layer factory creator with constructor
 */
std::unique_ptr<Layer> createLayer(const std::string &type);

/**
 * @brief Loss Layer Factory creator with constructor
 */
// std::unique_ptr<Layer> createLoss(LossType type);

} /* namespace nntrainer */

#endif /* __cplusplus */
#endif /* __LAYER_FACTORY_H__ */
