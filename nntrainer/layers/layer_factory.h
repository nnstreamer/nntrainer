// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file	layer_factory.h
 * @date	7 October 2020
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug		No known bugs except for NYI items
 * @brief	This is the layer factory.
 */

#ifndef __LAYER_FACTORY_H__
#define __LAYER_FACTORY_H__
#ifdef __cplusplus

#include <layer_internal.h>

namespace nntrainer {

/**
 * @brief Factory creator with constructor
 */
std::unique_ptr<Layer> createLayer(LayerType type);

} /* namespace nntrainer */

#endif /* __cplusplus */
#endif /* __LAYER_FACTORY_H__ */
