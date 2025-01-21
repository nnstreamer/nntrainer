// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>
 *
 * @file    subgraph.h
 * @date    07 Jan 2025
 * @see     https://github.com/nnstreamer/nntrainer
 * @author  Eunju Yang <ej.yang@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   This is a Network SubGraph Class for Neural Network
 */
#ifndef __SUBGRAPH_H__
#define __SUBGRAPH_H__
#ifdef __cplusplus

#include <compiler_fwd.h>
#include <subgraph_base.h>
#include <subgraph_cpu.h>

namespace nntrainer {

class SubGraphBase;
class SubGraphCpu;

#define SGNODE(x) std::static_pointer_cast<SubGraphBase>(x)

/**
 * @brief LayerNode creator with constructor
 *
 * @params[in] properties Properties of the layer
 */
SubGraphNode createSubGraph(const std::vector<std::string> &properties = {});

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __SUBGRAPH_H__ */
