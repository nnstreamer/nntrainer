// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Eunju Yang <ej.yang@samsung.com>
 *
 * @file   layer_impl_cl.h
 * @date   04 Nov 2024
 * @brief  This is base Layer implementation class for OpenCL
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * @details LayerImplCl forms the base class for all the opencl_layer with
 * weights and bias parameters. LayerImpl provides parsing of properties like
 * Weight/bias initializer and regularizers. LayerImpl also provides checks for
 * double calls to finalize function. This is wrpper class of layer_impl for
 * OpenCL.
 */
#ifndef __LAYER_IMPL_CL_H__
#define __LAYER_IMPL_CL_H__
#ifdef __cplusplus

#include <cl_context.h>
#include <engine.h>
#include <layer_impl.h>

namespace nntrainer {

/**
 * @class LayerImplCl
 * @brief LayerImplCl
 */
class LayerImplCl : public LayerImpl {

public:
  /**
   * @brief     Constructor of Layer Class
   */
  NNTR_EXPORT LayerImplCl() : LayerImpl(){};

  /**
   * @brief     Destructor of Layer Class
   */
  NNTR_EXPORT virtual ~LayerImplCl() = default;

  /**
   *  @brief  Move constructor of LayerImpl Layer.
   *  @param[in] LayerImplCl &&
   */
  NNTR_EXPORT LayerImplCl(LayerImplCl &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs LayerImplCl to be moved.
   */
  NNTR_EXPORT LayerImplCl &operator=(LayerImplCl &&rhs) = default;

  /**
   * @brief     register ClKernels for this layer
   * registerClKernels() is called in global ClContext.
   */
  NNTR_EXPORT static bool registerClKernels();
};

} // namespace nntrainer

#endif /** __cplusplus */
#endif /** LAYER_IMPL_CL */
