// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   optimizer_common_properties.cpp
 * @date   17 May 2024
 * @brief  This file contains common properties for optimizer
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */
#include <optimizer_common_properties.h>

#include <nntrainer_log.h>
#include <util_func.h>

namespace nntrainer::props {

Rho::Rho(double value) { set(value); };

PropsEpsilon::PropsEpsilon(double value) { set(value); };

TorchRef::TorchRef(bool value) { set(value); };

} // namespace nntrainer::props
