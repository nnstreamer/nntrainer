// SPDX-License-Identifier: Apache-2.0-only
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file nntrainer_doc.h
 * @date 10 July 2020
 * @brief Tizen C-API Declaration for Tizen SDK
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug No known bugs except for NYI items
 */

#ifndef __TIZEN_MACHINELEARNING_NNTRAINER_DOC_H__
#define __TIZEN_MACHINELEARNING_NNTRAINER_DOC_H__

/**
 * @ingroup  CAPI_ML_FRAMEWORK
 * @defgroup CAPI_ML_NNTRAINER_TRAIN_MODULE Trainer
 * @addtogroup CAPI_ML_NNTRAINER_TRAIN_MODULE
 * @brief The NNTrainer function provides interfaces to create and train Machine
 * Learning models on the device locally.
 * @section CAPI_ML_NNTRAINER_TRAIN_HEADER Required Header
 *   \#include <nntrainer/nntrainer.h>\n
 * @section CAPI_ML_NNTRAINER_TRAIN_OVERVIEW Overview
 * The NNTrainer API provides interfaces to create and train Machine
 * Learning models on the device locally.
 *
 *  This function allows the following operations with NNTrainer:
 * - Interfaces to create a machine learning predefined model or from scratch.
 * - Create/destroy and add new layers to the model.
 * - Create/destroy and set optimizer to the model.
 * - Interfaces to set datasets to feed data to the model.
 * - Summarize the model with the set configuration.
 * - Interfaces to compile and run the model.
 * - Utility functions to set properties for the models and its various
 * sub-parts.
 *
 *  Note that this function set is supposed to be thread-safe.
 *
 * @section CAPI_ML_NNTRAINER_TRAIN_FEATURE Related Features
 * This function is related with the following features:\n
 *  - %http://tizen.org/feature/machine_learning\n
 *  - %http://tizen.org/feature/machine_learning.training\n
 *
 * It is recommended to probe features in your application for reliability.\n
 * You can check if a device supports the related features for this function by
 * using
 * @ref CAPI_SYSTEM_SYSTEM_INFO_MODULE, thereby controlling the procedure of
 * your application.\n
 * To ensure your application is only running on the device with specific
 * features, please define the features in your manifest file using the manifest
 * editor in the SDK.\n
 * For example, your application accesses to the camera device,
 * then you have to add %http://tizen.org/privilege/camera into the manifest of
 * your application.\n More details on featuring your application can be found
 * from <a
 * href="https://docs.tizen.org/application/tizen-studio/native-tools/manifest-text-editor#feature-element">
 *    <b>Feature Element</b>.
 * </a>
 */

#endif /* __TIZEN_MACHINELEARNING_NNTRAINER_DOC_H__ */
