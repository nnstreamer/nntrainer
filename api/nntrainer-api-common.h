// SPDX-License-Identifier: Apache-2.0-only
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file nntrainer-api-common.h
 * @date 02 April 2020
 * @brief NNTrainer Common-API Header.
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug No known bugs except for NYI items
 */

/**
 * @brief Dataset generator callback function for train/valid/test data.
 * @details The Containers passed will already be allocated with sufficient
 * space to contain the data by the caller. This function should return a batch
 * of input and label of data in the passed containers. The callback should fill
 * the data row-wise in the containers obliging to the input shape set for the
 * model. The order of the inputs in case of multiple input layers will be
 * determined based on the sequence of addition of the input layers to the
 * model.
 * @note This function can be called multiple times in parallel.
 * @param[out] input Container to hold all the input data.
 * @param[out] label Container to hold corresponding label data.
 * @param[out] last Container to notify if data is finished. Set true if no more
 * data to provide, else set false.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_INVALID_PARAMETER Invalid Parameter.
 */
typedef int (*ml_train_datagen_cb)(float **input, float **label, bool *last);
