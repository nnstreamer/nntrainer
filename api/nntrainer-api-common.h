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

#ifndef __TIZEN_MACHINELEARNING_NNTRAINER_API_COMMON_H__
#define __TIZEN_MACHINELEARNING_NNTRAINER_API_COMMON_H__

/**
 * @addtogroup CAPI_ML_NNTRAINER_TRAIN_MODULE
 * @{
 */

/**
 * @brief Dataset generator callback function for train/valid/test data.
 *
 * @details The user of the API must provide this callback function to supply
 * data to the model and register the callback with
 * ml_train_dataset_create_with_generator(). The model will call this callback
 * whenever it needs more data. This function should provide a single element of
 * input and label data in the passed containers. The containers passed by the
 * caller will already be allocated with sufficient space to contain the data.
 * This function callback should fill the data row-wise in the containers
 * provided. The containers represent array of memory to hold inputs for the
 * model. If the model contains two inputs, then input[0] will hold the first
 * input, and input[1] will hold the second input. The same applies for labels
 * as well. The number of inputs and labels, and the size of each input and
 * label should match with the shape of each input and label set in the model.
 * The order of the inputs/labels, in case of multiple of inputs/labels, will be
 * determined based on the sequence of addition of the input layers to the
 * model.
 * @since_tizen 6.0
 * @note This function can be called multiple times in parallel when total
 * number of samples are set as a property for this dataset. In this case, last
 * is only used for verification purposes. If total number of samples for the
 * dataset is unknown, this function will be called in sequence.
 * @param[out] input Container to hold all the input data. Should not be freed
 * by the user.
 * @param[out] label Container to hold corresponding label data. Should not be
 * freed by the user.
 * @param[out] last Container to notify if data is finished. Set true if no more
 * data to provide, else set false. Should not be freed by the user.
 * @param[in] user_data User application's private data.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_INVALID_PARAMETER Invalid parameter.
 *
 * A sample implementation of this function is below:
 * Note : input data information available from outside this function.
 * num_samples : total number of data samples in the dataset.
 * count : number of samples already given.
 * num_inputs : number of inputs.
 * num_labels : number of labels.
 * input_length[num_inputs] : length of the input. With (batch, c, h, w) as
 *                            input shape, the length will be c * h * w.
 * label_length[num_labels] : length of the label. With (batch, l) as label
 *                            shape, then length will be l.
 * @code
 * // function signature :
 * // int rand_dataset_generator (float **input, float **label, bool *last,
 * // void *user_data).
 *
 * // This sample fills inputs and labels with random data.
 * srand(0);
 * if (count > num_samples) {
 *   *last = true;
 *   // handle preparation for start of next epoch
 *   return ML_ERROR_NONE;
 * }
 *
 * // Fill input data
 * for (int idx = 0; idx < num_inputs; ++ idx) {
 *   for (int len = 0; len < input_length[idx]; ++ len) {
 *     input[idx][len] = rand();
 *   }
 * }
 *
 * // Fill label data
 * for (int idx = 0; idx < num_inputs; ++ idx) {
 *   for (int len = 0; len < label_length[idx]; ++ len) {
 *     label[idx][len] = rand();
 *   }
 * }
 *
 * // Update the helper variables
 * count += 1;
 *
 * return ML_ERROR_NONE;
 * @endcode
 *
 *
 * Below is an example of the usage of this sample:
 * @code
 * int status;
 * ml_train_dataset_h handle;
 *
 * status = ml_train_dataset_create_with_generator(&handle,
 *      rand_dataset_generator, NULL, NULL);
 * if (status != ML_ERROR_NONE) {
 *    // handle error case.
 *    return status;
 * }
 *
 * // Destroy the handle if not added to a model.
 * status = ml_train_layer_destroy(handle);
 * if (status != ML_ERROR_NONE) {
 *    // handle error case
 *    return status;
 * }
 * @endcode
 */
typedef int (*ml_train_datagen_cb)(float **input, float **label, bool *last,
                                   void *user_data);

/**
 * @brief Enumeration for the neural network summary verbosity of NNTrainer.
 * @since_tizen 6.0
 */
typedef enum {
  ML_TRAIN_SUMMARY_MODEL = 0, /**< Overview of model
                                   summary with one-line layer information */
  ML_TRAIN_SUMMARY_LAYER, /**< Detailed model summary with layer properties */
  ML_TRAIN_SUMMARY_TENSOR /**< Model summary layer's including weight
                             information */
} ml_train_summary_type_e;

/**
 * @}
 */

#endif /* __TIZEN_MACHINELEARNING_NNTRAINER_API_COMMON_H__ */
