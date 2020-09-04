// SPDX-License-Identifier: Apache-2.0-only
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file	databuffer_v2.cpp
 * @date	3 September 2020
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug		No known bugs except for NYI items
 * @brief	This is databuffer class for Neural Network
 */

#include <databuffer_v2.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>

namespace nntrainer {

void *incVoidPtr(void *ptr, size_t bytes) {
  return static_cast<char *>(ptr) + bytes;
}

void **allocateBatchedDataHolder(const std::vector<size_t> &data_size,
                                 const unsigned int batch_size) {
  void **data = (void **)malloc(sizeof(void *) * data_size.size());

  for (size_t idx = 0; idx < data_size.size(); idx++)
    data[idx] = malloc(data_size[idx] * batch_size);

  return data;
}

void deallocateDataHolder(const std::vector<size_t> &data_size, void **data) {
  for (size_t i = 0; i < data_size.size(); i++)
    free(data[i]);

  free(data);
}

// void DataBuffer_v2::pushBatchedData() {
//     std::vector<void *> batched_input_holder =
//     allocateBatchedDataHolder(input_size); std::vector<void *>
//     batched_label_holder = allocateBatchedDataHolder(label_size);
//
//     batched_buffer.push_back(std::make_pair(batched_input_holder,
//     batched_label_holder)); for (unsigned int b = 0; b < batch_size; b++) {
//       unsigned int buf_idx = batched_buflen * batch_size + b;
//       std::vector<void *> input_holder;
//       for (auto const in&: batched_input_holder):
//         input_holder.push_back(invVoidPtr(in[]))
//       buffer.push_back(std::make_pair(input_holder));
//     }
// }

void DataBuffer_v2::setDataSource(datagen_cb gen_cb, void *user_data) {
  if (type == DataBufferType::DATA_BUFFER_FILE)
    throw std::invalid_argument("setting generator for file based dataset");

  if (gen_cb == nullptr)
    throw std::invalid_argument("callback function is null");

  if (generator)
    ml_logw("Overwriting existing data generator source.");

  type = DataBufferType::DATA_BUFFER_GENERATOR;
  generator = gen_cb;
  gen_user_data = user_data;
}

void DataBuffer_v2::setDataSource(const std::string file) {
  DataBufferFileUserData *file_usr_data;
  if (type == DataBufferType::DATA_BUFFER_GENERATOR)
    throw std::invalid_argument("setting file for generator based dataset");

  if (gen_user_data) {
    ml_logw("Overwriting existing data file source.");
    file_usr_data = (DataBufferFileUserData *)gen_user_data;
    delete file_usr_data;
  }

  type = DataBufferType::DATA_BUFFER_FILE;
  generator = file_dat_cb;
  file_usr_data = new DataBufferFileUserData(file);
  gen_user_data = static_cast<void *>(file_usr_data);
}

void DataBuffer_v2::init() {
  if (type == DataBufferType::DATA_BUFFER_UNKNOWN)
    throw std::runtime_error("Data source has not been set for the dataset");

  for (auto const &size : input_size)
    if (size == 0)
      throw std::invalid_argument("Input size is not set for all the elements");

  for (auto const &size : label_size)
    if (size == 0)
      throw std::invalid_argument("Label size is not set for all the elements");

  if (type == DataBufferType::DATA_BUFFER_FILE) {
    DataBufferFileUserData *file_usr_data =
      (DataBufferFileUserData *)gen_user_data;
    file_usr_data->setInputLabelSize(input_size, label_size);
    total_data_entries =
      file_usr_data->size / file_usr_data->getSingleDataSize();
  }

  if (type == DataBufferType::DATA_BUFFER_FILE &&
      batch_size > total_data_entries)
    throw std::runtime_error("Batch size exceeds the total datasize");

  if (buffer_len > total_data_entries)
    buffer_len = total_data_entries;

  if (buffer_len < batch_size)
    buffer_len = batch_size;

  /**
   * batch size data is allocated at once in the buffer for data to avoid
   * copying to data back into another holder which has contiguous data.
   * This limits the buffer length to be a multiple of batch size.
   */
  batched_buffer_len = (buffer_len / batch_size);
  buffer_len = batched_buffer_len * batch_size;
}

void DataBuffer_v2::start() {
  {
    /// wrap buffer lock
    // TODO: this lock can be ommited as not thread running?? check case of
    // start again after stop
    std::lock(buffer_m, batched_buffer_m);
    std::lock_guard<std::mutex> buffer_init_lock(buffer_m, std::adopt_lock);
    std::lock_guard<std::mutex> batched_buffer_init_lock(batched_buffer_m,
                                                         std::adopt_lock);

    if (started)
      return;

    started = true;
    buffer.clear();
    batched_buffer.clear();
    avail_buffer_idx = 0;
  }

  /** create buffer of size buffer_len */
  for (size_t batched_buflen = 0; batched_buflen < batched_buffer_len;
       batched_buflen++) {
    pushBatchedData();
    if (batched_buflen == 0)
      collect_thread = std::thread(&DataBuffer_v2::collectData, this);
  }
}

void DataBuffer_v2::feedData(std::vector<Tensor> &inputs,
                             std::vector<Tensor> &labels) {
  start();

  while (true) {
    bool data_available;
    /** check if data is available */
    {
      std::lock_guard<std::mutex> batched_buffer_data_avail_check_lock(
        batched_buffer_m);
      auto batched_buffer_elem = batched_buffer.front();
      data_available = std::get<2>(batched_buffer_elem) == batch_size;
    }

    if (data_available) {
      /** Fill the data into inputs and labels, and remove from batch buffers */
      // TODO: set inputs data from batched buffer_elem

      {
        std::lock(buffer_m, batched_buffer_m);
        std::lock_guard<std::mutex> buffer_data_avail_lock(buffer_m,
                                                           std::adopt_lock);
        std::lock_guard<std::mutex> batched_buffer_data_avail_lock(
          batched_buffer_m, std::adopt_lock);
        // TODO: remove front 1 batched_buffer_elem
        // TODO: remove front batch_size buffer_elem
      }

      // TODO: notify all the producers
    }

    // TODO: wait for buffer produce notify
  }
}

void DataBuffer_v2::collectData() {
  bool last_element = false;
  while (!last_element) {
    bool buffer_space_available;
    std::tuple<void **, void **> buffer_elem;
    // TODO: check for stop sent by the main thread

    // check if buffer has space available to load the data
    {
      std::lock_guard<std::mutex> buffer_data_fill_lock(buffer_m);
      buffer_space_available = buffer_len - avail_buffer_idx - 1 > 0;
      if (buffer_space_available) {
        buffer_elem = getNthBufferElement();
        avail_buffer_idx += 1;
      }
    }

    if (buffer_space_available) {
      void **input_holder = std::get<0>(buffer_elem);
      void **label_holder = std::get<1>(buffer_elem);

      int status = generator((float **)input_holder, (float **)label_holder,
                             &last_element, gen_user_data);

      if (status != ML_ERROR_NONE && !last_element) {
        /** Update the batched buffer element loaded count */
        std::lock_guard<std::mutex> batched_buffer_update_lock(
          batched_buffer_m);
        auto batched_buffer_elem = getNthBatchedBufferElement();
        std::get<2>(batched_buffer_elem) += 1;
        // TODO: notify on buffer produce
      } else if (status == ML_ERROR_NONE) {
        // TODO: data epoch completion notify
      } else {
        // TODO: error notify
      }
    } else {
      // TODO: wait for buffer consume notify
    }
  }
}

} // namespace nntrainer
