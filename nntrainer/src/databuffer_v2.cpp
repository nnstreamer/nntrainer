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

void DataBuffer_v2::pushBatchedData() { throw std::runtime_error("NYI"); }

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

void DataBuffer_v2::init() {
  if (thread_state != ThreadStates::THREAD_NULL)
    throw std::runtime_error("Calling init on an already initialized dataset");
  if (type == DataBufferType::DATA_BUFFER_UNKNOWN)
    throw std::runtime_error("Data source has not been set for the dataset");

  for (auto const &size : input_size)
    if (size == 0)
      throw std::invalid_argument("Input size is not set for all the elements");

  for (auto const &size : label_size)
    if (size == 0)
      throw std::invalid_argument("Label size is not set for all the elements");

  if (type == DataBufferType::DATA_BUFFER_FILE) {
    throw std::runtime_error("NYI");
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
  thread_state = ThreadStates::THREAD_READY;
}

void DataBuffer_v2::stop() {
  /** wrap with thread lock */
  std::unique_lock<std::mutex> thread_stop(thread_m);

  if (thread_state == ThreadStates::THREAD_NULL)
    ml_logw("stopping an un-initialized dataset");
  else if (thread_state == ThreadStates::THREAD_REQUEST_TO_STOP ||
           thread_state == ThreadStates::THREAD_STOPPED)
    ml_logw("stopping a data buffer which is being stopped");
  else if (thread_state == ThreadStates::THREAD_READY)
    ml_logw("stopping an already stopped data buffer");

  /** request the thread to stop */
  if (thread_state == ThreadStates::THREAD_RUNNING) {
    thread_state = ThreadStates::THREAD_REQUEST_TO_STOP;

    /** wake up all the background threads */
    buffer_cond_hungry.notify_all();

    /** wait for thread to stop */
    buffer_cond_filled.wait(thread_stop, [this] {
      return thread_state != ThreadStates::THREAD_REQUEST_TO_STOP;
    });
  }

  /** wait for thread to join */
  joinDataCollectors();

  /** throw error in case of error */
  if (thread_state == ThreadStates::THREAD_ERROR)
    throw std::logic_error("Dataset background thread resulted in error");

  thread_state = ThreadStates::THREAD_READY;
  buffer.clear();
  batched_buffer.clear();
  avail_buffer_idx = 0;
}

void DataBuffer_v2::start() {
  {
    /**
     * Lock is needed here as the data buffer might already be running
     * or the stop called hasnt been finished
     */
    std::lock_guard<std::mutex> thread_start(thread_m);

    if (thread_state == ThreadStates::THREAD_NULL)
      throw std::runtime_error("starting an un-initilaized dataset");
    else if (thread_state == ThreadStates::THREAD_ERROR)
      throw std::runtime_error("starting a data buffer which is in error");
    else if (thread_state != ThreadStates::THREAD_READY)
      throw std::runtime_error("starting an already running data buffer");

    runDataCollectors();
    thread_state = ThreadStates::THREAD_RUNNING;
  }

  /** create buffer of size buffer_len */
  for (size_t batched_buflen = 0; batched_buflen < batched_buffer_len;
       batched_buflen++) {
    pushBatchedData();

    /** As soon as there are more data containers available, start the data
     * collectors */
    // TODO: move this notify inside the pushBatchedData() itself.
    buffer_cond_hungry.notify_all();
  }
}

int DataBuffer_v2::getData(std::vector<sharedTensor> &inputs,
                           std::vector<sharedTensor> &labels) {
  if (inputs.size() != input_size.size() || labels.size() != label_size.size())
    throw std::invalid_argument(
      "Number of inputs and labels mismatch with the set configuration");

  /** Ensure that the dataset is actually running */
  {
    std::lock_guard<std::mutex> thread_start(thread_m);
    if (thread_state != ThreadStates::THREAD_RUNNING &&
        thread_state != ThreadStates::THREAD_EPOCH_FINISHED)
      throw std::runtime_error("Calling get data on a thread not running");
  }

  /** Keep on looping till a batch of data is ready */
  while (true) {
    {
      std::lock_guard<std::mutex> thread_start(thread_m);
      if (thread_state == ThreadStates::THREAD_EPOCH_FINISHED)
        return false;
      else if (thread_state != ThreadStates::THREAD_RUNNING)
        throw std::logic_error("Background thread is not running anymore");
    }

    bool data_available;
    /** check if data is available */
    {
      std::lock_guard<std::mutex> batched_buffer_data_avail_check_lock(
        batched_buffer_m);
      auto batched_buffer_elem = batched_buffer.front();
      data_available = std::get<2>(batched_buffer_elem) == batch_size;
    }

    /**
     * Fill the data into inputs and labels, and remove from batch buffers
     * decrement avail buffer idx
     */
    if (data_available) {
      // TODO: remove the front element first, so that background threads have
      // to wait less
      /**
       * set inputs and labels data from batched buffer_elem
       * As this batch is already completed, background thread will not change
       * this data. So no batch lock is needed here. No thread lock is needed
       * here as the data buffer caller is our thread and it is assumed that
       * when getData() is being called, stop() wont be called.
       */
      auto batched_buffer_elem = batched_buffer.front();
      void **batched_inputs = std::get<0>(batched_buffer_elem);
      for (unsigned int idx = 0; idx < inputs.size(); idx++)
        inputs[idx]->setData(static_cast<float *>(batched_inputs[idx]));

      void **batched_labels = std::get<1>(batched_buffer_elem);
      for (unsigned int idx = 0; idx < labels.size(); idx++)
        labels[idx]->setData(static_cast<float *>(batched_labels[idx]));

      /** Remove the front consumed data from the buffer */
      {
        std::lock(buffer_m, batched_buffer_m);
        std::lock_guard<std::mutex> buffer_data_avail_lock(buffer_m,
                                                           std::adopt_lock);
        std::lock_guard<std::mutex> batched_buffer_data_avail_lock(
          batched_buffer_m, std::adopt_lock);
        batched_buffer.pop_front();
        for (unsigned int idx = 0; idx < batch_size; idx++)
          buffer.pop_front();
        avail_buffer_idx -= batch_size;

        if (avail_buffer_idx < 0 || avail_buffer_idx > buffer_len)
          throw std::logic_error("Buffer overflow error");
      }

      /** Add new space to the buffer */
      pushBatchedData();
      /** Wake up background threads waiting for more space to fill data */
      buffer_cond_hungry.notify_all();
      return true;
    }

    // TODO: optimize to wake up only when full batch is filled by the
    // background thread
    /** wait for background thread to fill the data */
    std::unique_lock<std::mutex> batched_buffer_wait_for_data(batched_buffer_m);
    buffer_cond_filled.wait(batched_buffer_wait_for_data);
  }

  throw std::logic_error("Reach unexpected code location");
}

void DataBuffer_v2::collectData() {
  bool last_element = false;
  while (!last_element) {
    bool buffer_space_available;
    std::tuple<void **, void **> buffer_elem;

    /**
     * THREAD_REQUEST_TO_STOP : requested by main thread to stop
     * THREAD_ERROR : thread is in error state, exit
     * THREAD_EPOCH_FINISHED : thread finished work, exit
     */
    {
      std::lock_guard<std::mutex> thread_stop(thread_m);
      if (thread_state != ThreadStates::THREAD_RUNNING) {
        thread_state = ThreadStates::THREAD_STOPPED;
        buffer_cond_filled.notify_all();
        return;
      }
    }

    /** check if buffer has space available to load the data */
    {
      std::lock_guard<std::mutex> buffer_data_fill_lock(buffer_m);
      buffer_space_available = buffer_len - avail_buffer_idx - 1 > 0;
      if (buffer_space_available) {
        buffer_elem = getNthBufferElement();
        avail_buffer_idx += 1;
      }
    }

    /** if the buffer space is received, then call generator cb and get the data
     */
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
        /** notify on buffer produce */
        buffer_cond_filled.notify_all();
      } else if (status == ML_ERROR_NONE) {
        /** Notify data finished as last_element was true */
        std::lock_guard<std::mutex> thread_stop(thread_m);
        thread_state = ThreadStates::THREAD_EPOCH_FINISHED;
        buffer_cond_filled.notify_all();
        return;
      } else {
        /** Return error as return state of generator was error */
        std::lock_guard<std::mutex> thread_stop(thread_m);
        thread_state = ThreadStates::THREAD_ERROR;
        buffer_cond_filled.notify_all();
        return;
      }
    } else {
      /**
       * wait for main thread to consume some data
       * @note this wake up has no check so that thread state conditions can be
       * checked and then buffer based conditions are checked.
       */
      std::unique_lock<std::mutex> buffer_wait_for_consume_lock(buffer_m);
      buffer_cond_hungry.wait(buffer_wait_for_consume_lock);
    }
  }
}

} // namespace nntrainer
