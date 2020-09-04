// SPDX-License-Identifier: Apache-2.0-only
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file	databuffer_v2.h
 * @date	3 September 2020
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug		No known bugs except for NYI items
 * @brief	This is databuffer class for Neural Network
 *
 * @todo TODO: Support multi files for dataset with files
 * @todo TODO: Support multi threads with more than 1 thread and use thread
 * pooling
 * @todo TODO: Support label size to be 0 for inference based scenarios
 * @todo TODO: rename data buffer to dataset
 * @todo TODO: move databufferfileuserdata to databuffer_file_cb.h
 * @todo TODO: manage with just 1 buffer
 */

#ifndef __DATABUFFER_V2_H__
#define __DATABUFFER_V2_H__
#ifdef __cplusplus

#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include <databuffer_file_cb.h>
#include <nntrainer-api-common.h>
#include <tensor.h>

namespace nntrainer {

/**
 * @brief   Dataset generator callback type declaration
 */
typedef std::function<std::remove_pointer<ml_train_datagen_cb>::type>
  datagen_cb;

/**
 * @class   DataBuffer Data Buffers
 * @brief   Data Buffer for read and manage data
 */
class DataBuffer_v2 {
public:
  /**
   * @brief     Constructor
   */
  DataBuffer_v2() :
    type(DataBufferType::DATA_BUFFER_UNKNOWN),
    buffer_len(1),
    avail_buffer_idx(0),
    total_data_entries(0),
    batch_size(1),
    num_threads(1),
    generator(nullptr),
    gen_user_data(nullptr),
    started(false) {
    buffer.clear();
    label_size.resize(1, 0);
    input_size.resize(1, 0);
  }

  /**
   * @brief     Destructor
   */
  ~DataBuffer_v2() {
    // TODO: signal thread to exit and wait for it to join
    if (type == DataBufferType::DATA_BUFFER_FILE) {
      DataBufferFileUserData *file_user_data =
        (DataBufferFileUserData *)gen_user_data;
      delete file_user_data;
    }
    buffer.clear();
  }

  /**
   * @brief     Initialize Buffer with set properties
   * @throws std::invalid_argument
   * @throws std::runtime_error
   */
  void init();

  /**
   * @brief     Update Data Buffer
   * @throws std::runtime_error
   */
  void collectData();

  /**
   * @brief     start the thread for collection the data
   * @throws std::runtime_error
   */
  void start();

  /**
   * @brief     function for thread to collect the data
   * @throws std::runtime_error
   */
  void stop() {
    started = false;
    throw std::runtime_error("NYI");
  }

  /**
   * @brief     get Data from Data Buffer using databuffer param
   * @param[in] inputs list of input tensors
   * @param[in] inputs list of label tensors
   */
  void feedData(std::vector<Tensor> &inputs, std::vector<Tensor> &labels);

  /**
   * @brief     set the number of inputs (defaults to 1)
   * @param[in] num_inputs number of inputs
   * @throws std::invalid_argument
   */
  void setNumInputs(const unsigned int num_inputs = 1) {
    if (num_inputs == 0)
      throw std::invalid_argument("Number of inputs must be at least 1");

    if (num_inputs != input_size.size()) {
      input_size.resize(num_inputs, 0);
    }
  }

  /**
   * @brief     sets the number of labels (defaults to 1)
   * @param[in] num_labels number of labels
   * @throws std::invalid_argument
   */
  void setNumLabels(const unsigned int num_labels = 1) {
    if (num_labels == 0)
      throw std::invalid_argument("Number of labels must be at least 1");

    if (num_labels != label_size.size()) {
      label_size.resize(num_labels, 0);
    }
  }

  /**
   * @brief     set the size of the label data
   * @param[in] bytes size in bytes
   * @param[in] idx index of the label
   * @throws std::invalid_argument
   */
  void setLabelSize(const size_t bytes, const unsigned int idx = 0) {
    if (bytes == 0)
      throw std::invalid_argument("Label size should be more than 0");

    if (idx >= label_size.size()) {
      if (label_size.size() > 1)
        throw std::invalid_argument(
          "Index exceeds the total size set for the label");
      label_size.resize(idx + 1);
    }

    label_size[idx] = bytes;
  }

  /**
   * @brief     set buffer size
   * @param[in] n number of entries of data loaded in memory
   * @throws std::invalid_argument
   */
  void setBufferSize(const size_t n) {
    if (n == 0)
      throw std::invalid_argument("Buffer size should be more than 0");
    buffer_len = n;
  }

  /**
   * @brief     set the size of the input data
   * @param[in] bytes size in bytes
   * @param[in] idx index of the label
   * @throws std::invalid_argument
   */
  void setInputSize(const size_t bytes, const unsigned int idx = 0) {
    if (bytes == 0)
      throw std::invalid_argument("Input size should be more than 0");

    if (idx >= input_size.size()) {
      if (label_size.size() > 1)
        throw std::invalid_argument(
          "Index exceeds the total size set for the input");
      input_size.resize(idx + 1);
    }

    input_size[idx] = bytes;
  }

  /**
   * @brief     set batch size
   * @param[in] n batch size
   * @throws std::invalid_argument
   */
  void setBatchSize(const unsigned int n) {
    if (n == 0)
      throw std::invalid_argument("Batch size should be more than 0");
    batch_size = n;
  }

  /**
   * @brief     get the total number of batches in the dataset
   * @retval    number of batches in this dataset
   * @throws std::runtime_error
   */
  size_t getTotalNumBatches() const {
    if (type != DataBufferType::DATA_BUFFER_FILE)
      throw std::runtime_error("Getting total number of batches in the dataset "
                               "is only supported for file based dataset");

    if (total_data_entries == 0)
      throw std::runtime_error(
        "Total number of batches in dataset is available after init");

    return total_data_entries / batch_size;
  }

  /**
   * @brief     set property
   * @param[in] values values of property
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int setProperty(const std::vector<std::string> values);

  /**
   * @brief     set function pointer as the data source
   * @param[in] gen_cb call back function pointer
   * @param[in] user_data users private data to be passed to the cb
   * @throws std::invalid_argument
   */
  void setDataSource(datagen_cb gen_cb, void *user_data);

  /**
   * @brief     set data file path
   * @param[in] path file path
   * @throws std::invalid_argument
   */
  void setDataSource(const std::string file);

  /**
   * @brief Enumeration for the properties supported by data buffer
   * TODO: update these
   */
  enum class PropertyType { data = 0, buffer_len = 4, unknown = 5 };

  /**
   * @brief Get the nth buffer element
   * @note This function must be called after acquiring the lock on the buffer
   */
  inline auto getNthBufferElement() {
    auto it = buffer.begin();
    std::advance(it, avail_buffer_idx);
    return *it;
  }

  /**
   * @brief Get the nth buffer batched element
   * @note This function must be called after acquiring the lock on the buffer
   * batched
   */
  inline auto getNthBatchedBufferElement() {
    auto it = batched_buffer.begin();
    std::advance(it, avail_buffer_idx / batch_size);
    return *it;
  }

  /**
   * @brief Push a batch of data holder onto the buffer
   * @note This must NOT be called while holding the buffer/batched_buffer lock
   * as this function acquires both the locks internally
   */
  void pushBatchedData() { throw std::runtime_error("NYI"); }

private:
  /**
   * @brief     Enumeration for data buffer type
   */
  enum class DataBufferType {
    DATA_BUFFER_GENERATOR, /**< Data collected from a generator function */
    DATA_BUFFER_FILE,      /**< Data collected from a set of files */
    DATA_BUFFER_UNKNOWN    /**< Unknown data collection setup */
  };

  DataBufferType type; /**< Type of the data buffer */
  std::list<std::tuple<void **, void **>>
    buffer; /**< Buffer to capture the data */
  std::list<std::tuple<void **, void **, unsigned int>>
    batched_buffer; /**< Buffer to capture the data */
  std::vector<size_t> label_size,
    input_size;      /**< size of all inputs and labels */
  size_t buffer_len; /**< max length of the buffer, limits the total number of
                        data entries loaded into the memory */
  size_t avail_buffer_idx;   /**< idx of the buffer entry which is empty */
  size_t batched_buffer_len; /**< maximum number of batches that fits in the
                                buffer. */
  size_t total_data_entries; /**< total number of data points in this dataset. 0
                                means it cant be interpreted. */
  unsigned int batch_size; /**< batch size of single data element to be returned
                              by the dataset */
  unsigned int num_threads; /**< number of parallel threads for data loading */
  datagen_cb generator;     /**< generator callback for data production */
  void *gen_user_data; /**< user private data to be given to the data generator
                          callback */
  bool started;        /**< collect data thread has started */

  std::thread collect_thread;            /**< data collection thread */
  std::mutex buffer_m, batched_buffer_m; /**< mutex locks for buffers */
};

/**
 * @brief Increment the given pointer by bytes size
 */
void *incVoidPtr(void *ptr, size_t bytes);

/**
 * @brief Allocate memory for given list of data size
 * @TODO: use an external allocator which can allocate memory for the buffers
 */
void **allocateBatchedDataHolder(const std::vector<size_t> &data_size,
                                 const unsigned int batch_size);

/**
 * @brief Deallocate passed memory with the given data size
 */
void deallocateDataHolder(const std::vector<size_t> &data_size, void **data);

} // namespace nntrainer
#endif /* __cplusplus */
#endif /* __DATABUFFER_V2_H__ */
