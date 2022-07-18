// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2022 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file nntr_threads.h
 * @date 07 July 2022
 * @brief Thread Management for NNTrainer
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug No known bugs except for NYI items
 */
#ifndef __NNTR_THREADS_H__
#define __NNTR_THREADS_H__

#include <string>
#include <thread>
#include <vector>

#include <common_properties.h>
#include <nntrainer_error.h>
#include <util_func.h>

typedef void (*loop_cb)(unsigned int start, unsigned int end, void *user_data);

typedef std::function<std::remove_pointer<loop_cb>::type> threaded_cb;

namespace nntrainer {

/**
 * @brief ParallelBatch class to parallelize along batch direction
 *
 */
class ParallelBatch {
public:
  /**
   * @brief Construct a new ParallelBatch object
   * @param unsigned int total number of batch
   *
   */

  ParallelBatch(unsigned int batch);

  /**
   * @brief Construct a new ParallelBatch object
   * @param threaded_cb the function run in thread
   * @param unsigned int total number of batch
   * @param void* user data for the threaded callback function
   *
   */
  ParallelBatch(threaded_cb threaded_cb_, unsigned int batch, void *user_data_);

  /**
   * @brief Destroy the ParallelBatch object
   *
   */
  ~ParallelBatch();

  /**
   * @brief Run the workders
   *
   */
  void run();

  /**
   * @brief set the thread callback function
   * @param Threadedcb the function run in thread
   * @param void* user data for the threaded callback function
   */
  void setCallback(threaded_cb t_cb, void *user_data);

  /**
   * @brief return the number of workders
   * @return unsigned int the number of workers
   *
   */
  unsigned int getNumWorkers() { return num_workers; }

private:
  threaded_cb cb;
  unsigned int batch;
  unsigned int num_workers;
  std::vector<std::thread> workers;
  std::unique_ptr<props::PropsUserData> user_data_prop;
};

} // namespace nntrainer
#endif // __NODE_EXPORTER_H__
