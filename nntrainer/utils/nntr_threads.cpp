// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2022 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file nntr_threads.cpp
 * @date 07 July 2022
 * @brief Thread Management for NNTrainer
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <nntr_threads.h>

namespace nntrainer {

ParallelBatch::ParallelBatch(threaded_cb threaded_cb, unsigned int batch_size,
                             void *user_data_) :
  cb(threaded_cb),
  batch(batch_size),
  num_workers(NNTR_NUM_THREADS > batch ? 1 : NNTR_NUM_THREADS),
  user_data_prop(new props::PropsUserData(user_data_)) {}

ParallelBatch::~ParallelBatch() {}

void ParallelBatch::run() {

  unsigned int start = 0;
  unsigned int end = batch;

  unsigned int chunk = (end - start + (num_workers - 1)) / num_workers;

  for (unsigned int i = 0; i < num_workers; ++i) {
    unsigned int s = start + i * chunk;
    unsigned int e = s + chunk;
    if (e > end)
      e = end;
    workers.push_back(std::thread(cb, s, e, user_data_prop->get()));
  }

  for (unsigned int i = 0; i < num_workers; ++i)
    workers[i].join();
}

} // namespace nntrainer
