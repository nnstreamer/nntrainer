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

#include <algorithm>
#include <nntr_threads.h>

#ifdef NNTR_NUM_THREADS
static const unsigned int nntr_num_threads = NNTR_NUM_THREADS;
#else
static const unsigned int nntr_num_threads = 1;
#endif

namespace nntrainer {

ParallelBatch::ParallelBatch(unsigned int batch_size) :
  cb(nullptr),
  batch(batch_size),
  num_workers(nntr_num_threads > batch ? 1 : nntr_num_threads),
  user_data_prop(new props::PropsUserData(nullptr)){};

ParallelBatch::ParallelBatch(threaded_cb threaded_cb_, unsigned int batch_size,
                             void *user_data_) :
  cb(threaded_cb_),
  batch(batch_size),
  num_workers(nntr_num_threads > batch ? 1 : nntr_num_threads),
  user_data_prop(new props::PropsUserData(user_data_)) {}

ParallelBatch::~ParallelBatch() {}

void ParallelBatch::run() {

  if (!cb) {
    throw std::invalid_argument("nntrainer threads: callback is not defined");
  }

  unsigned int start = 0;
  unsigned int end = batch;

  unsigned int chunk = (end - start + (num_workers - 1)) / num_workers;

  for (unsigned int i = 0; i < num_workers; ++i) {
    unsigned int s = start + i * chunk;
    unsigned int e = s + chunk;
    if (e > end)
      e = end;
    workers.push_back(std::thread(cb, s, e, i, user_data_prop->get()));
  }

  std::for_each(workers.begin(), workers.end(),
                std::mem_fn(&std::thread::join));
}

void ParallelBatch::setCallback(threaded_cb threaded_cb_, void *user_data_) {
  cb = threaded_cb_;
  user_data_prop = std::make_unique<props::PropsUserData>(user_data_);
}

} // namespace nntrainer
