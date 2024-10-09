// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Yash Singh <yash.singh@samsung.com>>
 *
 * @file   custom_vocab_selection.cpp
 * @date   1 Oct 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Yash Singh <yash.singh@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Implementation of custom vocab selection
 */

#include <algorithm>
#include <custom_vocab_selection.h>

nntrainer::VocabSelection::VocabSelection(LshType lshType, int lshChoices,
                                          int hiddenSize, int vocabCnt) :
  lshType(lshType),
  lshChoices(lshChoices),
  vocabCnt(vocabCnt),
  hiddenSize(hiddenSize),
  lshBlockNum(0),
  lshBits(0) {}

nntrainer::VocabSelection::~VocabSelection() {}

nntrainer::VocabSelectionNNTrainer::VocabSelectionNNTrainer(
  LshType lshType, int lshChoices, int hiddenSize, int vocabCnt,
  nntrainer::Tensor &weights) :
  VocabSelection(lshType, lshChoices, hiddenSize, vocabCnt) {
  this->lshBlockNum = (hiddenSize + lshBlockSize - 1) / lshBlockSize;
  this->lshBits = lshBlockNum * lshBlockSize;
  this->lshData = std::vector<lshDataBlock>(this->vocabCnt * lshBlockNum);

  // for (unsigned int i = 0; i < vocabCnt; ++i) {
  //     for (unsigned int j = 0; j < lshBlockNum; ++j) {
  //         unsigned int actualSize = std::min(lshBlockSize, hiddenSize -
  //         (int)j * lshBlockSize); lshDataBlock d; for (unsigned int k = 0; k
  //         < actualSize; ++k) {
  //             d[k] = weights.getValue<_FP16>(0, 0, i, j * lshBlockSize + k) >
  //             0 ? 1 : 0;
  //         }
  //         for (unsigned int k = actualSize; k < lshBlockSize; ++k) {
  //             d[k] = 0;
  //         }
  //         this->lshData[i * lshBlockNum + j] = d;
  //     }
  // }

  for (unsigned int i = 0; i < lshBlockNum; ++i) {
    unsigned int actualSize =
      std::min(lshBlockSize, hiddenSize - (int)i * lshBlockSize);
    for (unsigned int j = 0; j < vocabCnt; ++j) {
      lshDataBlock d;
      for (unsigned int k = 0; k < actualSize; ++k) {
        if (weights.getDataType() == nntrainer::TensorDim::DataType::FP32) {
          d[k] = weights.getValue(0, 0, i * lshBlockSize + k, j) > 0 ? 1 : 0;
        } else if (weights.getDataType() ==
                   nntrainer::TensorDim::DataType::FP16) {
          d[k] =
            weights.getValue<_FP16>(0, 0, i * lshBlockSize + k, j) > 0 ? 1 : 0;
        }
      }
      for (unsigned int k = actualSize; k < lshBlockSize; ++k) {
        d[k] = 0;
      }
      this->lshData[j * lshBlockNum + i] = d;
    }
  }
}

std::vector<std::vector<int>>
nntrainer::VocabSelectionNNTrainer::getVocabs(const nntrainer::Tensor &input) {
  unsigned int batchSize = input.height();

  std::vector<std::vector<int>> res = std::vector<std::vector<int>>(batchSize);
  for (int i = 0; i < batchSize; i++) {
    std::vector<lshDataBlock> d(lshBlockNum);
    for (int k = 0; k < lshBlockNum; k++) {
      int actualSize = std::min(lshBlockSize, hiddenSize - k * lshBlockSize);
      for (int j = 0; j < actualSize; j++) {
        if (input.getDataType() == nntrainer::TensorDim::DataType::FP32) {
          d[k][j] = input.getValue(0, 0, i, j + k * lshBlockSize) >= 0 ? 1 : 0;
        } else if (input.getDataType() ==
                   nntrainer::TensorDim::DataType::FP16) {
          d[k][j] =
            input.getValue<_FP16>(0, 0, i, j + k * lshBlockSize) >= 0 ? 1 : 0;
        }
      }
      for (int j = actualSize; j < lshBlockSize; j++) {
        d[k][j] = 0;
      }
    }
    std::vector<int> simResult(vocabCnt, 0);
    std::vector<int> simCount(lshBits + 1, 0);
    for (int j = 0; j < vocabCnt; j++) {
      for (int k = 0; k < lshBlockNum; k++) {
        simResult[j] += (d[k] ^ lshData[j * lshBlockNum + k]).count();
      }
      simCount[simResult[j]]++;
    }
    int cut = lshBits + 1;
    int leftover = 0;
    int countSum = 0;
    for (int j = 0; j <= lshBits; j++) {
      countSum += simCount[j];
      if (countSum > lshChoices) {
        cut = j;
        leftover = simCount[j] - (countSum - lshChoices);
        break;
      }
    }
    std::vector<int> selectedVocabs(lshChoices);
    int pos = 0;
    for (int j = 0; j < vocabCnt; j++) {
      if (simResult[j] <= cut) {
        if (simResult[j] < cut) {
          selectedVocabs[pos] = j;
          pos++;
        } else if (leftover > 0) {
          selectedVocabs[pos] = j;
          pos++;
          leftover--;
        }
      }
    }
    res[i] = selectedVocabs;
  }
  return res;
}
