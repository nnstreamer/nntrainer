// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Hyeonseok Lee <hs89.lee@samsung.com>
 * Copyright (C) 2024 Yash Singh <yash.singh@samsung.com>
 *
 * @file   custom_vocab_selection.h
 * @date   1 Oct 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Yash Singh <yash.singh@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Implementation of custom vocab selection
 */

#ifndef VOCAB_SELECTION_H
#define VOCAB_SELECTION_H

#include <tensor.h>

#ifndef LSH_BLOCK_SIZE
#define LSH_BLOCK_SIZE 256
#endif

using namespace std;

namespace nntrainer {

/**
 * @brief Enumeration for different types of LSH algorithms used in vocab
 * selection
 *
 */
enum LshType { NONE = 0, SIMHASH = 1, ORTHOSIMHASH = 2 };
typedef std::bitset<LSH_BLOCK_SIZE> lshDataBlock;

/**
 * @brief Vocab Selection class to select the vocabs from model output using LSH
 *
 */
class VocabSelection {
protected:
  int hiddenSize;
  int vocabCnt;
  const int lshBlockSize = LSH_BLOCK_SIZE;
  int lshBlockNum;
  int lshBits; // lshBlockSize * lshBlockNum
  int lshChoices;
  LshType lshType;
  std::vector<lshDataBlock> lshData;

public:
  /**
   * @brief Constructor of VocabSelection class
   *
   */
  VocabSelection(LshType lshType, int lshChoices, int hiddenSize, int vocabCnt);
  virtual std::
    vector<std::vector<int>>

    /**
     * @brief Get the Vocabs object
     */
    getVocabs(const nntrainer::Tensor &modelOutput) = 0;

  /**
   * @brief Destructor of VocabSelection class
   */
  ~VocabSelection();
};

/**
 * @brief Vocab Selection NNTrainer class to select the vocabs from model output
 * using LSH
 *
 */
class VocabSelectionNNTrainer : public VocabSelection {
protected:
  nntrainer::Tensor lshWeight;

public:
  /**
   * @brief Constructor of VocabSelectionNNTrainer class
   */
  VocabSelectionNNTrainer(LshType lshType, int lshChoices, int hiddenSize,
                          int vocabCnt, nntrainer::Tensor &weights);
  virtual std::
    vector<std::vector<int>>

    /**
     * @brief Get the Vocabs object
     *
     */
    getVocabs(const nntrainer::Tensor &modelOutput);

  /**
   * @brief Destructor of VocabSelectionNNTrainer class
   */
  ~VocabSelectionNNTrainer(){};
};

} // namespace nntrainer

#endif
