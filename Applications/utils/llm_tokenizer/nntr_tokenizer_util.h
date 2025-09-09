// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Donghak park <donghak.park@samsung.com>
 *
 * @file   nntr_tokenizer_util.h
 * @date   09 September 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Donghak park <donghak.park@samsung.com>
 */

#ifndef NNTRAINER_NNTR_TOKENIZER_UTIL_H
#define NNTRAINER_NNTR_TOKENIZER_UTIL_H

#pragma once

#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#define WSTR std::wstring
#define WCHAR_P wchar_t *
#else
#define WIN_EXPORT
#define WSTR std::string
#define WCHAR_P std::string &
#endif

#include <tokenizers_cpp.h>
#include <vector>

namespace nntrainer {
WIN_EXPORT class Tokenizer_Util {
public:
  Tokenizer_Util() { pending_ids_.clear(); };

  ~Tokenizer_Util() { pending_ids_.clear(); };

  void registerOutputs(std::unique_ptr<tokenizers::Tokenizer> &tokenizer,
                       std::vector<unsigned int> ids, unsigned int pos,
                       const std::vector<bool> &eos_list,
                       unsigned int *ids_history, unsigned int MAX_SEQ_LEN,
                       std::vector<std::string> &output_list);

private:
  std::vector<int> pending_ids_;
};

} // namespace nntrainer

#endif // NNTRAINER_NNTR_TOKENIZER_UTIL_H
