// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Donghak park <donghak.park@samsung.com>
 *
 * @file   nntr_tokenizer_util.h
 * @date   09 September 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Donghak park <donghak.park@samsung.com>
 */

#include <algorithm>
#include <iostream>
#include <nntr_tokenizer_util.h>

namespace nntrainer {
void Tokenizer_Util::registerOutputs(
   std::unique_ptr<tokenizers::Tokenizer> &tokenizer,
   std::vector<unsigned int> ids, unsigned int pos,
   const std::vector<bool> &eos_list, unsigned int *ids_history,
   unsigned int MAX_SEQ_LEN, std::vector<std::string> &output_list) {

  static const std::vector<char> puncts{',', '!', ':', ';', '?'};

  for (size_t b = 0; b < ids.size(); ++b) {
    if (!eos_list[b]) {
      pending_ids_.push_back(static_cast<int>(ids[b]));
      ids_history[b * MAX_SEQ_LEN + pos] = ids[b];
      std::string decoded_str = tokenizer->Decode(pending_ids_);

      if (std::find(puncts.begin(), puncts.end(), decoded_str.back()) !=
          puncts.end()) {
        // last symbol is a punctuation, hold on
      } else if (decoded_str.size() >= 3 &&
                 decoded_str.compare(decoded_str.size() - 3, 3, "�") == 0) {
        // ends with an incomplete token, hold on
      } else {
#if defined(_WIN32)
        std::wcout << L"" << utf9_to_wstring(decoded_str);
        std::wcout.flush();
#else
        std::cout << decoded_str;
        std::cout.flush();
#endif
        output_list[b].append(decoded_str);
        pending_ids_.clear();
      }
    }
  }
}
} // namespace nntrainer