
/**
 * Copyright (c) 2023 by Contributors
 * @file   huggingface_tokenizer.cpp
 * @brief  Huggingface tokenizer
 * @author Donghak PARK <donghak.park@samsung.com>
 * @bug    No known bugs except for NYI items
 */
#include <env_compat.h>
#include <tokenizers_c.h>
#include <tokenizers_cpp.h>

#include <cassert>

namespace tokenizers {
/**
 * @brief A simple c++ header of tokenizer via C API.
 */
class HFTokenizer : public Tokenizer {
public:
  explicit HFTokenizer(TokenizerHandle handle) : handle_(handle) {
#ifdef COMPILE_WASM_RUNTIME
    setenv("TOKENIZERS_PARALLELISM", "false", true);
#endif
  }

  HFTokenizer(const HFTokenizer &) = delete;
  HFTokenizer(HFTokenizer &&other) { std::swap(other.handle_, handle_); }

  ~HFTokenizer() {
    if (handle_ != nullptr) {
      tokenizers_free(handle_);
    }
  }

  // use i32 to be consistent with sentencepiece
  std::vector<int32_t> Encode(const std::string &text,
                              bool add_special_tokens) {
    TokenizerEncodeResult result;
    tokenizers_encode(handle_, text.data(), text.length(),
                      static_cast<int>(add_special_tokens), &result);
    std::vector<int32_t> ret(result.token_ids, result.token_ids + result.len);
    tokenizers_free_encode_results(&result, 1);
    return ret;
  }

  // use i32 to be consistent with sentencepiece
  std::vector<int32_t> Encode(const std::string &text) final {
    return Encode(text, false);
  }

  std::vector<std::vector<int32_t>>
  EncodeBatch(const std::vector<std::string> &texts, bool add_special_tokens) {
    std::vector<const char *> texts_raw;
    std::vector<size_t> seq_lens;
    size_t num_seqs = texts.size();
    texts_raw.reserve(num_seqs);
    seq_lens.reserve(num_seqs);
    for (const auto &text : texts) {
      texts_raw.push_back(text.data());
      seq_lens.push_back(text.length());
    }
    std::vector<TokenizerEncodeResult> results(num_seqs);
    tokenizers_encode_batch(handle_, texts_raw.data(), seq_lens.data(),
                            texts.size(), static_cast<int>(add_special_tokens),
                            results.data());
    std::vector<std::vector<int32_t>> ret;
    ret.reserve(texts.size());
    for (size_t i = 0; i < texts.size(); ++i) {
      ret.push_back(std::vector<int32_t>(
        results[i].token_ids, results[i].token_ids + results[i].len));
    }
    tokenizers_free_encode_results(results.data(), texts.size());
    return ret;
  }

  std::vector<std::vector<int32_t>>
  EncodeBatch(const std::vector<std::string> &texts) final {
    return EncodeBatch(texts, false);
  }

  // use i32 to be consistent with sentencepiece
  std::string Decode(const std::vector<int32_t> &ids,
                     bool skip_special_tokens) {
    tokenizers_decode(handle_, reinterpret_cast<const uint32_t *>(ids.data()),
                      ids.size(), static_cast<int>(skip_special_tokens));
    const char *data;
    size_t len;
    tokenizers_get_decode_str(handle_, &data, &len);
    return std::string(data, len);
  }

  std::string Decode(const std::vector<int32_t> &ids) final {
    return Decode(ids, false);
  }

  size_t GetVocabSize() final {
    size_t size;
    tokenizers_get_vocab_size(handle_, &size);
    assert(size > 0);
    return size;
  }

  std::string IdToToken(int32_t id) final {
    const char *data;
    size_t len;
    tokenizers_id_to_token(handle_, static_cast<uint32_t>(id), &data, &len);
    return std::string(data, len);
  }

  int32_t TokenToId(const std::string &token) final {
    int32_t id;
    tokenizers_token_to_id(handle_, token.data(), token.length(), &id);
    return id;
  }

private:
  // internal handle
  TokenizerHandle handle_{nullptr};
};

std::unique_ptr<Tokenizer> Tokenizer::FromBlobJSON(const std::string &json) {
  return std::make_unique<HFTokenizer>(
    tokenizers_new_from_str(json.data(), json.length()));
}

std::unique_ptr<Tokenizer>
Tokenizer::FromBlobByteLevelBPE(const std::string &vocab,
                                const std::string &merges,
                                const std::string &added_tokens) {
  return std::make_unique<HFTokenizer>(byte_level_bpe_tokenizers_new_from_str(
    vocab.data(), vocab.length(), merges.data(), merges.length(),
    added_tokens.data(), added_tokens.length()));
}

#ifndef CAUSALLM_DISABLE_TOKENIZER_CACHE
std::unique_ptr<Tokenizer> Tokenizer::FromSnapshot(const std::string &blob) {
  TokenizerHandle handle =
    tokenizers_new_from_snapshot(blob.data(), blob.length());
  if (handle == nullptr) {
    return nullptr; // caller falls back to the JSON parse path
  }
  return std::make_unique<HFTokenizer>(handle);
}

std::string Tokenizer::SnapshotFromJSON(const std::string &json) {
  char *data = nullptr;
  size_t len = 0;
  tokenizers_snapshot_from_json(json.data(), json.length(), &data, &len);
  if (data == nullptr || len == 0) {
    return {};
  }
  std::string blob(data, len);
  tokenizers_snapshot_free(data);
  return blob;
}
#else
// The linked libtokenizers_c predates the snapshot entry points: no snapshot
// can be read or written, so every load takes the JSON parse path.
std::unique_ptr<Tokenizer> Tokenizer::FromSnapshot(const std::string &blob) {
  (void)blob;
  return nullptr;
}

std::string Tokenizer::SnapshotFromJSON(const std::string &json) {
  (void)json;
  return {};
}
#endif
} // namespace tokenizers
