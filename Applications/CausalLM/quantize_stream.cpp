// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   quantize_stream.cpp
 * @brief  Bounded-memory weight quantizer for supported CausalLM models.
 * @author Jungwon-Lee <jungone.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <cpu_backend.h>
#include <htp_wh_layout.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-literal-operator"
#endif
#include "json.hpp"
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

namespace {

using json = nlohmann::json;

constexpr size_t QK4_0 = 32;
constexpr size_t Q4_0_BLOCK_BYTES = 18;
constexpr size_t QK_K = 256;
constexpr size_t Q4_K_BLOCK_BYTES = 144;
constexpr size_t Q6_K_BLOCK_BYTES = 210;
constexpr size_t MAX_TENSOR_BUFFER_BYTES = 64U * 1024U * 1024U;

using DType = ml::train::TensorDim::DataType;

struct QuantizationPlan {
  DType fc_dtype;
  DType embedding_dtype;
  DType lmhead_dtype;
  /** MoE expert FFN dtype; defaults to fc_dtype. Mirrors nntr_quantize's
   *  --moe_dtype so both quantizers accept the same options. */
  DType moe_dtype;
  ml::train::ISA target_isa;
  DType router_dtype = DType::FP32;
};

struct Qwen3MoePlan {
  size_t hidden_size;
  size_t vocab_size;
  size_t num_layers;
  size_t num_attention_heads;
  size_t num_key_value_heads;
  size_t head_dim;
  size_t intermediate_size;
  size_t num_experts;
  bool tied_embeddings;
};

struct Gemma4MoePlan {
  size_t hidden_size;
  size_t vocab_size;
  size_t num_layers;
  size_t num_attention_heads;
  size_t num_key_value_heads;
  size_t head_dim;
  size_t global_head_dim;
  size_t num_global_key_value_heads;
  size_t intermediate_size;
  size_t moe_intermediate_size;
  size_t num_experts;
  size_t per_layer_input_size;
  size_t per_layer_vocab_size;
  std::vector<std::string> layer_types;
  bool attention_k_eq_v;
  bool tied_embeddings;
};

struct Lfm2MoePlan {
  size_t hidden_size;
  size_t vocab_size;
  size_t num_layers;
  size_t num_attention_heads;
  size_t num_key_value_heads;
  size_t head_dim;
  size_t intermediate_size;     /**< dense SwiGLU FFN width */
  size_t moe_intermediate_size; /**< per-expert FFN width */
  size_t num_experts;
  size_t num_dense_layers; /**< layers [0, this) keep the dense FFN */
  size_t conv_dim;
  size_t conv_l_cache;
  std::vector<std::string> layer_types; /**< "conv" or "attention" per layer */
  bool tied_embeddings;
};

std::string upper(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return std::toupper(c); });
  return value;
}

std::string lower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return value;
}

size_t checkedMultiply(size_t lhs, size_t rhs, const std::string &name) {
  if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
    throw std::overflow_error("Tensor size overflow for " + name);
  }
  return lhs * rhs;
}

size_t tensorElements(size_t rows, size_t columns, const std::string &name) {
  return checkedMultiply(rows, columns, name);
}

size_t tensorBytes(size_t rows, size_t columns, const std::string &name) {
  return checkedMultiply(tensorElements(rows, columns, name), sizeof(float),
                         name);
}

json readJson(const std::filesystem::path &path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open " + path.string());
  }

  json value;
  file >> value;
  return value;
}

size_t getSize(const json &cfg, const std::string &key) {
  if (!cfg.contains(key) || !cfg[key].is_number_unsigned()) {
    throw std::runtime_error("config.json is missing unsigned numeric key: " +
                             key);
  }
  return cfg[key].get<size_t>();
}

std::string getArchitecture(const json &cfg) {
  if (!cfg.contains("architectures") || !cfg["architectures"].is_array() ||
      cfg["architectures"].empty()) {
    throw std::runtime_error("config.json is missing architectures[0]");
  }
  return cfg["architectures"][0].get<std::string>();
}

Qwen3MoePlan makeQwen3MoePlan(const json &cfg) {
  const std::string architecture = getArchitecture(cfg);
  if (architecture != "Qwen3MoeForCausalLM") {
    throw std::runtime_error(
      "nntr_quantize_stream only supports Qwen3MoeForCausalLM, but got " +
      architecture);
  }

  const size_t hidden_size = getSize(cfg, "hidden_size");
  const size_t num_attention_heads = getSize(cfg, "num_attention_heads");
  if (num_attention_heads == 0) {
    throw std::runtime_error("num_attention_heads must be greater than zero");
  }

  const size_t head_dim = cfg.contains("head_dim")
                            ? getSize(cfg, "head_dim")
                            : hidden_size / num_attention_heads;
  const size_t num_key_value_heads = cfg.contains("num_key_value_heads")
                                       ? getSize(cfg, "num_key_value_heads")
                                       : num_attention_heads;

  if (head_dim == 0 || num_key_value_heads == 0) {
    throw std::runtime_error(
      "head_dim and num_key_value_heads must be greater than zero");
  }

  return {
    hidden_size,
    getSize(cfg, "vocab_size"),
    getSize(cfg, "num_hidden_layers"),
    num_attention_heads,
    num_key_value_heads,
    head_dim,
    getSize(cfg, "moe_intermediate_size"),
    getSize(cfg, "num_experts"),
    cfg.value("tie_word_embeddings", false),
  };
}

Gemma4MoePlan makeGemma4MoePlan(const json &root_cfg) {
  const std::string architecture = getArchitecture(root_cfg);
  if (architecture != "Gemma4ForCausalLM" &&
      architecture != "Gemma4ForConditionalGeneration") {
    throw std::runtime_error("Unsupported Gemma4 architecture: " +
                             architecture);
  }

  const json &cfg =
    root_cfg.contains("text_config") ? root_cfg.at("text_config") : root_cfg;
  if (!cfg.is_object())
    throw std::runtime_error("Gemma4 text_config must be an object");
  if (!cfg.value("enable_moe_block", false)) {
    throw std::runtime_error(
      "Gemma4 streaming quantization requires enable_moe_block=true");
  }

  const size_t hidden_size = getSize(cfg, "hidden_size");
  const size_t num_layers = getSize(cfg, "num_hidden_layers");
  const size_t num_attention_heads = getSize(cfg, "num_attention_heads");
  if (num_attention_heads == 0)
    throw std::runtime_error("num_attention_heads must be greater than zero");

  const size_t head_dim = cfg.contains("head_dim")
                            ? getSize(cfg, "head_dim")
                            : hidden_size / num_attention_heads;
  const size_t num_key_value_heads = cfg.contains("num_key_value_heads")
                                       ? getSize(cfg, "num_key_value_heads")
                                       : num_attention_heads;
  const size_t global_head_dim =
    cfg.contains("global_head_dim") && !cfg["global_head_dim"].is_null()
      ? getSize(cfg, "global_head_dim")
      : head_dim;
  const size_t num_global_key_value_heads =
    cfg.contains("num_global_key_value_heads") &&
        !cfg["num_global_key_value_heads"].is_null()
      ? getSize(cfg, "num_global_key_value_heads")
      : num_key_value_heads;

  const size_t num_kv_shared_layers =
    cfg.value("num_kv_shared_layers", size_t{0});
  if (num_kv_shared_layers != 0) {
    throw std::runtime_error(
      "Gemma4 MoE streaming quantization does not support shared KV layers");
  }

  std::vector<std::string> layer_types(num_layers,
                                       std::string("sliding_attention"));
  if (cfg.contains("layer_types")) {
    layer_types = cfg["layer_types"].get<std::vector<std::string>>();
    if (layer_types.size() != num_layers) {
      throw std::runtime_error(
        "Gemma4 layer_types size must match num_hidden_layers");
    }
  }
  for (const auto &type : layer_types) {
    if (type != "sliding_attention" && type != "full_attention")
      throw std::runtime_error("Unsupported Gemma4 layer type: " + type);
  }

  const size_t per_layer_input_size =
    cfg.value("hidden_size_per_layer_input", size_t{0});
  const size_t per_layer_vocab_size =
    cfg.value("vocab_size_per_layer_input", getSize(cfg, "vocab_size"));
  if (per_layer_input_size != 0 && per_layer_vocab_size == 0) {
    throw std::runtime_error(
      "vocab_size_per_layer_input must be greater than zero");
  }

  const bool tied_embeddings = cfg.contains("tie_word_embeddings")
                                 ? cfg["tie_word_embeddings"].get<bool>()
                                 : root_cfg.value("tie_word_embeddings", true);
  return {
    hidden_size,
    getSize(cfg, "vocab_size"),
    num_layers,
    num_attention_heads,
    num_key_value_heads,
    head_dim,
    global_head_dim,
    num_global_key_value_heads,
    getSize(cfg, "intermediate_size"),
    getSize(cfg, "moe_intermediate_size"),
    getSize(cfg, "num_experts"),
    per_layer_input_size,
    per_layer_vocab_size,
    std::move(layer_types),
    cfg.value("attention_k_eq_v", false),
    tied_embeddings,
  };
}

Lfm2MoePlan makeLfm2MoePlan(const json &cfg) {
  const std::string architecture = getArchitecture(cfg);
  if (architecture != "Lfm2MoeForCausalLM") {
    throw std::runtime_error("Unsupported LFM2 architecture: " + architecture);
  }

  const size_t hidden_size = getSize(cfg, "hidden_size");
  const size_t num_layers = getSize(cfg, "num_hidden_layers");
  const size_t num_attention_heads = getSize(cfg, "num_attention_heads");
  if (num_attention_heads == 0) {
    throw std::runtime_error("num_attention_heads must be greater than zero");
  }

  const size_t head_dim = cfg.contains("head_dim")
                            ? getSize(cfg, "head_dim")
                            : hidden_size / num_attention_heads;
  const size_t num_key_value_heads = cfg.contains("num_key_value_heads")
                                       ? getSize(cfg, "num_key_value_heads")
                                       : num_attention_heads;
  if (head_dim == 0 || num_key_value_heads == 0) {
    throw std::runtime_error(
      "head_dim and num_key_value_heads must be greater than zero");
  }

  // Lfm2CausalLM::setupParameters defaults conv_dim to hidden_size and
  // requires layer_types; mirror both so a config it accepts is a config
  // this tool accepts.
  const size_t conv_dim =
    cfg.contains("conv_dim") ? getSize(cfg, "conv_dim") : hidden_size;
  const size_t conv_l_cache = cfg.value("conv_L_cache", size_t{0});

  if (!cfg.contains("layer_types")) {
    throw std::runtime_error("Lfm2Moe requires layer_types in config.json");
  }
  auto layer_types = cfg["layer_types"].get<std::vector<std::string>>();
  if (layer_types.size() != num_layers) {
    throw std::runtime_error("layer_types size must match num_hidden_layers");
  }
  for (const auto &type : layer_types) {
    if (type != "conv" && type != "attention" && type != "full_attention")
      throw std::runtime_error("Unsupported LFM2 layer type: " + type);
    if (type == "conv" && conv_l_cache == 0)
      throw std::runtime_error("conv layers require a non-zero conv_L_cache");
  }

  const size_t num_dense_layers = cfg.value("num_dense_layers", size_t{0});
  if (num_dense_layers > num_layers) {
    throw std::runtime_error("num_dense_layers exceeds num_hidden_layers");
  }

  return {
    hidden_size,
    getSize(cfg, "vocab_size"),
    num_layers,
    num_attention_heads,
    num_key_value_heads,
    head_dim,
    getSize(cfg, "intermediate_size"),
    getSize(cfg, "moe_intermediate_size"),
    getSize(cfg, "num_experts"),
    num_dense_layers,
    conv_dim,
    conv_l_cache,
    std::move(layer_types),
    // Hugging Face LFM2 configs omit tie_word_embeddings while the model
    // defaults it to true and ships no standalone lm_head -- same default
    // Lfm2CausalLM::setupParameters uses.
    cfg.value("tie_word_embeddings", true),
  };
}

DType parseDType(const std::string &value) {
  const std::string dtype = upper(value);
  if (dtype == "FP32")
    return DType::FP32;
  if (dtype == "Q4_0" || dtype == "Q40")
    return DType::Q4_0;
  if (dtype == "Q4_K" || dtype == "Q4K")
    return DType::Q4_K;
  if (dtype == "Q6_K" || dtype == "Q6K")
    return DType::Q6_K;
  if (dtype == "QS4CX")
    return DType::QS4CX;
  if (dtype == "QS4CX_WH")
    return DType::QS4CX_WH;
  throw std::invalid_argument("Unsupported dtype: " + value +
                              " (supported: FP32, Q4_0, Q4_K, Q6_K, QS4CX)");
}

const char *dtypeName(DType dtype) {
  switch (dtype) {
  case DType::FP32:
    return "FP32";
  case DType::Q4_0:
    return "Q4_0";
  case DType::Q4_K:
    return "Q4_K";
  case DType::Q6_K:
    return "Q6_K";
  case DType::QS4CX:
    return "QS4CX";
  case DType::QS4CX_WH:
    return "QS4CX_WH";
  default:
    throw std::invalid_argument("Unknown dtype");
  }
}

std::string dtypeSuffix(DType dtype) {
  std::string suffix = lower(dtypeName(dtype));
  suffix.erase(std::remove(suffix.begin(), suffix.end(), '_'), suffix.end());
  return suffix;
}

ml::train::ISA parseIsa(const std::string &value) {
  const std::string isa = upper(value);
  if (isa == "DEFAULT")
    return ml::train::ISA::DEFAULT;
  if (isa == "X86")
    return ml::train::ISA::X86;
  if (isa == "ARM")
    return ml::train::ISA::ARM;
  throw std::invalid_argument("Unsupported ISA: " + value +
                              " (supported: DEFAULT, X86, ARM)");
}

const char *isaName(ml::train::ISA isa) {
  switch (isa) {
  case ml::train::ISA::DEFAULT:
    return "DEFAULT";
  case ml::train::ISA::X86:
    return "X86";
  case ml::train::ISA::ARM:
    return "ARM";
  }
  throw std::invalid_argument("Unknown ISA");
}

/** @brief Packed int4 bytes for one QS4CX output channel of @a columns */
size_t qs4cxRowBytes(size_t columns) { return (columns + 1) / 2; }

size_t quantizedSize(DType dtype, size_t rows, size_t columns, bool repack,
                     const std::string &name) {
  switch (dtype) {
  case DType::FP32:
    return tensorBytes(rows, columns, name);
  case DType::Q4_0:
    if (columns % QK4_0 != 0 || (repack && rows % QK4_0 != 0)) {
      throw std::invalid_argument(
        name + " Q4_0 shape must have columns divisible by 32" +
        (repack ? " and rows divisible by 32" : ""));
    }
    return checkedMultiply(checkedMultiply(rows, columns / QK4_0, name),
                           Q4_0_BLOCK_BYTES, name);
  case DType::Q4_K:
    if (columns % QK_K != 0 || (repack && rows % 8 != 0)) {
      throw std::invalid_argument(
        name + " Q4_K shape must have columns divisible by 256" +
        (repack ? " and rows divisible by 8" : ""));
    }
    return checkedMultiply(checkedMultiply(rows, columns / QK_K, name),
                           Q4_K_BLOCK_BYTES, name);
  case DType::Q6_K:
    if (columns % QK_K != 0) {
      throw std::invalid_argument(name +
                                  " Q6_K columns must be divisible by 256");
    }
    return checkedMultiply(checkedMultiply(rows, columns / QK_K, name),
                           Q6_K_BLOCK_BYTES, name);
  case DType::QS4CX:
    // One int4 per weight packed two-per-byte, plus one f32 scale per output
    // channel -- the layout layer_devel.h's QS4CX save branch writes, with
    // every nibble first and every scale after (see writeQuantized).
    return checkedMultiply(rows, qs4cxRowBytes(columns) + sizeof(float), name);
  case DType::QS4CX_WH:
    // The same nibbles rearranged into HMX weight tiles, which is the same
    // byte count -- a 32x32 tile is 512 bytes either way -- plus a second f32
    // per output channel for the column sum the matmul needs and cannot
    // cheaply recompute from packed nibbles.
    if (rows == 0 || columns == 0 || rows % 32 || columns % 32 ||
        rows > std::numeric_limits<uint32_t>::max() ||
        columns > std::numeric_limits<uint32_t>::max()) {
      throw std::invalid_argument(name + " QS4CX_WH dimensions must be nonzero "
                                         "multiples of 32 that fit uint32");
    }
    return checkedMultiply(rows, qs4cxRowBytes(columns) + 2 * sizeof(float),
                           name);
  default:
    break;
  }
  throw std::invalid_argument("Unknown dtype for " + name);
}

/** @brief One tensor of a .safetensors header: its name and its payload size,
 *  in the order the payload stores them. See TensorWriter::source_tensors_. */
struct SourceTensor {
  std::string name;
  size_t bytes;
};

class TensorWriter {
public:
  /**
   * @param source_bytes size of the source .bin, only read in dry-run mode
   * @param dry_run count the source bytes the layout walk needs, touching
   *        neither stream -- see expectedInputBytes()
   * @param source_tensors the input's tensors in payload order when it is a
   *        .safetensors, else null -- see expectSourceTensor()
   * @param tensor_buffer_bytes FP32 block budget; smaller budgets also allow
   *        tests to compare blocked and whole-tensor serialization cheaply
   */
  TensorWriter(std::ifstream &input, std::ofstream &output,
               ml::train::ISA target_isa, size_t source_bytes = 0,
               bool dry_run = false,
               const std::vector<SourceTensor> *source_tensors = nullptr,
               size_t tensor_buffer_bytes = MAX_TENSOR_BUFFER_BYTES) :
    input_(input),
    output_(output),
    target_isa_(target_isa),
    source_bytes_(source_bytes),
    dry_run_(dry_run),
    source_tensors_(source_tensors),
    tensor_buffer_bytes_(tensor_buffer_bytes) {}

  bool isDryRun() const { return dry_run_; }

  /** @brief FP32 source bytes the layout walk consumed (dry run only) */
  size_t expectedInputBytes() const { return expected_input_bytes_; }

  /** @brief FC source ranges for tools/htp/prepare_fc_wh.py. */
  const json &preparedFcWeights() const { return prepared_fc_weights_; }

  void copyFp32(size_t elements, const std::string &name) {
    const size_t bytes = checkedMultiply(elements, sizeof(float), name);
    expectSourceTensor(bytes, name);
    if (dry_run_) {
      expected_input_bytes_ += bytes;
      return;
    }
    copyBytes(bytes, name);
  }

  void discardFp32(size_t elements, const std::string &name) {
    const size_t bytes = checkedMultiply(elements, sizeof(float), name);
    expectSourceTensor(bytes, name);
    if (dry_run_) {
      expected_input_bytes_ += bytes;
      return;
    }
    discardBytes(bytes, name);
  }

  bool hasRemainingBytes() {
    // Gemma4's optional trailing embedding makes the layout depend on what is
    // left in the source, so the dry run has to answer the same question from
    // the byte count it has accumulated so far.
    if (dry_run_)
      return expected_input_bytes_ < source_bytes_;
    input_.peek();
    const bool remaining = !input_.eof();
    input_.clear();
    return remaining;
  }

  void writeEmbedding(size_t rows, size_t columns, DType dtype,
                      const std::string &name) {
    const size_t source_bytes = tensorBytes(rows, columns, name);
    expectSourceTensor(source_bytes, name);
    if (dtype == DType::FP32) {
      if (dry_run_) {
        expected_input_bytes_ += source_bytes;
        return;
      }
      copyBytes(source_bytes, name);
      return;
    }
    if (dtype == DType::Q4_K) {
      throw std::invalid_argument(
        "Q4_K embedding is not supported by the CausalLM embedding layer");
    }
    // An embedding is a lookup, not a matmul against the HMX unit, so there
    // is nothing for a weight-tile layout to be right for.
    if (dtype == DType::QS4CX_WH) {
      throw std::invalid_argument(
        "QS4CX_WH is a weight layout for the HTP matmul and cannot be used "
        "for an embedding");
    }
    quantizedSize(dtype, rows, columns, false, name);
    if (dry_run_) {
      expected_input_bytes_ += source_bytes;
      return;
    }

    const size_t bytes_per_row = tensorBytes(1, columns, name);
    const size_t rows_per_block =
      std::max<size_t>(1, tensor_buffer_bytes_ / bytes_per_row);

    for (size_t row = 0; row < rows;) {
      const size_t block_rows = std::min(rows_per_block, rows - row);
      std::vector<float> source(tensorElements(block_rows, columns, name));
      readFloats(source, name);
      writeQuantized(source, block_rows, columns, dtype, false, name);
      row += block_rows;
    }
    flushQs4cxScales(dtype, name);
  }

  /**
   * The FP32 NNTrainer file stores FC weights as [input, output]. Quantized FC
   * tensors are stored row-wise as [output, input], so only quantized output
   * is transposed. Large matrices (normally an untied LM head) use a seek-based
   * blocked transpose to cap memory use.
   */
  void writeFc(size_t input_size, size_t output_size, DType dtype,
               const std::string &name) {
    const size_t source_bytes = tensorBytes(input_size, output_size, name);
    expectSourceTensor(source_bytes, name);
    if (dtype == DType::FP32) {
      if (dry_run_) {
        expected_input_bytes_ += source_bytes;
        return;
      }
      copyBytes(source_bytes, name);
      return;
    }

    // Validate the complete shape before writing any part of this tensor.
    quantizedSize(dtype, output_size, input_size, true, name);
    if (dry_run_) {
      expected_input_bytes_ += source_bytes;
      return;
    }

    if (dtype == DType::QS4CX_WH)
      recordPreparedFc(input_size, output_size, name);
    if (source_bytes <= tensor_buffer_bytes_) {
      std::vector<float> source(tensorElements(input_size, output_size, name));
      readFloats(source, name);
      std::vector<float> transposed(source.size());
      for (size_t input = 0; input < input_size; ++input) {
        for (size_t output = 0; output < output_size; ++output) {
          transposed[output * input_size + input] =
            source[input * output_size + output];
        }
      }
      writeQuantized(transposed, output_size, input_size, dtype, true, name);
      flushQs4cxScales(dtype, name);
      return;
    }

    writeBlockedTransposed(input_size, output_size, dtype, name);
  }

  void requireEndOfFile() {
    if (source_tensors_ != nullptr &&
        source_cursor_ != source_tensors_->size()) {
      throw std::runtime_error(
        "The layout walk stopped after " + std::to_string(source_cursor_) +
        " of " + std::to_string(source_tensors_->size()) +
        " tensors in the input's header; the next one the converter wrote is " +
        (*source_tensors_)[source_cursor_].name + ".");
    }
    // The dry run's own total is compared against the file size instead.
    if (dry_run_)
      return;
    const std::streampos consumed = input_.tellg();
    input_.peek();
    if (!input_.eof()) {
      throw std::runtime_error(
        "Input weight file has trailing unread bytes after offset " +
        std::to_string(static_cast<std::streamoff>(consumed)));
    }
  }

private:
  void recordPreparedFc(size_t K, size_t N, const std::string &name) {
    const size_t split = name.find('_');
    if (name.compare(0, 5, "layer") != 0 || split == std::string::npos ||
        split == 5 ||
        !std::all_of(name.begin() + 5, name.begin() + split,
                     [](unsigned char c) { return std::isdigit(c); }))
      return;
    const std::string role = name.substr(split + 1);
    const bool conv = role == "conv_in_proj" || role == "conv_out_proj";
    if (!conv && role != "ffn_up" && role != "ffn_gate" && role != "ffn_down")
      return;
    const std::streampos offset = output_.tellp();
    if (offset < 0)
      throw std::runtime_error("Failed to determine output offset for " + name);
    prepared_fc_weights_.push_back(
      {{"file_offset", static_cast<uint64_t>(offset)},
       {"K", K},
       {"N", N},
       {"name", name},
       {"layer_type", conv ? "conv_block" : "dense_ffn"}});
  }

  /**
   * @brief One tensor read here is one tensor the converter wrote.
   *
   * The .safetensors path reads the payload positionally and used to trust
   * the order outright. It is not a safe thing to trust: a 26B MoE model
   * whose FP32 intermediate came from a converter that wrote the router
   * scale AFTER the router matrix and each expert's gate and up as two
   * tensors instead of the fused one quantized and ran end to end, because
   * the per-layer BYTE TOTALS of the two layouts are identical -- the
   * dry-run total below cannot see the difference. What came out was a
   * router norm made of the router matrix's first rows and a gate|up weight
   * whose halves were consecutive gate rows, which is noise, and the search
   * for it went to the device.
   *
   * The invariant that does catch it costs one comparison: every tensor this
   * tool reads is exactly one tensor in the input's header, so a read that
   * splits one or spans two is a layout disagreement and is named here
   * rather than at the end of a 40-minute run. Only element counts are
   * compared, not names: the converter's keys ("layer0_sparse_moe:gate") and
   * this tool's ("layer0_router") are different spellings of the same
   * tensor, and mapping them would be a second place to keep in step.
   *
   * A .bin input has no header, so there is nothing to check and nothing
   * changes for it.
   */
  void expectSourceTensor(size_t bytes, const std::string &name) {
    if (source_tensors_ == nullptr)
      return;
    if (source_cursor_ >= source_tensors_->size()) {
      throw std::runtime_error("The layout walk wants " + name + " (" +
                               std::to_string(bytes) +
                               " bytes) past the last tensor in the input's "
                               "header.");
    }
    const SourceTensor &have = (*source_tensors_)[source_cursor_];
    if (have.bytes != bytes) {
      throw std::runtime_error(
        "Input layout mismatch at tensor " + std::to_string(source_cursor_) +
        ": this tool reads " + name + " as " + std::to_string(bytes) +
        " bytes, but the next tensor the converter wrote is " + have.name +
        " at " + std::to_string(have.bytes) +
        " bytes. The converter is the authority -- regenerate the FP32 file "
        "with the res/<model>/weight_converter.py of this revision, or fix "
        "the order this tool walks. Byte totals can still agree while the "
        "order does not, so do not read a passing size check as agreement.");
    }
    ++source_cursor_;
  }

  void writeBytes(const void *source, size_t bytes, const std::string &name) {
    if (bytes >
        static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
      throw std::overflow_error("Write size is too large for " + name);
    }
    output_.write(static_cast<const char *>(source),
                  static_cast<std::streamsize>(bytes));
    if (!output_)
      throw std::runtime_error("Failed to write " + name);
  }

  void copyBytes(size_t bytes, const std::string &name) {
    constexpr size_t COPY_BUFFER_BYTES = 16U * 1024U * 1024U;
    std::vector<char> buffer(std::min(COPY_BUFFER_BYTES, bytes));
    size_t remaining = bytes;
    while (remaining != 0) {
      const size_t chunk = std::min(buffer.size(), remaining);
      readExact(buffer.data(), chunk, name);
      writeBytes(buffer.data(), chunk, name);
      remaining -= chunk;
    }
  }

  void discardBytes(size_t bytes, const std::string &name) {
    constexpr size_t DISCARD_BUFFER_BYTES = 16U * 1024U * 1024U;
    std::vector<char> buffer(std::min(DISCARD_BUFFER_BYTES, bytes));
    size_t remaining = bytes;
    while (remaining != 0) {
      const size_t chunk = std::min(buffer.size(), remaining);
      readExact(buffer.data(), chunk, name);
      remaining -= chunk;
    }
  }

  void readExact(char *destination, size_t bytes, const std::string &name) {
    if (bytes >
        static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
      throw std::overflow_error("Read size is too large for " + name);
    }
    const std::streamoff offset_before = input_.tellg();
    input_.read(destination, static_cast<std::streamsize>(bytes));
    const std::streamsize got = input_.gcount();
    if (got != static_cast<std::streamsize>(bytes)) {
      // A wrong shape formula and a truncated source file produce the same
      // symptom (readExact comes up short), but they are not the same bug --
      // one is in this file, the other is upstream. Report enough for the
      // caller to tell which without a manual offset calculation: how far
      // in this happened, and the source file's actual total size, which a
      // truncated conversion (e.g. weight_converter.py killed partway
      // through, OOM) makes far short of what the config implies.
      input_.clear();
      input_.seekg(0, std::ios::end);
      const std::streamoff file_size = input_.tellg();
      throw std::runtime_error(
        "Unexpected EOF while reading " + name + ": needed " +
        std::to_string(bytes) + " bytes at source offset " +
        std::to_string(offset_before) + ", got " + std::to_string(got) +
        ". The source file is only " + std::to_string(file_size) +
        " bytes total. If that is far short of what this model's config "
        "implies, the FP32 .bin is truncated or incomplete -- re-run "
        "whatever produced it (e.g. weight_converter.py) to completion "
        "before quantizing, rather than treating this as a shape bug.");
    }
  }

  void readFloats(std::vector<float> &destination, const std::string &name) {
    readExact(reinterpret_cast<char *>(destination.data()),
              checkedMultiply(destination.size(), sizeof(float), name), name);
  }

  /** @brief Destination of a column block in a full k-major WH image. */
  struct WhDestination {
    std::streampos tensor_start;
    size_t full_columns;
    size_t column_begin;
  };

  void writeQuantized(const std::vector<float> &source, size_t rows,
                      size_t columns, DType dtype, bool repack,
                      const std::string &name,
                      const WhDestination *wh_destination = nullptr) {
    const size_t output_size =
      quantizedSize(dtype, rows, columns, repack, name);
    if (dtype == DType::FP32) {
      throw std::invalid_argument("Internal FP32 quantization request for " +
                                  name);
    }

    std::vector<char> quantized(output_size);
    size_t written = 0;
    if (dtype == DType::Q4_0) {
      if (repack) {
        std::vector<char> plain(output_size);
        written = nntrainer::quantize_q4_0(
          source.data(), plain.data(), static_cast<int64_t>(rows),
          static_cast<int64_t>(columns), nullptr);
        nntrainer::repack_q4_0(quantized.data(), plain.data(), output_size,
                               static_cast<unsigned int>(rows),
                               static_cast<unsigned int>(columns), target_isa_);
      } else {
        written = nntrainer::quantize_q4_0(
          source.data(), quantized.data(), static_cast<int64_t>(rows),
          static_cast<int64_t>(columns), nullptr);
      }
    } else if (dtype == DType::Q4_K) {
      if (repack) {
        std::vector<char> plain(output_size);
        written = nntrainer::quantize_q4_K(
          source.data(), plain.data(), static_cast<int64_t>(rows),
          static_cast<int64_t>(columns), nullptr);
        nntrainer::repack_q4_K(quantized.data(), plain.data(), output_size,
                               static_cast<unsigned int>(rows),
                               static_cast<unsigned int>(columns));
      } else {
        written = nntrainer::quantize_q4_K(
          source.data(), quantized.data(), static_cast<int64_t>(rows),
          static_cast<int64_t>(columns), nullptr);
      }
    } else if (dtype == DType::Q6_K) {
      written = nntrainer::quantize_q6_K(
        source.data(), quantized.data(), static_cast<int64_t>(rows),
        static_cast<int64_t>(columns), nullptr);
    } else if (dtype == DType::QS4CX) {
      // QS4CX stores every packed nibble first and every per-channel scale
      // after, for the whole tensor (layer_devel.h). A chunked write would
      // interleave the two, so only the nibbles go out now and the scales
      // are held until flushQs4cxScales() closes the tensor. That buffer is
      // one float per output channel -- 256 KiB for a 65k-row embedding.
      std::vector<char> nibbles(
        checkedMultiply(rows, qs4cxRowBytes(columns), name));
      const size_t scale_begin = pending_scales_.size();
      pending_scales_.resize(scale_begin + rows);
      nntrainer::quant_qs4cx_f32(
        rows, columns, const_cast<float *>(source.data()), nibbles.data(),
        pending_scales_.data() + scale_begin,
        /*is_nxk=*/true);
      writeBytes(nibbles.data(), nibbles.size(), name);
      return;
    } else if (dtype == DType::QS4CX_WH) {
      // Same quantizer, then a repack. Going through quant_qs4cx_f32 rather
      // than quantizing straight into tiles keeps this bit-identical to the
      // QS4CX path above: the values and scales are the ones the device has
      // been running, and only where the nibbles sit changes.
      std::vector<char> nibbles(
        checkedMultiply(rows, qs4cxRowBytes(columns), name));
      const size_t scale_begin = pending_scales_.size();
      pending_scales_.resize(scale_begin + rows);
      nntrainer::quant_qs4cx_f32(
        rows, columns, const_cast<float *>(source.data()), nibbles.data(),
        pending_scales_.data() + scale_begin,
        /*is_nxk=*/true);

      // Unpack to one sign-extended int8 per value in K x N row-major, which
      // is what whPack takes and what htp_qs4cx_from_packed builds at load
      // time today -- same nibble convention: q + 8 stored unsigned, even k
      // in the low half. The column sums come out of the same pass, because
      // recomputing them from packed nibbles at load costs about 20 seconds
      // across this model.
      const size_t K = columns, N = rows;
      std::vector<int8_t> rm(checkedMultiply(K, N, name));
      const size_t colsum_begin = pending_colsums_.size();
      pending_colsums_.resize(colsum_begin + N, 0.0f);
      const size_t stride = qs4cxRowBytes(columns);
      for (size_t n = 0; n < N; ++n) {
        const uint8_t *row =
          reinterpret_cast<const uint8_t *>(nibbles.data()) + n * stride;
        int32_t sum = 0;
        for (size_t k = 0; k < K; ++k) {
          const uint8_t byte = row[k >> 1];
          const uint8_t nibble = (k & 1u) ? (byte >> 4) : (byte & 0x0Fu);
          const int8_t q =
            static_cast<int8_t>(static_cast<int32_t>(nibble) - 8);
          rm[k * N + n] = q;
          sum += q;
        }
        // Held as float so it travels beside the scales in one array and
        // lands in the file as the f32 the loader reads; the values are
        // small integers and exact.
        pending_colsums_[colsum_begin + n] = static_cast<float>(sum);
      }

      std::vector<uint8_t> wh(nntrainer::whBytes(K, N));
      nntrainer::whPack(rm.data(), static_cast<uint32_t>(K),
                        static_cast<uint32_t>(N), wh.data());
      if (wh_destination) {
        // Output-channel blocks are independent quantizations. Scatter their
        // complete WH tiles to the final k-major offsets rather than append
        // block-major images; no whole-weight buffer is needed.
        const size_t tile_row_bytes = (N / 32) * nntrainer::WH_TILE_BYTES;
        for (size_t kt = 0; kt < K / 32; ++kt) {
          const size_t tile_offset = kt * (wh_destination->full_columns / 32) +
                                     wh_destination->column_begin / 32;
          output_.seekp(wh_destination->tensor_start +
                        static_cast<std::streamoff>(tile_offset *
                                                    nntrainer::WH_TILE_BYTES));
          if (!output_)
            throw std::runtime_error("Failed to seek WH output for " + name);
          writeBytes(wh.data() + kt * tile_row_bytes, tile_row_bytes, name);
        }
      } else {
        writeBytes(wh.data(), wh.size(), name);
      }
      return;
    }

    if (written != output_size) {
      throw std::runtime_error("Unexpected quantized size for " + name +
                               ": expected " + std::to_string(output_size) +
                               ", got " + std::to_string(written));
    }

    writeBytes(quantized.data(), quantized.size(), name);
  }

  /**
   * @brief Write the per-channel scales QS4CX deferred, closing the tensor
   *
   * A no-op for every other dtype, so callers can call it unconditionally
   * once they have written a tensor's last chunk.
   */
  void flushQs4cxScales(DType dtype, const std::string &name) {
    if (dtype != DType::QS4CX && dtype != DType::QS4CX_WH)
      return;
    if (pending_scales_.empty())
      return;
    writeBytes(pending_scales_.data(),
               checkedMultiply(pending_scales_.size(), sizeof(float), name),
               name);
    pending_scales_.clear();
    // Column sums follow the scales, so the tensor reads back as
    // [nibbles][scale per output channel][colsum per output channel] --
    // which is QS4CX_WH_Tensor::size(). Empty for plain QS4CX.
    if (!pending_colsums_.empty()) {
      writeBytes(pending_colsums_.data(),
                 checkedMultiply(pending_colsums_.size(), sizeof(float), name),
                 name);
      pending_colsums_.clear();
    }
  }

  void writeBlockedTransposed(size_t input_size, size_t output_size,
                              DType dtype, const std::string &name) {
    const std::streampos tensor_start = input_.tellg();
    if (tensor_start < 0) {
      throw std::runtime_error("Failed to determine input offset for " + name);
    }

    // QS4CX keeps row_alignment 1: its scale is per output channel over the
    // full input width, so rows are independent and any block size is exact.
    size_t row_alignment = 1;
    if (dtype == DType::Q4_0 || dtype == DType::QS4CX_WH)
      row_alignment = 32;
    else if (dtype == DType::Q4_K)
      row_alignment = 8;

    const size_t target_row_bytes = tensorBytes(1, input_size, name);
    size_t rows_per_block =
      std::max(row_alignment, tensor_buffer_bytes_ / target_row_bytes);
    rows_per_block -= rows_per_block % row_alignment;
    rows_per_block = std::min(rows_per_block, output_size);

    const std::streampos wh_start = output_.tellp();
    if (dtype == DType::QS4CX_WH && wh_start < 0)
      throw std::runtime_error("Failed to determine WH output offset for " +
                               name);
    std::vector<float> strip(rows_per_block);
    for (size_t output_begin = 0; output_begin < output_size;) {
      const size_t block_rows =
        std::min(rows_per_block, output_size - output_begin);
      std::vector<float> transposed(
        tensorElements(block_rows, input_size, name));

      for (size_t input = 0; input < input_size; ++input) {
        const size_t element_offset =
          checkedMultiply(input, output_size, name) + output_begin;
        const size_t byte_offset =
          checkedMultiply(element_offset, sizeof(float), name);
        input_.clear();
        input_.seekg(tensor_start + static_cast<std::streamoff>(byte_offset));
        if (!input_) {
          throw std::runtime_error("Failed to seek while reading " + name);
        }
        readExact(reinterpret_cast<char *>(strip.data()),
                  checkedMultiply(block_rows, sizeof(float), name), name);
        for (size_t output = 0; output < block_rows; ++output) {
          transposed[output * input_size + input] = strip[output];
        }
      }

      const WhDestination destination{wh_start, output_size, output_begin};
      writeQuantized(transposed, block_rows, input_size, dtype, true, name,
                     dtype == DType::QS4CX_WH ? &destination : nullptr);
      output_begin += block_rows;
    }
    if (dtype == DType::QS4CX_WH) {
      output_.seekp(wh_start + static_cast<std::streamoff>(
                                 nntrainer::whBytes(input_size, output_size)));
      if (!output_)
        throw std::runtime_error("Failed to seek WH scales for " + name);
    }
    flushQs4cxScales(dtype, name);

    const size_t source_bytes = tensorBytes(input_size, output_size, name);
    input_.clear();
    input_.seekg(tensor_start + static_cast<std::streamoff>(source_bytes));
    if (!input_) {
      throw std::runtime_error("Failed to advance past " + name);
    }
  }

  std::ifstream &input_;
  std::ofstream &output_;
  ml::train::ISA target_isa_;
  size_t source_bytes_ = 0;
  bool dry_run_ = false;
  size_t expected_input_bytes_ = 0;
  /** The input's tensors in payload order, or null for a .bin. */
  const std::vector<SourceTensor> *source_tensors_ = nullptr;
  size_t tensor_buffer_bytes_;
  /** How many of them the layout walk has consumed. */
  size_t source_cursor_ = 0;
  /** QS4CX_WH per-output-channel column sums awaiting flushQs4cxScales() */
  std::vector<float> pending_colsums_;
  /** QS4CX per-channel scales awaiting flushQs4cxScales() */
  std::vector<float> pending_scales_;
  /** Actual output offsets, published only after successful conversion. */
  json prepared_fc_weights_ = json::array();
};

void validateSourceConfig(const json &nntr_cfg) {
  const auto requireFp32 = [&nntr_cfg](const std::string &key) {
    if (nntr_cfg.contains(key) &&
        upper(nntr_cfg[key].get<std::string>()) != "FP32") {
      throw std::runtime_error("Source " + key + " must be FP32");
    }
  };

  requireFp32("fc_layer_dtype");
  requireFp32("embedding_dtype");
  requireFp32("lmhead_dtype");
  requireFp32("router_dtype");
  if (nntr_cfg.contains("model_tensor_type") &&
      upper(nntr_cfg["model_tensor_type"].get<std::string>()) != "FP32-FP32") {
    throw std::runtime_error(
      "Source model_tensor_type must be FP32-FP32 for streaming quantization");
  }
}

void writeQwen3Moe(TensorWriter &writer, const Qwen3MoePlan &model,
                   const QuantizationPlan &quant) {
  writer.writeEmbedding(model.vocab_size, model.hidden_size,
                        quant.embedding_dtype, "embedding0");

  const size_t query_width =
    checkedMultiply(model.num_attention_heads, model.head_dim, "query width");
  const size_t kv_width =
    checkedMultiply(model.num_key_value_heads, model.head_dim, "KV width");

  for (size_t layer = 0; layer < model.num_layers; ++layer) {
    const std::string prefix = "layer" + std::to_string(layer);
    writer.copyFp32(model.hidden_size, prefix + "_attention_norm");
    writer.writeFc(model.hidden_size, query_width, quant.fc_dtype,
                   prefix + "_wq");
    writer.copyFp32(model.head_dim, prefix + "_q_norm");
    writer.writeFc(model.hidden_size, kv_width, quant.fc_dtype, prefix + "_wk");
    writer.copyFp32(model.head_dim, prefix + "_k_norm");
    writer.writeFc(model.hidden_size, kv_width, quant.fc_dtype, prefix + "_wv");
    writer.writeFc(query_width, model.hidden_size, quant.fc_dtype,
                   prefix + "_attention_out");
    writer.copyFp32(model.hidden_size, prefix + "_ffn_norm");

    // The router is explicitly FP32 in qwen_moe_layer::finalize(). Keeping it
    // unquantized also avoids invalid Q4_0 shapes such as [hidden, 128].
    writer.copyFp32(
      tensorElements(model.hidden_size, model.num_experts, prefix + "_router"),
      prefix + "_router");

    for (size_t expert = 0; expert < model.num_experts; ++expert) {
      const std::string expert_prefix =
        prefix + "_expert" + std::to_string(expert);
      writer.writeFc(model.hidden_size, model.intermediate_size, quant.fc_dtype,
                     expert_prefix + "_up");
      writer.writeFc(model.hidden_size, model.intermediate_size, quant.fc_dtype,
                     expert_prefix + "_gate");
      writer.writeFc(model.intermediate_size, model.hidden_size, quant.fc_dtype,
                     expert_prefix + "_down");
    }

    if (!writer.isDryRun()) {
      std::cout << "  Quantized layer " << layer + 1 << "/" << model.num_layers
                << '\n';
    }
  }

  writer.copyFp32(model.hidden_size, "output_norm");
  if (!model.tied_embeddings) {
    writer.writeFc(model.hidden_size, model.vocab_size, quant.lmhead_dtype,
                   "output_of_causallm");
  }
  writer.requireEndOfFile();
}

void writeGemma4Moe(TensorWriter &writer, const Gemma4MoePlan &model,
                    const QuantizationPlan &quant) {
  writer.writeEmbedding(model.vocab_size, model.hidden_size,
                        quant.embedding_dtype, "embedding0");

  for (size_t layer = 0; layer < model.num_layers; ++layer) {
    const std::string prefix = "layer" + std::to_string(layer);
    const bool is_sliding = model.layer_types[layer] == "sliding_attention";
    const size_t head_dim = is_sliding ? model.head_dim : model.global_head_dim;
    const size_t kv_heads = is_sliding || !model.attention_k_eq_v
                              ? model.num_key_value_heads
                              : model.num_global_key_value_heads;
    const size_t query_width = checkedMultiply(
      model.num_attention_heads, head_dim, prefix + " query width");
    const size_t kv_width =
      checkedMultiply(kv_heads, head_dim, prefix + " KV width");

    writer.copyFp32(model.hidden_size, prefix + "_attention_norm");
    writer.writeFc(model.hidden_size, query_width, quant.fc_dtype,
                   prefix + "_wq");
    writer.copyFp32(head_dim, prefix + "_q_norm");
    writer.writeFc(model.hidden_size, kv_width, quant.fc_dtype, prefix + "_wk");
    writer.copyFp32(head_dim, prefix + "_k_norm");
    if (!model.attention_k_eq_v || is_sliding) {
      writer.writeFc(model.hidden_size, kv_width, quant.fc_dtype,
                     prefix + "_wv");
    }
    writer.writeFc(query_width, model.hidden_size, quant.fc_dtype,
                   prefix + "_attention_out");

    writer.copyFp32(model.hidden_size, prefix + "_post_attention_norm");
    writer.copyFp32(model.hidden_size, prefix + "_pre_ffn_norm");
    writer.writeFc(model.hidden_size, model.intermediate_size, quant.fc_dtype,
                   prefix + "_ffn_gate");
    writer.writeFc(model.hidden_size, model.intermediate_size, quant.fc_dtype,
                   prefix + "_ffn_up");
    writer.writeFc(model.intermediate_size, model.hidden_size, quant.fc_dtype,
                   prefix + "_ffn_down");

    // The MoE half of the block, in the compiled graph's order (checked by
    // unittest_causallm_gemma4_moe) that res/gemma4/weight_converter.py
    // writes too: three norms (the router's scale and
    // hidden^-0.5 folded into _router_norm), then the lfm2_moe layer's
    // weights as writeLfm2Moe has them -- router gate and the [E] vector
    // FP32 (Lfm2MoELayer::save never quantizes them), then per expert the
    // fused gate|up and the down projection in the MoE dtype, which is what
    // lets --moe_dtype QS4CX_WH stream them onto the HTP (doc 55).
    writer.copyFp32(model.hidden_size, prefix + "_post_ffn_norm_1");
    writer.copyFp32(model.hidden_size, prefix + "_pre_ffn_norm_2");
    writer.copyFp32(model.hidden_size, prefix + "_router_norm");
    writer.copyFp32(
      tensorElements(model.hidden_size, model.num_experts, prefix + "_router"),
      prefix + "_router");
    writer.copyFp32(model.num_experts, prefix + "_per_expert_scale");

    for (size_t expert = 0; expert < model.num_experts; ++expert) {
      const std::string expert_prefix =
        prefix + "_expert" + std::to_string(expert);
      writer.writeFc(
        model.hidden_size,
        checkedMultiply(2, model.moe_intermediate_size, expert_prefix),
        quant.moe_dtype, expert_prefix + "_gate_up");
      writer.writeFc(model.moe_intermediate_size, model.hidden_size,
                     quant.moe_dtype, expert_prefix + "_down");
    }

    writer.copyFp32(model.hidden_size, prefix + "_post_ffn_norm_2");
    writer.copyFp32(model.hidden_size, prefix + "_post_ffn_norm");

    if (model.per_layer_input_size != 0) {
      writer.writeFc(model.hidden_size, model.per_layer_input_size,
                     quant.fc_dtype, prefix + "_per_layer_input_gate");
      if (layer == 0) {
        const size_t total_per_layer_size =
          checkedMultiply(model.num_layers, model.per_layer_input_size,
                          "per-layer input total size");
        writer.writeEmbedding(model.per_layer_vocab_size, total_per_layer_size,
                              quant.fc_dtype, "per_layer_input_embedding");
        writer.writeFc(model.hidden_size, total_per_layer_size, quant.fc_dtype,
                       "per_layer_input_projection");
        writer.copyFp32(model.per_layer_input_size,
                        "per_layer_projection_norm");
      }
      writer.writeFc(model.per_layer_input_size, model.hidden_size,
                     quant.fc_dtype, prefix + "_per_layer_input_proj");
      writer.copyFp32(model.hidden_size, prefix + "_post_per_layer_input_norm");
    }
    writer.copyFp32(1, prefix + "_layer_scalar");

    if (!writer.isDryRun()) {
      std::cout << "  Quantized layer " << layer + 1 << "/" << model.num_layers
                << '\n';
    }
  }

  writer.copyFp32(model.hidden_size, "output_norm");
  if (model.tied_embeddings) {
    // The Gemma4 converter retains a legacy trailing shared embedding, while
    // NNTrainer model saving emits only embedding0. Accept either FP32 source.
    if (writer.hasRemainingBytes()) {
      writer.discardFp32(tensorElements(model.vocab_size, model.hidden_size,
                                        "tied output_of_causallm"),
                         "tied output_of_causallm");
    }
  } else {
    writer.writeFc(model.hidden_size, model.vocab_size, quant.lmhead_dtype,
                   "output_of_causallm");
  }
  writer.requireEndOfFile();
}

/**
 * The tensor order below mirrors, in this order:
 *   - the graph Lfm2CausalLM::constructModel builds (embedding, then one
 *     conv or attention block per layer_types entry, then output_norm and
 *     the LM head),
 *   - Lfm2MoELayer::finalize's weight request order for the MoE FFN
 *     (router gate, expert_bias, then per expert the fused gate|up and the
 *     down projection),
 *   - and res/lfm2_moe/lfm2-8b-a1b/weight_converter.py, which is what
 *     actually writes the FP32 .bin this reads.
 * All three have to agree; the converter is the authority when they differ,
 * because it produced the bytes.
 */
template <typename Writer>
void writeLfm2Moe(Writer &writer, const Lfm2MoePlan &model,
                  const QuantizationPlan &quant) {
  writer.writeEmbedding(model.vocab_size, model.hidden_size,
                        quant.embedding_dtype, "embedding0");

  const size_t query_width =
    checkedMultiply(model.num_attention_heads, model.head_dim, "query width");
  const size_t kv_width =
    checkedMultiply(model.num_key_value_heads, model.head_dim, "KV width");

  for (size_t layer = 0; layer < model.num_layers; ++layer) {
    const std::string prefix = "layer" + std::to_string(layer);

    // operator_norm: _conv_norm on a conv layer, _attention_norm otherwise.
    writer.copyFp32(model.hidden_size, prefix + "_operator_norm");

    if (model.layer_types[layer] == "conv") {
      // createConvBlock: in_proj expands to 3*conv_dim (gate_a|gate_b|gate_c),
      // the depthwise kernel is [conv_L_cache, conv_dim] and explicitly FP32
      // (weight_dtype=FP32 in the graph), out_proj comes back to hidden_size.
      writer.writeFc(model.hidden_size,
                     checkedMultiply(3, model.conv_dim, prefix + "_conv_in"),
                     quant.fc_dtype, prefix + "_conv_in_proj");
      writer.copyFp32(
        tensorElements(model.conv_l_cache, model.conv_dim, prefix + "_conv"),
        prefix + "_conv_conv");
      writer.writeFc(model.conv_dim, model.hidden_size, quant.fc_dtype,
                     prefix + "_conv_out_proj");
    } else {
      writer.writeFc(model.hidden_size, query_width, quant.fc_dtype,
                     prefix + "_wq");
      writer.copyFp32(model.head_dim, prefix + "_q_norm");
      writer.writeFc(model.hidden_size, kv_width, quant.fc_dtype,
                     prefix + "_wk");
      writer.copyFp32(model.head_dim, prefix + "_k_norm");
      writer.writeFc(model.hidden_size, kv_width, quant.fc_dtype,
                     prefix + "_wv");
      writer.writeFc(query_width, model.hidden_size, quant.fc_dtype,
                     prefix + "_attention_out");
    }

    writer.copyFp32(model.hidden_size, prefix + "_ffn_norm");

    if (layer < model.num_dense_layers) {
      // Dense SwiGLU FFN, in NNTrainer's historical up, gate, down order.
      writer.writeFc(model.hidden_size, model.intermediate_size, quant.fc_dtype,
                     prefix + "_ffn_up");
      writer.writeFc(model.hidden_size, model.intermediate_size, quant.fc_dtype,
                     prefix + "_ffn_gate");
      writer.writeFc(model.intermediate_size, model.hidden_size, quant.fc_dtype,
                     prefix + "_ffn_down");
    } else {
      // Router dtype is independent of the expert and dense FC dtypes.
      writer.writeFc(model.hidden_size, model.num_experts, quant.router_dtype,
                     prefix + "_router");
      writer.copyFp32(model.num_experts, prefix + "_expert_bias");

      for (size_t expert = 0; expert < model.num_experts; ++expert) {
        const std::string expert_prefix =
          prefix + "_expert" + std::to_string(expert);
        writer.writeFc(
          model.hidden_size,
          checkedMultiply(2, model.moe_intermediate_size, expert_prefix),
          quant.moe_dtype, expert_prefix + "_gate_up");
        writer.writeFc(model.moe_intermediate_size, model.hidden_size,
                       quant.moe_dtype, expert_prefix + "_down");
      }
    }

    if (!writer.isDryRun()) {
      std::cout << "  Quantized layer " << layer + 1 << "/" << model.num_layers
                << '\n';
    }
  }

  writer.copyFp32(model.hidden_size, "output_norm");
  if (!model.tied_embeddings) {
    writer.writeFc(model.hidden_size, model.vocab_size, quant.lmhead_dtype,
                   "output_of_causallm");
  }
  writer.requireEndOfFile();
}

std::string stripKnownDtypeSuffix(std::string base) {
  const std::vector<std::string> suffixes = {"_fp32", "_fp16", "_q40", "_q4_0",
                                             "_q4k",  "_q4_k", "_q6k", "_q6_k"};
  for (const auto &suffix : suffixes) {
    if (base.size() >= suffix.size() &&
        base.compare(base.size() - suffix.size(), suffix.size(), suffix) == 0) {
      base.resize(base.size() - suffix.size());
      break;
    }
  }
  return base;
}

std::string defaultOutputBin(const std::string &input_bin,
                             const QuantizationPlan &quant) {
  std::filesystem::path input_path(input_bin);
  std::string base = stripKnownDtypeSuffix(input_path.stem().string());
  std::string result = base + "_" + dtypeSuffix(quant.fc_dtype);
  if (quant.embedding_dtype != quant.fc_dtype)
    result += "_embd" + dtypeSuffix(quant.embedding_dtype);
  if (quant.lmhead_dtype != quant.embedding_dtype)
    result += "_lmhead" + dtypeSuffix(quant.lmhead_dtype);
  result += "_" + lower(isaName(quant.target_isa)) + ".bin";
  return result;
}

std::filesystem::path
defaultOutputDirectory(const std::filesystem::path &model_dir,
                       const QuantizationPlan &quant) {
  return model_dir /
         ("quantized_fc_" + lower(dtypeName(quant.fc_dtype)) + "_embd_" +
          lower(dtypeName(quant.embedding_dtype)) + "_lmhead_" +
          lower(dtypeName(quant.lmhead_dtype)) + "_isa_" +
          lower(isaName(quant.target_isa)) +
          (quant.router_dtype == DType::FP32 ? "" : "_router_qs4cx"));
}

void copyAuxiliaryFiles(const std::filesystem::path &model_dir,
                        const std::filesystem::path &output_dir) {
  const char *files[] = {
    "config.json",           "generation_config.json",  "tokenizer.json",
    "tokenizer_config.json", "special_tokens_map.json", "tokenizer.model",
    "spiece.model",          "sentencepiece.bpe.model", "modules.json"};
  for (const char *file : files) {
    const std::filesystem::path source = model_dir / file;
    if (std::filesystem::exists(source)) {
      std::filesystem::copy_file(
        source, output_dir / file,
        std::filesystem::copy_options::overwrite_existing);
    }
  }
}

void writeOutputConfig(const std::filesystem::path &model_dir,
                       const std::filesystem::path &output_dir,
                       const std::string &output_bin,
                       const QuantizationPlan &quant, json nntr_cfg) {
  nntr_cfg["model_file_name"] = output_bin;
  nntr_cfg["model_tensor_type"] =
    std::string(dtypeName(quant.fc_dtype)) + "-FP32";
  nntr_cfg["fc_layer_dtype"] = dtypeName(quant.fc_dtype);
  nntr_cfg["embedding_dtype"] = dtypeName(quant.embedding_dtype);
  nntr_cfg["lmhead_dtype"] = dtypeName(quant.lmhead_dtype);
  // Lfm2MoeCausalLM::setupParameters (and Qwen3's/Gemma4's MoE equivalents)
  // read this key with a fallback to fc_layer_dtype -- writeLfm2Moe already
  // writes the expert tensors in quant.moe_dtype's byte layout, so omitting
  // this key here silently mismatches: a --moe_dtype QS4CX run's .bin holds
  // QS4CX bytes while the loader tags those weights Q4_0 (fc_layer_dtype's
  // value) and dispatches Q4_0 kernels against them -- a byte-layout
  // mismatch, not just a missed optimization.
  nntr_cfg["moe_layer_dtype"] = dtypeName(quant.moe_dtype);
  nntr_cfg["router_dtype"] = dtypeName(quant.router_dtype);

  const std::filesystem::path output_config =
    std::filesystem::equivalent(model_dir, output_dir)
      ? output_dir / "nntr_config_quantized.json"
      : output_dir / "nntr_config.json";
  std::ofstream file(output_config);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open " + output_config.string());
  }
  file << nntr_cfg.dump(4) << '\n';
}

/** Copy an already quantized LFM2 file, replacing only its FP32 routers.
 * Other tensor payloads remain byte-identical, including QS2CX_WH headers.
 * Reuse the model's tensor walk so a new layout cannot silently skip weights.
 */
class RouterOnlyWriter {
public:
  RouterOnlyWriter(std::ifstream &input, std::ofstream &output) :
    input_(input), output_(output), buffer_(4u * 1024u * 1024u) {}
  bool isDryRun() const { return false; }
  void copyFp32(size_t count, const std::string &name) {
    copy(checkedMultiply(count, sizeof(float), name), name);
  }
  void writeEmbedding(size_t rows, size_t cols, DType dtype,
                      const std::string &name) {
    copy(quantizedSize(dtype, rows, cols, false, name), name);
  }
  void writeFc(size_t K, size_t N, DType dtype, const std::string &name) {
    if (name.size() >= 7 && name.compare(name.size() - 7, 7, "_router") == 0) {
      TensorWriter writer(input_, output_, ml::train::ISA::DEFAULT, 0, false);
      writer.writeFc(K, N, DType::QS4CX, name);
      return;
    }
    if (dtype == DType::QS2CX_WH) {
      if (!K || !N || K % 32 || N % 32)
        throw std::invalid_argument("Invalid QS2CX_WH shape: " + name);
      copy(checkedMultiply(K, N, name) / 4 + 4 + checkedMultiply(N, 8, name),
           name);
    } else {
      copy(quantizedSize(dtype, N, K, true, name), name);
    }
  }
  void requireEndOfFile() {
    if (input_.peek() != std::char_traits<char>::eof() || input_.bad())
      throw std::runtime_error("Router conversion: unexpected trailing input");
  }

private:
  void copy(size_t bytes, const std::string &name) {
    while (bytes) {
      const size_t n = std::min(bytes, buffer_.size());
      input_.read(buffer_.data(), n);
      if (input_.gcount() != static_cast<std::streamsize>(n))
        throw std::runtime_error("Router conversion: truncated " + name);
      output_.write(buffer_.data(), n);
      if (!output_)
        throw std::runtime_error("Router conversion: write failed at " + name);
      bytes -= n;
    }
  }
  std::ifstream &input_;
  std::ofstream &output_;
  std::vector<char> buffer_;
};

int convertRouterOnly(const std::filesystem::path &model_dir,
                      std::filesystem::path output_dir, std::string output_bin,
                      const json &cfg, json nntr_cfg) {
  const auto model = makeLfm2MoePlan(cfg);
  if (nntr_cfg.value("router_dtype", std::string("FP32")) != "FP32")
    throw std::invalid_argument("--router_only requires an FP32 source router");
  const auto source =
    model_dir / nntr_cfg.at("model_file_name").get<std::string>();
  if (source.extension() != ".bin")
    throw std::invalid_argument("--router_only requires a .bin model");
  if (output_dir.empty())
    output_dir = model_dir / "router_qs4cx";
  if (output_bin.empty())
    output_bin = source.stem().string() + "_router_qs4cx.bin";
  if (std::filesystem::path(output_bin).filename() != output_bin)
    throw std::invalid_argument("output_bin must be a filename");
  std::filesystem::create_directories(output_dir);
  const auto target = output_dir / output_bin;
  const auto temporary = output_dir / (output_bin + ".partial");
  const auto config_tmp = output_dir / "nntr_config.json.partial";
  if (std::filesystem::exists(target) || std::filesystem::exists(temporary) ||
      std::filesystem::exists(config_tmp) ||
      std::filesystem::exists(output_dir / "nntr_config.json"))
    throw std::invalid_argument(
      "Router conversion requires unused output files");
  auto storedType = [&](const char *key, const std::string &fallback) {
    const auto value = upper(nntr_cfg.value(key, fallback));
    return value == "QS2CX_WH" ? DType::QS2CX_WH : parseDType(value);
  };
  const auto fc = nntr_cfg.at("fc_layer_dtype").get<std::string>();
  const auto emb = nntr_cfg.at("embedding_dtype").get<std::string>();
  const QuantizationPlan plan{
    storedType("fc_layer_dtype", fc), storedType("embedding_dtype", emb),
    storedType("lmhead_dtype", emb),  storedType("moe_layer_dtype", fc),
    ml::train::ISA::DEFAULT,          DType::QS4CX};
  if (model.tied_embeddings && plan.embedding_dtype != plan.lmhead_dtype)
    throw std::invalid_argument("Tied embedding and LM head dtypes disagree");
  std::ifstream input(source, std::ios::binary);
  if (!input)
    throw std::runtime_error("Cannot read " + source.string());
  try {
    std::ofstream output(temporary, std::ios::binary);
    if (!output)
      throw std::runtime_error("Cannot write " + temporary.string());
    RouterOnlyWriter writer(input, output);
    writeLfm2Moe(writer, model, plan);
    output.close();
    if (!output)
      throw std::runtime_error("Failed to finalize router model");
    copyAuxiliaryFiles(model_dir, output_dir);
    nntr_cfg["router_dtype"] = "QS4CX";
    nntr_cfg["model_file_name"] = output_bin;
    std::ofstream config(config_tmp);
    config << nntr_cfg.dump(4) << '\n';
    config.close();
    if (!config)
      throw std::runtime_error("Failed to write router config");
    std::filesystem::rename(temporary, target);
    std::filesystem::rename(config_tmp, output_dir / "nntr_config.json");
  } catch (...) {
    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);
    std::filesystem::remove(config_tmp, ignored);
    throw;
  }
  std::cout << "Router QS4CX model: " << target << '\n';
  return EXIT_SUCCESS;
}

void printUsage(const char *program) {
  std::cout
    << "Usage: " << program << " <model_path> [options]\n\n"
    << "Stream-quantize a supported FP32 CausalLM .bin model without loading "
       "the full model.\n\n"
    << "Options:\n"
    << "  --output, -o <path>   Output directory (default: dtype/ISA "
       "subdirectory)\n"
    << "  --fc_dtype <type>     Attention/expert dtype (default: Q4_0)\n"
    << "  --moe_dtype <type>    MoE expert dtype (default: the fc dtype)\n"
    << "  --router_dtype <type> LFM2 router: FP32 or QS4CX (default: FP32)\n"
    << "  --router_only        Convert only FP32 routers of an existing LFM2 "
       ".bin to QS4CX\n"
    << "  --embd_dtype <type>   Embedding dtype (default: FP32)\n"
    << "  --lmhead_dtype <type> LM head dtype (default: embedding dtype)\n"
    << "  --isa <target>        DEFAULT, X86, or ARM (default: DEFAULT)\n"
    << "  --output_bin <name>   Output .bin filename\n"
    << "  --config <path>       Read target dtype fields and filename from an "
       "nntr config\n"
    << "  -h, --help            Show this help\n\n"
    << "Architectures: Qwen3MoeForCausalLM, Lfm2MoeForCausalLM, "
       "Gemma4ForCausalLM,\n"
    << "               Gemma4ForConditionalGeneration\n"
    << "Supported dtypes: FP32, Q4_0, Q4_K, Q6_K, QS4CX, QS4CX_WH\n"
    << "QS4CX_WH conv/dense FC weights emit htp_fc_manifest.json for "
       "tools/htp/prepare_fc_wh.py.\n"
    << "QS4CX_WH is QS4CX pre-arranged into HMX weight tiles: it loads "
       "without\n"
    << "the on-device conversion that costs 48% of prefill, and no CPU "
       "kernel can\n"
    << "read it, so a model using it runs its MoE experts on the HTP or "
       "not at all.\n"
    << "Gemma4 MoE FC weights currently support FP32 or Q4_0; the experts "
       "take\n"
    << "--moe_dtype like LFM2 (QS4CX_WH for the HTP expert streaming).\n";
}

int run(int argc, char **argv) {
  if (argc < 2) {
    printUsage(argv[0]);
    return EXIT_FAILURE;
  }
  if (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
    printUsage(argv[0]);
    return EXIT_SUCCESS;
  }

  const std::filesystem::path model_dir = argv[1];
  std::filesystem::path output_dir;
  std::string fc_dtype = "Q4_0";
  std::string embedding_dtype = "FP32";
  std::string lmhead_dtype;
  std::string moe_dtype;
  std::string router_dtype = "FP32";
  bool router_only = false;
  std::string target_isa = "DEFAULT";
  std::string output_bin;
  std::filesystem::path target_config;
  // Which dtypes the command line actually asked for, so --config fills in
  // the rest instead of overruling them. Without this,
  // "--config old.json --moe_dtype QS4CX_WH" quietly quantizes to whatever
  // old.json says and the run looks like it worked.
  std::set<std::string> given;

  for (int index = 2; index < argc; ++index) {
    const std::string argument = argv[index];
    const auto requireValue = [&](const std::string &option) {
      if (index + 1 >= argc)
        throw std::invalid_argument("Missing value for " + option);
      given.insert(option);
      return std::string(argv[++index]);
    };

    if (argument == "--output" || argument == "-o")
      output_dir = requireValue(argument);
    else if (argument == "--fc_dtype")
      fc_dtype = requireValue(argument);
    else if (argument == "--embd_dtype")
      embedding_dtype = requireValue(argument);
    else if (argument == "--lmhead_dtype")
      lmhead_dtype = requireValue(argument);
    else if (argument == "--moe_dtype")
      moe_dtype = requireValue(argument);
    else if (argument == "--router_only")
      router_only = true;
    else if (argument == "--router_dtype")
      router_dtype = requireValue(argument);
    else if (argument == "--isa")
      target_isa = requireValue(argument);
    else if (argument == "--output_bin")
      output_bin = requireValue(argument);
    else if (argument == "--config")
      target_config = requireValue(argument);
    else if (argument == "--help" || argument == "-h") {
      printUsage(argv[0]);
      return EXIT_SUCCESS;
    } else {
      throw std::invalid_argument("Unknown option: " + argument);
    }
  }

  const json cfg = readJson(model_dir / "config.json");
  json nntr_cfg = readJson(model_dir / "nntr_config.json");
  if (router_only) {
    for (const auto &option : given)
      if (option != "--output" && option != "-o" && option != "--output_bin" &&
          option != "--router_dtype")
        throw std::invalid_argument("--router_only cannot combine with " +
                                    option);
    if (given.count("--router_dtype") && upper(router_dtype) != "QS4CX")
      throw std::invalid_argument("--router_only targets QS4CX");
    return convertRouterOnly(model_dir, output_dir, output_bin, cfg, nntr_cfg);
  }
  validateSourceConfig(nntr_cfg);
  const std::string architecture = getArchitecture(cfg);
  const bool is_qwen3_moe = architecture == "Qwen3MoeForCausalLM";
  const bool is_gemma4_moe = architecture == "Gemma4ForCausalLM" ||
                             architecture == "Gemma4ForConditionalGeneration";
  const bool is_lfm2_moe = architecture == "Lfm2MoeForCausalLM";
  if (!is_qwen3_moe && !is_gemma4_moe && !is_lfm2_moe)
    throw std::runtime_error("Unsupported architecture: " + architecture);

  Qwen3MoePlan qwen3_model{};
  Gemma4MoePlan gemma4_model{};
  Lfm2MoePlan lfm2_model{};
  if (is_qwen3_moe)
    qwen3_model = makeQwen3MoePlan(cfg);
  else if (is_gemma4_moe)
    gemma4_model = makeGemma4MoePlan(cfg);
  else
    lfm2_model = makeLfm2MoePlan(cfg);

  const bool tied_embeddings = is_qwen3_moe    ? qwen3_model.tied_embeddings
                               : is_gemma4_moe ? gemma4_model.tied_embeddings
                                               : lfm2_model.tied_embeddings;
  const size_t num_layers = is_qwen3_moe    ? qwen3_model.num_layers
                            : is_gemma4_moe ? gemma4_model.num_layers
                                            : lfm2_model.num_layers;
  const size_t num_experts = is_qwen3_moe    ? qwen3_model.num_experts
                             : is_gemma4_moe ? gemma4_model.num_experts
                                             : lfm2_model.num_experts;
  const size_t hidden_size = is_qwen3_moe    ? qwen3_model.hidden_size
                             : is_gemma4_moe ? gemma4_model.hidden_size
                                             : lfm2_model.hidden_size;

  if (!target_config.empty()) {
    const json requested = readJson(target_config);
    // An explicit flag wins. The usual reason to pass both is to keep a
    // model's existing dtypes and change one of them.
    const auto take = [&](const char *key, const char *option,
                          std::string &field) {
      if (requested.contains(key) && given.count(option) == 0)
        field = requested[key].get<std::string>();
    };
    take("fc_layer_dtype", "--fc_dtype", fc_dtype);
    take("embedding_dtype", "--embd_dtype", embedding_dtype);
    take("lmhead_dtype", "--lmhead_dtype", lmhead_dtype);
    take("moe_layer_dtype", "--moe_dtype", moe_dtype);
    take("router_dtype", "--router_dtype", router_dtype);
    if (requested.contains("model_file_name") && output_bin.empty())
      output_bin = requested["model_file_name"].get<std::string>();
    if (requested.contains("moe_cache_size"))
      nntr_cfg["moe_cache_size"] = requested["moe_cache_size"];
  }

  if (lmhead_dtype.empty())
    lmhead_dtype = embedding_dtype;
  // Same default as nntr_quantize's --moe_dtype: the FC dtype, so a run that
  // does not ask for a split is byte-identical to one that never had the
  // option.
  if (moe_dtype.empty())
    moe_dtype = fc_dtype;
  const QuantizationPlan quant{
    parseDType(fc_dtype),     parseDType(embedding_dtype),
    parseDType(lmhead_dtype), parseDType(moe_dtype),
    parseIsa(target_isa),     parseDType(router_dtype)};

  if ((quant.router_dtype != DType::FP32 &&
       quant.router_dtype != DType::QS4CX) ||
      (!is_lfm2_moe && quant.router_dtype != DType::FP32))
    throw std::invalid_argument("Router quantization supports LFM2 QS4CX only");

  if (tied_embeddings && quant.embedding_dtype != quant.lmhead_dtype) {
    throw std::invalid_argument(
      "A tied model requires matching embedding and LM head dtypes");
  }
  // The experts take --moe_dtype; the FCs are plain fully_connected layers,
  // whose QS4CX weight the HTP registers without a Q4_0 detour.
  if (is_gemma4_moe && quant.fc_dtype != DType::FP32 &&
      quant.fc_dtype != DType::Q4_0 && quant.fc_dtype != DType::QS4CX) {
    throw std::invalid_argument(
      "Gemma4 MoE FC dtype must be FP32, Q4_0 or QS4CX");
  }
  const std::string input_bin =
    nntr_cfg.at("model_file_name").get<std::string>();
  const std::string input_ext =
    std::filesystem::path(input_bin).extension().string();
  if (input_ext != ".bin" && input_ext != ".safetensors") {
    throw std::invalid_argument("Streaming quantization requires an NNTrainer "
                                ".bin or .safetensors input");
  }
  if (output_dir.empty())
    output_dir = defaultOutputDirectory(model_dir, quant);
  std::filesystem::create_directories(output_dir);
  if (output_bin.empty())
    output_bin = defaultOutputBin(input_bin, quant);
  if (std::filesystem::path(output_bin).extension() != ".bin") {
    throw std::invalid_argument("--output_bin must use the .bin extension");
  }

  const std::filesystem::path input_path = model_dir / input_bin;
  const std::filesystem::path output_path = output_dir / output_bin;
  if (std::filesystem::exists(output_path) &&
      std::filesystem::equivalent(input_path, output_path)) {
    throw std::invalid_argument("Input and output weight paths must differ");
  }

  std::ifstream input(input_path, std::ios::binary);
  if (!input.is_open())
    throw std::runtime_error("Failed to open " + input_path.string());
  // A converter-written .safetensors is the .bin with a header in front:
  // weight_converter.py assigns data_offsets in the same walk that writes
  // the .bin, so past the header the bytes are in the .bin's order (the
  // tied lm head's trailing copy is the one thing the .bin has that this
  // does not, and hasRemainingBytes already allows for that). Skip
  // [8-byte little-endian header length][header JSON] and read on -- with
  // the header kept, so TensorWriter::expectSourceTensor can hold the walk
  // to one tensor per tensor the converter wrote.
  uintmax_t data_start = 0;
  std::vector<SourceTensor> source_tensors;
  if (input_ext == ".safetensors") {
    unsigned char len_bytes[8];
    input.read(reinterpret_cast<char *>(len_bytes), sizeof(len_bytes));
    if (!input)
      throw std::runtime_error("Failed to read the safetensors header of " +
                               input_path.string());
    uint64_t header_len = 0;
    for (int i = 7; i >= 0; --i)
      header_len = (header_len << 8) | len_bytes[i];
    data_start = 8 + header_len;
    if (data_start >= std::filesystem::file_size(input_path))
      throw std::runtime_error("safetensors header length is past the end of " +
                               input_path.string());
    std::string header_json(static_cast<size_t>(header_len), '\0');
    input.read(header_json.data(), static_cast<std::streamsize>(header_len));
    if (!input)
      throw std::runtime_error("Failed to read the safetensors header of " +
                               input_path.string());
    // Named, not a temporary: items() hands back a proxy that refers to the
    // object, and a temporary one dies before the loop reads it.
    const json header = json::parse(header_json);
    std::vector<std::pair<uint64_t, SourceTensor>> at;
    for (const auto &entry : header.items()) {
      if (entry.key() == "__metadata__")
        continue;
      const auto &offsets = entry.value().at("data_offsets");
      const uint64_t begin = offsets.at(0).get<uint64_t>();
      const uint64_t end = offsets.at(1).get<uint64_t>();
      at.emplace_back(
        begin, SourceTensor{entry.key(), static_cast<size_t>(end - begin)});
    }
    std::sort(at.begin(), at.end(),
              [](const auto &a, const auto &b) { return a.first < b.first; });
    source_tensors.reserve(at.size());
    for (auto &pair : at)
      source_tensors.push_back(std::move(pair.second));
    input.seekg(static_cast<std::streamoff>(data_start));
  }
  std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
  if (!output.is_open())
    throw std::runtime_error("Failed to open " + output_path.string());

  std::cout << "NNTrainer CausalLM streaming quantizer\n"
            << "  Architecture: " << architecture << '\n'
            << "  Source: " << input_path << '\n'
            << "  Target: " << output_path << '\n'
            << "  Layers: " << num_layers << '\n'
            << "  Experts per layer: " << num_experts << '\n'
            << "  FC dtype: " << dtypeName(quant.fc_dtype) << '\n'
            << "  MoE dtype: " << dtypeName(quant.moe_dtype) << '\n'
            << "  Embedding dtype: " << dtypeName(quant.embedding_dtype) << '\n'
            << "  LM head dtype: " << dtypeName(quant.lmhead_dtype) << '\n'
            << "  Target ISA: " << isaName(quant.target_isa) << '\n';

  const auto walk = [&](TensorWriter &tensor_writer) {
    if (is_qwen3_moe)
      writeQwen3Moe(tensor_writer, qwen3_model, quant);
    else if (is_gemma4_moe)
      writeGemma4Moe(tensor_writer, gemma4_model, quant);
    else
      writeLfm2Moe(tensor_writer, lfm2_model, quant);
  };

  // Every shape below comes from config.json, while the FP32 .bin was written
  // from the checkpoint's actual tensors. When the two disagree, the run
  // otherwise dies on an EOF tens of GB and tens of minutes in, naming a
  // tensor that is not itself the problem. Walk the layout first with the
  // same code, consuming nothing, and compare the total against the file.
  const uintmax_t source_bytes =
    std::filesystem::file_size(input_path) - data_start;
  TensorWriter probe(input, output, quant.target_isa,
                     static_cast<size_t>(source_bytes), /*dry_run=*/true,
                     source_tensors.empty() ? nullptr : &source_tensors);
  walk(probe);
  if (probe.expectedInputBytes() != source_bytes) {
    const bool file_is_short = probe.expectedInputBytes() > source_bytes;
    const uintmax_t difference = file_is_short
                                   ? probe.expectedInputBytes() - source_bytes
                                   : source_bytes - probe.expectedInputBytes();
    const size_t row_bytes =
      checkedMultiply(hidden_size, sizeof(float), "hidden row");
    std::string hint;
    if (row_bytes != 0 && difference % row_bytes == 0) {
      hint = " That difference is exactly " +
             std::to_string(difference / row_bytes) + " rows of hidden_size (" +
             std::to_string(hidden_size) + ").";
    }
    throw std::runtime_error(
      "config.json describes " + std::to_string(probe.expectedInputBytes()) +
      " bytes of FP32 weights, but " + input_path.string() + " is " +
      std::to_string(source_bytes) + " bytes -- " + std::to_string(difference) +
      " bytes " + (file_is_short ? "short" : "extra") + "." + hint +
      " Either the conversion did not run to completion, or a config.json "
      "field disagrees with the checkpoint's real tensor shapes (vocab_size, "
      "intermediate_size, moe_intermediate_size, num_experts, "
      "num_dense_layers, num_key_value_heads, head_dim, layer_types). Compare "
      "the shapes weight_converter.py printed against those fields; the "
      "converter writes the checkpoint's actual tensors, this tool trusts "
      "config.json.");
  }

  TensorWriter writer(input, output, quant.target_isa, 0, /*dry_run=*/false,
                      source_tensors.empty() ? nullptr : &source_tensors);
  walk(writer);
  output.close();
  if (!output)
    throw std::runtime_error("Failed to finalize " + output_path.string());

  if (!std::filesystem::equivalent(model_dir, output_dir))
    copyAuxiliaryFiles(model_dir, output_dir);
  writeOutputConfig(model_dir, output_dir, output_bin, quant, nntr_cfg);

  const uintmax_t input_size = source_bytes;
  const uintmax_t output_size = std::filesystem::file_size(output_path);
  if (!writer.preparedFcWeights().empty()) {
    // The walk has checked EOF and the model/config writes have succeeded.
    // Never publish a manifest for a partial or failed conversion.
    const auto manifest_path =
      output_path.parent_path() / "htp_fc_manifest.json";
    const auto temporary_path = manifest_path.string() + ".tmp";
    const json manifest = {{"schema_version", 1},
                           {"dtype", "QS4CX_WH"},
                           {"model_size", output_size},
                           {"weights", writer.preparedFcWeights()}};
    std::ofstream manifest_file(temporary_path, std::ios::trunc);
    manifest_file << manifest.dump(2) << '\n';
    manifest_file.close();
    if (!manifest_file)
      throw std::runtime_error("Failed to write " + temporary_path);
    // std::filesystem::rename cannot replace an existing file on Windows.
    if (std::filesystem::exists(manifest_path))
      std::filesystem::remove(manifest_path);
    std::filesystem::rename(temporary_path, manifest_path);
    std::cout << "  Offline HTP FC manifest: " << manifest_path << '\n';
  }
  const double ratio = input_size == 0
                         ? 0.0
                         : 100.0 * static_cast<double>(output_size) /
                             static_cast<double>(input_size);
  std::cout << "  Source size: " << input_size / (1024 * 1024) << " MiB\n"
            << "  Output size: " << output_size / (1024 * 1024) << " MiB\n"
            << "  Output/source: " << std::fixed << std::setprecision(1)
            << ratio << "%\n"
            << "Streaming quantization complete\n";
  return EXIT_SUCCESS;
}

} // namespace

int main(int argc, char **argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception &error) {
    std::cerr << "[!] FATAL ERROR: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
