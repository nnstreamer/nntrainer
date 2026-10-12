// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 * @brief Checks blocked WH serialization and offline FC manifests using the
 *        real CPU quantizer. Link with libnntrainer; pass an empty work dir.
 */
#define main nntr_quantize_stream_main
#include "../../../Applications/CausalLM/quantize_stream.cpp"
#undef main

#include <iterator>

namespace {
void check(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

std::vector<char> contents(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  return {(std::istreambuf_iterator<char>(input)), {}};
}

json convertTensor(const std::filesystem::path &source,
                   const std::filesystem::path &target, size_t budget) {
  std::ifstream input(source, std::ios::binary);
  std::ofstream output(target, std::ios::binary);
  TensorWriter writer(input, output, ml::train::ISA::DEFAULT, 0, false, nullptr,
                      budget);
  writer.copyFp32(7, "prefix");
  writer.writeFc(64, 224, DType::QS4CX_WH, "layer0_ffn_up");
  writer.copyFp32(9, "middle");
  writer.writeFc(224, 64, DType::QS4CX_WH, "layer0_ffn_down");
  writer.copyFp32(5, "suffix");
  writer.requireEndOfFile();
  output.close();
  check(bool(output), "failed to finalize tensor fixture");
  return writer.preparedFcWeights();
}

void blockedMatchesWhole(const std::filesystem::path &directory) {
  const auto source = directory / "source.bin";
  std::vector<float> values(7 + 64 * 224 + 9 + 224 * 64 + 5);
  for (size_t i = 0; i < values.size(); ++i)
    values[i] = (int((i * 17 + i / 31) % 137) - 68) * 0.03125f;
  {
    std::ofstream output(source, std::ios::binary);
    output.write(reinterpret_cast<const char *>(values.data()),
                 values.size() * sizeof(float));
  }
  const auto whole = directory / "whole.bin";
  const auto blocked = directory / "blocked.bin";
  const auto whole_manifest = convertTensor(source, whole, 1u << 20);
  // First tensor uses 96, 96, 32 output channels; the transposed second
  // tensor uses two 32-channel blocks with K=224. Both exercise tiled seeks.
  const auto block_manifest = convertTensor(source, blocked, 64 * 96 * 4);
  check(contents(whole) == contents(blocked), "blocked WH differs from whole");
  check(whole_manifest == block_manifest, "blocked manifest offsets differ");
  check(block_manifest.size() == 2, "expected two FC entries");
  check(block_manifest[0]["file_offset"] == 28,
        "first FC offset includes copied prefix");
  const size_t second_offset = 28 + 64 * 224 / 2 + 224 * 8 + 9 * 4;
  check(block_manifest[1]["file_offset"] == second_offset,
        "second FC offset follows scales and column sums");
  check(std::filesystem::file_size(blocked) ==
          second_offset + 224 * 64 / 2 + 64 * 8 + 5 * 4,
        "blocked WH overwrote following tensors");
  // Truncation/extra input is still detected after a blocked tensor walk.
  {
    std::ofstream extra(source, std::ios::binary | std::ios::app);
    extra.put('x');
  }
  bool rejected = false;
  try {
    convertTensor(source, directory / "trailing.bin", 64 * 96 * 4);
  } catch (const std::runtime_error &) {
    rejected = true;
  }
  check(rejected, "trailing input accepted");
  std::cout << "BLOCKED WH BYTE-IDENTICAL, OFFSETS AND EOF OK\n";
}

int invokeQuantizer(const std::filesystem::path &model,
                    const std::filesystem::path &output) {
  std::vector<std::string> args{
    "quantize_stream", model.string(), "--output",   output.string(),
    "--output_bin",    "weights.bin",  "--fc_dtype", "QS4CX_WH"};
  std::vector<char *> argv;
  for (auto &arg : args)
    argv.push_back(arg.data());
  return run(static_cast<int>(argv.size()), argv.data());
}

void manifestPublication(const std::filesystem::path &directory) {
  const auto model = directory / "model";
  std::filesystem::create_directories(model);
  const auto cfg = json::parse(R"({
    "architectures":["Lfm2MoeForCausalLM"], "hidden_size":64,
    "vocab_size":32, "num_hidden_layers":1, "num_attention_heads":1,
    "num_key_value_heads":1, "head_dim":64, "intermediate_size":96,
    "moe_intermediate_size":32, "num_experts":1, "num_dense_layers":1,
    "conv_L_cache":3, "layer_types":["conv"], "tie_word_embeddings":true
  })");
  std::ofstream(model / "config.json") << cfg.dump();
  std::ofstream(model / "nntr_config.json")
    << R"({"model_file_name":"source.bin","model_tensor_type":"FP32-FP32",
           "fc_layer_dtype":"FP32","embedding_dtype":"FP32","lmhead_dtype":"FP32"})";
  const size_t elements = 32 * 64 + 64 + 64 * 192 + 3 * 64 + 64 * 64 + 64 +
                          64 * 96 * 2 + 96 * 64 + 64;
  {
    std::vector<float> source(elements);
    for (size_t i = 0; i < source.size(); ++i)
      source[i] = (int(i % 29) - 14) * 0.125f;
    std::ofstream out(model / "source.bin", std::ios::binary);
    out.write(reinterpret_cast<const char *>(source.data()),
              source.size() * sizeof(float));
  }
  const auto converted = directory / "converted";
  check(invokeQuantizer(model, converted) == 0, "tiny LFM conversion failed");
  const auto manifest = readJson(converted / "htp_fc_manifest.json");
  check(manifest["schema_version"] == 1 && manifest["dtype"] == "QS4CX_WH",
        "bad manifest schema");
  check(manifest["model_size"] ==
          std::filesystem::file_size(converted / "weights.bin"),
        "manifest final size mismatch");
  const auto &weights = manifest["weights"];
  check(weights.size() == 5, "manifest missing conv/dense weights");
  const uint64_t offsets[] = {8448, 16896, 19712, 23552, 27392};
  const char *names[] = {"layer0_conv_in_proj", "layer0_conv_out_proj",
                         "layer0_ffn_up", "layer0_ffn_gate", "layer0_ffn_down"};
  for (size_t i = 0; i < 5; ++i)
    check(weights[i]["file_offset"] == offsets[i] &&
            weights[i]["name"] == names[i],
          "manifest entry disagrees with independently counted file offsets");
  {
    std::ofstream extra(model / "source.bin", std::ios::binary | std::ios::app);
    extra.put('x');
  }
  const auto bad = directory / "failed";
  bool rejected = false;
  try {
    invokeQuantizer(model, bad);
  } catch (const std::runtime_error &) {
    rejected = true;
  }
  check(rejected, "bad source accepted by CLI");
  check(!std::filesystem::exists(bad / "htp_fc_manifest.json"),
        "failed conversion published a manifest");
  check(!std::filesystem::exists(bad / "htp_fc_manifest.json.tmp"),
        "failed conversion staged a manifest");
  std::cout << "OFFLINE FC MANIFEST PUBLICATION OK\n";
}
} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 2)
      throw std::runtime_error(
        "usage: quantize_stream_wh_check <empty-work-dir>");
    const std::filesystem::path directory(argv[1]);
    if (std::filesystem::exists(directory))
      throw std::runtime_error("work directory must not already exist");
    std::filesystem::create_directories(directory);
    blockedMatchesWhole(directory);
    manifestPublication(directory);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
  }
}
