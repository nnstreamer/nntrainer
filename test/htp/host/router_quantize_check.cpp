// SPDX-License-Identifier: Apache-2.0
/** Verify router-only conversion against direct quantization, including EOF
 *  rejection and independent router storage configuration.
 * @file router_quantize_check.cpp
 * @brief Regression checks for router quantize check.
 * @author NNTrainer contributors
 * @bug See the repository issue tracker for known issues.
 */
#define main nntr_quantize_stream_main
#include "../../../Applications/CausalLM/quantize_stream.cpp"
#undef main
#include <iterator>

static void require(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}
static std::vector<char> bytes(const std::filesystem::path &p) {
  std::ifstream f(p, std::ios::binary);
  return {std::istreambuf_iterator<char>(f), {}};
}
static int invoke(std::vector<std::string> args) {
  std::vector<char *> argv;
  for (auto &a : args)
    argv.push_back(a.data());
  return run(argv.size(), argv.data());
}
int main(int argc, char **argv) {
  if (argc != 2)
    return 2;
  const std::filesystem::path root(argv[1]), model_dir = root / "source";
  std::filesystem::create_directories(model_dir);
  const json cfg = json::parse(R"({"architectures":["Lfm2MoeForCausalLM"],
    "hidden_size":64,"vocab_size":32,"num_hidden_layers":2,
    "num_attention_heads":1,"num_key_value_heads":1,"head_dim":64,
    "intermediate_size":96,"moe_intermediate_size":32,"num_experts":32,
    "num_dense_layers":0,"conv_L_cache":3,"layer_types":["conv","attention"],
    "tie_word_embeddings":true})");
  std::ofstream(model_dir / "config.json") << cfg.dump();
  std::ofstream(model_dir / "nntr_config.json")
    << R"({"model_file_name":"weights.bin","model_tensor_type":"FP32-FP32",
         "fc_layer_dtype":"FP32","embedding_dtype":"FP32","lmhead_dtype":"FP32"})";
  const auto model = makeLfm2MoePlan(cfg);
  std::ifstream unused_in;
  std::ofstream unused_out;
  TensorWriter dry(unused_in, unused_out, ml::train::ISA::DEFAULT, 0, true);
  const QuantizationPlan fp32{DType::FP32, DType::FP32, DType::FP32,
                              DType::FP32, ml::train::ISA::DEFAULT};
  writeLfm2Moe(dry, model, fp32);
  {
    std::ofstream f(model_dir / "weights.bin", std::ios::binary);
    for (size_t i = 0; i < dry.expectedInputBytes() / 4; ++i) {
      const float v = (int((i * 37 + i / 19) % 101) - 50) * .03125f;
      f.write(reinterpret_cast<const char *>(&v), 4);
    }
  }
  for (const bool router4 : {false, true}) {
    const auto out = root / (router4 ? "direct" : "legacy");
    require(invoke({"quantizer", model_dir.string(), "--output", out.string(),
                    "--output_bin", "weights.bin", "--fc_dtype", "QS4CX_WH",
                    "--moe_dtype", "QS4CX_WH", "--router_dtype",
                    router4 ? "QS4CX" : "FP32"}) == 0,
            "quantization failed");
  }
  require(invoke({"quantizer", (root / "legacy").string(), "--router_only",
                  "--output", (root / "converted").string(), "--output_bin",
                  "weights.bin"}) == 0,
          "router conversion failed");
  require(bytes(root / "direct/weights.bin") ==
            bytes(root / "converted/weights.bin"),
          "router-only output differs from direct quantization");
  auto config = readJson(root / "converted/nntr_config.json");
  require(config["router_dtype"] == "QS4CX" &&
            config["moe_layer_dtype"] == "QS4CX_WH",
          "router config changed expert dtype");
  const auto saved = 2 * (64 * 32 * 4 - (64 * 32 / 2 + 32 * 4));
  require(std::filesystem::file_size(root / "legacy/weights.bin") -
              std::filesystem::file_size(root / "converted/weights.bin") ==
            saved,
          "incorrect router byte size");
  bool rejected = false;
  try {
    invoke({"quantizer", (root / "converted").string(), "--router_only",
            "--output", (root / "duplicate").string()});
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  require(rejected, "re-quantizing INT4 router was accepted");
  // Truncated quantized payload must not publish a runnable model/config.
  auto legacy = root / "legacy/weights.bin";
  std::filesystem::resize_file(legacy, std::filesystem::file_size(legacy) - 1);
  rejected = false;
  try {
    invoke({"quantizer", (root / "legacy").string(), "--router_only",
            "--output", (root / "truncated").string()});
  } catch (const std::runtime_error &) {
    rejected = true;
  }
  require(rejected &&
            !std::filesystem::exists(root / "truncated/nntr_config.json"),
          "truncated input published a model");
  std::cout << "PASS: direct/converted bytes, dtype isolation, size, "
               "re-quantization and EOF rejection\n";
}
