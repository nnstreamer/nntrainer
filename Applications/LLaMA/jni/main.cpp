// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Jihoon Lee <sb92.hong@samsung.com>
 *
 * @file   main.cpp
 * @date   7 August 2023
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <array>
#include <chrono>
#include <codecvt>
#include <ctime>
#include <iostream>
#include <locale>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <layer.h>
#include <model.h>
#include <optimizer.h>

#include <app_context.h>
#include <rms_norm.h>
#include <rotary_embedding.h>
#include <swiglu.h>
#include <transpose_layer.h>

<<<<<<< HEAD
#if defined(ENABLE_ENCODER2)
=======
//include for tokenizer///////////////////
>>>>>>> c94a43c3 ([LLaMA] Add korean language)
#include "json.hpp"
#include <codecvt>
#include <encoder.hpp>
#include <locale>
#include <sstream>
using json = nlohmann::json;
<<<<<<< HEAD
#endif
=======
//////////////////////////////////////////
>>>>>>> c94a43c3 ([LLaMA] Add korean language)

using LayerHandle = std::shared_ptr<ml::train::Layer>;
using ModelHandle = std::unique_ptr<ml::train::Model>;

// Hyper params for LLaMA
int const DIM = 2304;
int const NUM_LAYERS = 28;
int const NUM_HEADS = 18;

int const MULTIPLE_OF = 256;

float const NORM_EPS = 0.000001;
int const NUM_VOCAB = 96000;
int MAX_SEQ_LEN = 1024;
int NUM_TO_GENERATE = 1;

constexpr unsigned int INIT_SEQ_LEN = 30;
unsigned int batch_size = 1;
unsigned int epoch = 1;

/** cache loss values post training for test */
float training_loss = 0.0;
float validation_loss = 0.0;
bool swap = false;

bool optimize = false;

/**
 * @brief make "key=value" from key and value
 *
 * @tparam T type of a value
 * @param key key
 * @param value value
 * @return std::string with "key=value"
 */
template <typename T>
static std::string withKey(const std::string &key, const T &value) {
  std::stringstream ss;
  ss << key << "=" << value;
  return ss.str();
}

template <typename T>
static std::string withKey(const std::string &key,
                           std::initializer_list<T> value) {
  if (std::empty(value)) {
    throw std::invalid_argument("empty data cannot be converted");
  }

  std::stringstream ss;
  ss << key << "=";

  auto iter = value.begin();
  for (; iter != value.end() - 1; ++iter) {
    ss << *iter << ',';
  }
  ss << *iter;

  return ss.str();
}

template <typename T>
T unwrap(std::optional<T> &&value, const std::string &error_msg) {
  if (value.has_value()) {
    return value.value();
  } else {
    throw std::runtime_error(error_msg);
  }
}

std::vector<LayerHandle> createAttentionLayer(const int layer_id, int seq_len,
                                              int n_heads, int head_dim,
                                              std::string query_name,
                                              std::string key_name,
                                              std::string value_name) {
  using ml::train::createLayer;

  std::vector<LayerHandle> layers;

  if (optimize) {
    // linear transformation of q
    for (int i = 0; i < n_heads; i++) {
      layers.push_back(
        createLayer("fully_connected",
                    {withKey("name", "layer" + std::to_string(layer_id) +
                                       "_wq_" + std::to_string(i)),
                     withKey("unit", head_dim), withKey("disable_bias", "true"),
                     withKey("input_layers", query_name)}));
    }

    // linear transformation of k
    for (int i = 0; i < n_heads; i++) {
      layers.push_back(
        createLayer("fully_connected",
                    {withKey("name", "layer" + std::to_string(layer_id) +
                                       "_wk_" + std::to_string(i)),
                     withKey("unit", head_dim), withKey("disable_bias", "true"),
                     withKey("input_layers", key_name)}));
    }

    // linear transformation of v
    for (int i = 0; i < n_heads; i++) {
      layers.push_back(
        createLayer("fully_connected",
                    {withKey("name", "layer" + std::to_string(layer_id) +
                                       "_wv_" + std::to_string(i)),
                     withKey("unit", head_dim), withKey("disable_bias", "true"),
                     withKey("input_layers", value_name)}));
    }

    std::string concat_input = "";
    // apply rotary embedding and dot_product attention
    for (int i = 0; i < n_heads; i++) {
      // reshape q, k, v (apply num_heads)
      layers.push_back(createLayer(
        "reshape", {withKey("name", "layer" + std::to_string(layer_id) +
                                      "_q_reshape_" + std::to_string(i)),
                    withKey("target_shape", "1:" + std::to_string(seq_len) +
                                              ":" + std::to_string(head_dim)),
                    withKey("input_layers", "layer" + std::to_string(layer_id) +
                                              "_wq_" + std::to_string(i))}));

      layers.push_back(createLayer(
        "reshape", {withKey("name", "layer" + std::to_string(layer_id) +
                                      "_k_reshape_" + std::to_string(i)),
                    withKey("target_shape", "1:" + std::to_string(seq_len) +
                                              ":" + std::to_string(head_dim)),
                    withKey("input_layers", "layer" + std::to_string(layer_id) +
                                              "_wk_" + std::to_string(i))}));

      layers.push_back(createLayer(
        "reshape", {withKey("name", "layer" + std::to_string(layer_id) +
                                      "_v_reshape_" + std::to_string(i)),
                    withKey("target_shape", "1:" + std::to_string(seq_len) +
                                              ":" + std::to_string(head_dim)),
                    withKey("input_layers", "layer" + std::to_string(layer_id) +
                                              "_wv_" + std::to_string(i))}));

      // apply rotary embedding to q, k
      layers.push_back(createLayer(
        "rotary_embedding",
        {withKey("name", "layer" + std::to_string(layer_id) + "_q_rotary_" +
                           std::to_string(i)),
         withKey("input_layers", "layer" + std::to_string(layer_id) +
                                   "_q_reshape_" + std::to_string(i))}));

      layers.push_back(createLayer(
        "rotary_embedding",
        {withKey("name", "layer" + std::to_string(layer_id) + "_k_rotary_" +
                           std::to_string(i)),
         withKey("input_layers", "layer" + std::to_string(layer_id) +
                                   "_k_reshape_" + std::to_string(i))}));

      // apply scaled-dot product attention
      layers.push_back(ml::train::layer::Attention(
        {"name=layer" + std::to_string(layer_id) + "_attention_" +
           std::to_string(i),
         "input_layers=layer" + std::to_string(layer_id) + "_q_rotary_" +
           std::to_string(i) + ",layer" + std::to_string(layer_id) +
           "_v_reshape_" + std::to_string(i) + ",layer" +
           std::to_string(layer_id) + "_k_rotary_" + std::to_string(i),
         "scaled_dot_product=true", "causal_mask=true"}));

      layers.push_back(createLayer(
        "reshape",
        {withKey("name", "layer" + std::to_string(layer_id) +
                           "_attention_output_" + std::to_string(i)),
         withKey("target_shape", "1:" + std::to_string(seq_len) + ":" +
                                   std::to_string(head_dim)),
         withKey("input_layers", "layer" + std::to_string(layer_id) +
                                   "_attention_" + std::to_string(i))}));

      concat_input += "layer" + std::to_string(layer_id) +
                      "_attention_output_" + std::to_string(i);

      if (i != n_heads - 1) {
        concat_input += ",";
      }
    }

    // concat attention output
    layers.push_back(createLayer(
      "concat", {withKey("name", "layer" + std::to_string(layer_id) +
                                   "_attention_concat"),
                 withKey("axis", 3), withKey("input_layers", concat_input)}));

    // reshape for flatten
    layers.push_back(createLayer(
      "reshape", {withKey("name", "layer" + std::to_string(layer_id) +
                                    "_attention_flatten"),
                  withKey("target_shape", "1:" + std::to_string(seq_len) + ":" +
                                            std::to_string(n_heads * head_dim)),
                  withKey("input_layers", "layer" + std::to_string(layer_id) +
                                            "_attention_concat")}));

    // linear transformation of attention output
    layers.push_back(createLayer(
      "fully_connected",
      {withKey("name", "layer" + std::to_string(layer_id) + "_attention_out"),
       withKey("unit", head_dim * n_heads), withKey("disable_bias", "true"),
       withKey("input_layers",
               "layer" + std::to_string(layer_id) + "_attention_flatten")}));
  } else {
    layers.push_back(createLayer(
      "multi_head_attention",
      {withKey("name", "layer" + std::to_string(layer_id) + "_attention_out"),
       withKey("num_heads", std::to_string(NUM_HEADS)),
       withKey("max_timestep", std::to_string(MAX_SEQ_LEN)),
       withKey("disable_bias", "true"),
       withKey("input_layers", {query_name, key_name, value_name})}));
  }

  return layers;
}

std::vector<LayerHandle> createFeedForwardLayer(const int layer_id, int dim,
                                                int hidden_dim,
                                                std::string input_name,
                                                int multiplier = 1) {
  using ml::train::createLayer;
  std::vector<LayerHandle> layers;

  hidden_dim = 2 * multiplier * hidden_dim / 3;
  hidden_dim = MULTIPLE_OF * ((hidden_dim + MULTIPLE_OF - 1) / MULTIPLE_OF);

  layers.push_back(
    createLayer("fully_connected",
                {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_1"),
                 withKey("unit", hidden_dim), withKey("disable_bias", "true"),
                 withKey("input_layers", input_name)}));
  layers.push_back(
    createLayer("fully_connected",
                {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_2"),
                 withKey("unit", hidden_dim), withKey("disable_bias", "true"),
                 withKey("input_layers", input_name)}));

  layers.push_back(createLayer(
    "swiglu",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_swiglu"),
     withKey("input_layers", "layer" + std::to_string(layer_id) + "_ffn_1," +
                               "layer" + std::to_string(layer_id) +
                               "_ffn_2")}));

  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_output"),
     withKey("unit", dim), withKey("disable_bias", "true"),
     withKey("input_layers",
             "layer" + std::to_string(layer_id) + "_ffn_swiglu")}));

  return layers;
}

std::vector<LayerHandle> createTransformerDecoder(const int layer_id,
                                                  std::string input_name) {
  using ml::train::createLayer;
  std::vector<LayerHandle> layers;

  layers.push_back(createLayer(
    "rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "_attention_norm"),
     withKey("input_layers", input_name),
     withKey("epsilon", std::to_string(NORM_EPS))}));

  auto att_layer = createAttentionLayer(
    layer_id, INIT_SEQ_LEN, NUM_HEADS, DIM / NUM_HEADS,
    "layer" + std::to_string(layer_id) + "_attention_norm",
    "layer" + std::to_string(layer_id) + "_attention_norm",
    "layer" + std::to_string(layer_id) + "_attention_norm");
  layers.insert(layers.end(), att_layer.begin(), att_layer.end());

  layers.push_back(createLayer(
    "addition",
    {withKey("name", "layer" + std::to_string(layer_id) + "_decoder_add"),
     withKey("input_layers", input_name + ",layer" + std::to_string(layer_id) +
                               "_attention_out")}));

  layers.push_back(createLayer(
    "rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_norm"),
     withKey("input_layers",
             "layer" + std::to_string(layer_id) + "_decoder_add"),
     withKey("epsilon", std::to_string(NORM_EPS))}));

  auto ffn_layer = createFeedForwardLayer(
    layer_id, DIM, 4 * DIM, "layer" + std::to_string(layer_id) + "_ffn_norm");
  layers.insert(layers.end(), ffn_layer.begin(), ffn_layer.end());

  layers.push_back(createLayer(
    "addition",
    {withKey("name", "layer" + std::to_string(layer_id) + "_decoder_output"),
     withKey("input_layers", "layer" + std::to_string(layer_id) +
                               "_decoder_add,layer" + std::to_string(layer_id) +
                               "_ffn_output")}));

  return layers;
}

ModelHandle createLLaMA() {
  using ml::train::createLayer;

  ModelHandle model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);

  std::vector<LayerHandle> layers;

  if (optimize) {
    layers.push_back(createLayer(
      "input",
      {withKey("name", "input0"),
       withKey("input_shape", "1:1:" + std::to_string(INIT_SEQ_LEN))}));
  } else {
    layers.push_back(createLayer(
      "input", {withKey("name", "input0"), withKey("input_shape", "1:1:1")}));
  }

  layers.push_back(ml::train::layer::Embedding(
    {"name=embedding0", "in_dim=" + std::to_string(NUM_VOCAB),
     "out_dim=" + std::to_string(DIM)}));

  for (int i = 0; i < NUM_LAYERS; i++) {
    std::vector<LayerHandle> transformer;
    if (i == 0)
      transformer = createTransformerDecoder(i, "embedding0");
    else
      transformer = createTransformerDecoder(
        i, "layer" + std::to_string(i - 1) + "_decoder_output");
    layers.insert(layers.end(), transformer.begin(), transformer.end());
  }

  int last_layer = NUM_LAYERS - 1;

  layers.push_back(createLayer(
    "rms_norm", {withKey("name", "output_norm"),
                 withKey("epsilon", std::to_string(NORM_EPS)),
                 withKey("input_layers", "layer" + std::to_string(last_layer) +
                                           "_decoder_output")}));

  layers.push_back(createLayer(
    "fully_connected",
    {withKey("name", "output_of_llama"), withKey("unit", NUM_VOCAB),
     withKey("disable_bias", "true"), withKey("input_layers", "output_norm")}));

  for (auto &layer : layers) {
    model->addLayer(layer);
  }

  return model;
}

void createAndRun(unsigned int epochs, unsigned int batch_size,
                  std::wstring text) {

  // setup model
  ModelHandle model = createLLaMA();
  model->setProperty({withKey("batch_size", batch_size),
                      withKey("epochs", epochs),
                      // #ifdef ENABLE_FP16
                      // withKey("model_tensor_type", "FP16-FP16"),
                      // #endif
                      withKey("save_path", "test_model.bin")});

  auto optimizer = ml::train::createOptimizer("sgd", {"learning_rate=0.001"});
  model->setOptimizer(std::move(optimizer));

  int status = model->compile();
  if (status) {
    throw std::invalid_argument("model compilation failed!");
  }

  status = model->initialize();
  if (status) {
    throw std::invalid_argument("model initialization failed!");
  }

  // model->summarize(std::cout, ML_TRAIN_SUMMARY_MODEL);

  std::string weight_path =
    optimize ? "./llama_v2_att.bin" : "/home/donghak/Desktop/llama_v2.bin";
  model->load(weight_path);

  std::vector<float *> input;
  std::vector<float *> label;

  int data_size = batch_size * INIT_SEQ_LEN;
  
  float *input_sample = (float *)malloc(sizeof(float) * data_size);

#if defined(ENABLE_ENCODER2)
  std::string vocab_file_name = "../Applications/LLaMA/jni/vocab.json";
  std::string merge_file_name = "../Applications/LLaMA/jni/merges.txt";

  auto tokenizer = unwrap(GPT2Encoder::load(vocab_file_name, merge_file_name),
                          "Error initialising GPT2 tokenizer\n");

  auto init_input = tokenizer.encode(text);
<<<<<<< HEAD
  INIT_SEQ_LEN = init_input.size();
=======

  INIT_SEQ_LEN = init_input.size();

>>>>>>> c94a43c3 ([LLaMA] Add korean language)
  ((uint *)(input_sample))[0] = init_input[0];
  input.push_back(input_sample);

#else
  float init_data[INIT_SEQ_LEN] = {
    0,  1,  2,  3,  4,  5,   6,   7,   8,   9,   10,  20,  30,  40,
    50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900};
  ((uint *)(input_sample))[0] = init_data[0];
  input.push_back(input_sample);
#endif

  for (unsigned int i = 1; i < INIT_SEQ_LEN + NUM_TO_GENERATE; ++i) {
    auto output =
      model->incremental_inference(1, input, label, INIT_SEQ_LEN, i - 1);

    std::vector<int64_t> tokens;
    nntrainer::Tensor output_tensor({batch_size, 1, 1, NUM_VOCAB}, output[0]);

    tokens.push_back(static_cast<int64_t>(output_tensor.argmax()[0]));
<<<<<<< HEAD
#if defined(ENABLE_ENCODER2)
    auto decoded_str = tokenizer.decode(tokens);
    std::cerr << decoded_str << std::flush;
#endif
=======

    auto decoded_str = tokenizer.decode(tokens);
    std::cerr << decoded_str << std::flush;
>>>>>>> c94a43c3 ([LLaMA] Add korean language)

    if (i < INIT_SEQ_LEN) {
#if defined(ENABLE_ENCODER2)
      ((uint *)(input_sample))[0] = init_input[i];
#else
      ((uint *)(input_sample))[0] = init_data[i];
#endif
    } else {
      ((uint *)(input_sample))[0] = output_tensor.argmax()[0];
    }
  }
}

#if defined(ENABLE_ENCODER2)
std::wstring decodeUnicodeEscape(const std::wstring &input) {
  std::wstringstream result;

  for (size_t i = 0; i < input.length(); ++i) {
    if (i + 5 < input.length() && input[i] == L'\\' && input[i + 1] == L'u') {
      std::wstring unicodeSeq;
      for (int j = 0; j < 4; ++j)
        unicodeSeq += input[i + 2 + j];

      result << static_cast<wchar_t>(std::stoi(unicodeSeq, nullptr, 16));
      i += 5;
    } else if (input[i] == L'\\' && input[i + 1] == L'n') {
      result << static_cast<wchar_t>('\n');
      i++;
    } else if (input[i] == L' ')
      result << static_cast<wchar_t>(' ');
    else
      result << input[i];
  }

  return result.str();
}
#endif

int main(int argc, char *argv[]) {
  // Setting locale
  std::locale::global(std::locale("ko_KR.UTF-8"));

<<<<<<< HEAD
#if defined(ENABLE_ENCODER2)

  // Getting arguments From terminal
  std::wstring input;
  std::getline(std::wcin, input);
  std::wstring test = decodeUnicodeEscape(input);
  std::wstring_convert<std::codecvt_utf16<wchar_t>> converter;
  std::string text = converter.to_bytes(test);
=======
  // Getting arguments From terminal
  // std::wstring input;
  // std::getline(std::wcin, input);
  // std::wstring test = decodeUnicodeEscape(input);
  // std::wstring_convert<std::codecvt_utf16<wchar_t>> converter;
  // std::string text = converter.to_bytes(test);
  /////////////////////////////////////////////////////////////////////

  std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
  std::ifstream ifs("/home/donghak/Desktop/workspace/N2S2/nntrainer/"
                    "Applications/LLaMA/jni/test3.json");
  json data = json::parse(ifs);

  // Load From json file [index][key]
  auto parsed_text = data[1]["source"].get<std::string>();
  
  std::wstring convert_text = converter.from_bytes(parsed_text);
  std::wstring text = decodeUnicodeEscape(convert_text);
>>>>>>> c94a43c3 ([LLaMA] Add korean language)

  std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
#else
  std::wstring text = L"This is sample input for LLaMA.";
#endif

  auto &app_context = nntrainer::AppContext::Global();
  try {
    app_context.registerFactory(nntrainer::createLayer<custom::SwiGLULayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
    return 1;
  }

  try {
    app_context.registerFactory(nntrainer::createLayer<custom::RMSNormLayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
    return 1;
  }
  createAndRun(epoch, batch_size, text);

  int status = EXIT_SUCCESS;
  return status;
}
