// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Hyeonseok Lee <hs89.lee@samsung.com>
 *
 * @file   main.cpp
 * @date   19 May 2023
 * @brief  task runner for the pico gpt
 * @see    https://github.com/nnstreamer/nntrainer
 *         https://github.com/jaymody/picoGPT
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <app_context.h>
#include <fstream>
#include <model.h>
#include <string.h>
#include <tensor.h>

#include "encoder.hpp"
#include <iostream>

const unsigned int BATCH_SIZE = 1;
const unsigned int NUM_LAYERS = 12;
const unsigned int NUM_HEADS = 12;
const unsigned int MODEL_DIM = 768;
// @Todo: Need to check
const unsigned int FC_UNIT = MODEL_DIM * 4;

const unsigned int NUM_VOCAB = 50257;
const unsigned int NUM_CTX = 1024;
const unsigned int NUM_TOKENS_TO_GENERATE = 40;

unsigned int init_input_seq_len;
// Todo: fix this
const unsigned int MAX_TOKEN_LEN = 10 + NUM_TOKENS_TO_GENERATE;

bool swap = false;
bool optimize = false;
// bool optimize = true;
bool optimize_attention = false;

template <typename T>
T unwrap(std::optional<T> &&value, const std::string &error_msg) {
  if (value.has_value()) {
    return value.value();
  } else {
    throw std::runtime_error(error_msg);
  }
}

std::shared_ptr<ml::train::Model> genModel() {
  std::shared_ptr<ml::train::Model> model;
  model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);
  model->setProperty({"batch_size=" + std::to_string(BATCH_SIZE),
                      "memory_optimization=false",
                      "model_tensor_type=FP16-FP16",
                      swap ? "memory_swap=true" : "memory_swap=false"});

  std::shared_ptr<ml::train::Layer> wte_input = ml::train::layer::Input(
    {"name=wte_input", "input_shape=1:1:" + std::to_string(MAX_TOKEN_LEN)});
  model->addLayer(wte_input);

  std::shared_ptr<ml::train::Layer> wte = ml::train::layer::Embedding(
    {"name=wte", "in_dim=" + std::to_string(NUM_VOCAB),
     "out_dim=" + std::to_string(MODEL_DIM)});
  model->addLayer(wte);

  std::shared_ptr<ml::train::Layer> wpe_input = ml::train::layer::Input(
    {"name=wpe_input", "input_shape=1:1:" + std::to_string(MAX_TOKEN_LEN)});
  model->addLayer(wpe_input);

  std::shared_ptr<ml::train::Layer> wpe = ml::train::layer::Embedding(
    {"name=wpe", "in_dim=" + std::to_string(NUM_CTX),
     "out_dim=" + std::to_string(MODEL_DIM)});
  model->addLayer(wpe);

  std::shared_ptr<ml::train::Layer> add =
    ml::train::layer::Addition({"name=add", "input_layers=wte, wpe"});
  model->addLayer(add);

  for (unsigned int i = 0; i < NUM_LAYERS; ++i) {
    std::shared_ptr<ml::train::Layer> ln_multiout1 = ml::train::layer::MultiOut(
      {"name=layer" + std::to_string(i) + "/ln_multiout1"});
    model->addLayer(ln_multiout1);

    std::shared_ptr<ml::train::Layer> ln1 =
      ml::train::layer::LayerNormalization(
        {"name=layer" + std::to_string(i) + "/ln1", "axis=3", "epsilon=1e-5"});
    model->addLayer(ln1);

    std::shared_ptr<ml::train::Layer> multiout1 = ml::train::layer::MultiOut(
      {"name=layer" + std::to_string(i) + "/multi_out1"});
    model->addLayer(multiout1);

    if (optimize) {
      std::string concat_input = "";

      for (unsigned int j = 0; j < NUM_HEADS; ++j) {
        std::shared_ptr<ml::train::Layer> multi_head_attention_v_fc =
          ml::train::layer::FullyConnected(
            {"name=layer" + std::to_string(i) + "/multi_head_attention/v_fc" +
               std::to_string(NUM_HEADS - 1 - j),
             "input_layers=layer" + std::to_string(i) + "/multi_out1(" +
               std::to_string(2 * NUM_HEADS + j) + ")",
             "unit=" + std::to_string(MODEL_DIM / NUM_HEADS)});
        model->addLayer(multi_head_attention_v_fc);
      }

      for (unsigned int j = 0; j < NUM_HEADS; ++j) {
        std::shared_ptr<ml::train::Layer> multi_head_attention_k_fc =
          ml::train::layer::FullyConnected(
            {"name=layer" + std::to_string(i) + "/multi_head_attention/k_fc" +
               std::to_string(NUM_HEADS - 1 - j),
             "input_layers=layer" + std::to_string(i) + "/multi_out1(" +
               std::to_string(NUM_HEADS + j) + ")",
             "unit=" + std::to_string(MODEL_DIM / NUM_HEADS)});
        model->addLayer(multi_head_attention_k_fc);
      }

      for (unsigned int j = 0; j < NUM_HEADS; ++j) {
        std::shared_ptr<ml::train::Layer> multi_head_attention_q_fc =
          ml::train::layer::FullyConnected(
            {"name=layer" + std::to_string(i) + "/multi_head_attention/q_fc" +
               std::to_string(NUM_HEADS - 1 - j),
             "input_layers=layer" + std::to_string(i) + "/multi_out1(" +
               std::to_string(j) + ")",
             "unit=" + std::to_string(MODEL_DIM / NUM_HEADS)});
        model->addLayer(multi_head_attention_q_fc);
      }

      for (unsigned int j = 0; j < NUM_HEADS; ++j) {
        if (optimize_attention) {
          //   std::shared_ptr<ml::train::Layer> multi_head_attention_bwdp1 =
          //     ml::train::layer::BatchwiseDotproduct(
          //       {"name=layer" + std::to_string(i) +
          //          "/multi_head_attention/bwdp1" +
          //          std::to_string(NUM_HEADS - 1 - j),
          //        "input_layers=layer" + std::to_string(i) +
          //          "/multi_head_attention/q_fc" +
          //          std::to_string(NUM_HEADS - 1 - j) + ",layer" +
          //          std::to_string(i) + "/multi_head_attention/k_fc" +
          //          std::to_string(NUM_HEADS - 1 - j),
          //        "transpose_key=true", "scaled_dot_product=true",
          //        "activation=softmax"});
          //   model->addLayer(multi_head_attention_bwdp1);

          //   std::shared_ptr<ml::train::Layer> multi_head_attention_bwdp2 =
          //     ml::train::layer::BatchwiseDotproduct(
          //       {"name=layer" + std::to_string(i) +
          //          "/multi_head_attention/bwdp2" +
          //          std::to_string(NUM_HEADS - 1 - j),
          //        "input_layers=layer" + std::to_string(i) +
          //          "/multi_head_attention/bwdp1" +
          //          std::to_string(NUM_HEADS - 1 - j) + ",layer" +
          //          std::to_string(i) + "/multi_head_attention/v_fc" +
          //          std::to_string(NUM_HEADS - 1 - j)});
          //   model->addLayer(multi_head_attention_bwdp2);

          //   std::shared_ptr<ml::train::Layer>
          //     multi_head_attention_attention = ml::train::layer::Identity(
          //       {"name=layer" + std::to_string(i) +
          //          "/multi_head_attention/attention" +
          //          std::to_string(NUM_HEADS - 1 - j),
          //        "input_layers=layer" + std::to_string(i) +
          //          "/multi_head_attention/bwdp2" +
          //          std::to_string(NUM_HEADS - 1 - j)});
          //   model->addLayer(multi_head_attention_attention);
        } else {
          std::shared_ptr<ml::train::Layer> multi_head_attention_attention =
            ml::train::layer::Attention(
              {"name=layer" + std::to_string(i) +
                 "/multi_head_attention/attention" +
                 std::to_string(NUM_HEADS - 1 - j),
               "input_layers=layer" + std::to_string(i) +
                 "/multi_head_attention/q_fc" +
                 std::to_string(NUM_HEADS - 1 - j) + ",layer" +
                 std::to_string(i) + "/multi_head_attention/v_fc" +
                 std::to_string(NUM_HEADS - 1 - j) + ",layer" +
                 std::to_string(i) + "/multi_head_attention/k_fc" +
                 std::to_string(NUM_HEADS - 1 - j),
               "scaled_dot_product=true", "causal_mask=true"});
          model->addLayer(multi_head_attention_attention);
        }

        concat_input += "layer" + std::to_string(i) +
                        "/multi_head_attention/attention" + std::to_string(j);
        if (j != NUM_HEADS - 1) {
          concat_input += ",";
        }
      }

      std::shared_ptr<ml::train::Layer> multi_head_attention_concat =
        ml::train::layer::Concat(
          {"name=layer" + std::to_string(i) + "/multi_head_attention/concat",
           "input_layers=" + concat_input, "axis=3"});
      model->addLayer(multi_head_attention_concat);

      std::shared_ptr<ml::train::Layer> multi_head_attention_fc =
        ml::train::layer::FullyConnected(
          {"name=layer" + std::to_string(i) + "/multi_head_attention/fc",
           "input_layers=layer" + std::to_string(i) +
             "/multi_head_attention/concat",
           "unit=" + std::to_string(MODEL_DIM)});
      model->addLayer(multi_head_attention_fc);

      std::shared_ptr<ml::train::Layer> multi_head_attention =
        ml::train::layer::Identity(
          {"name=layer" + std::to_string(i) + "/multi_head_attention",
           "input_layers=layer" + std::to_string(i) +
             "/multi_head_attention/fc"});
      model->addLayer(multi_head_attention);
    } else {
      std::shared_ptr<ml::train::Layer> masked_multi_head_attention =
        ml::train::layer::MultiHeadAttention(
          {"name=layer" + std::to_string(i) + "/multi_head_attention",
           "input_layers=layer" + std::to_string(i) + "/multi_out1(0), layer" +
             std::to_string(i) + "/multi_out1(1), layer" + std::to_string(i) +
             "/multi_out1(2)",
           "num_heads=" + std::to_string(NUM_HEADS)});
      model->addLayer(masked_multi_head_attention);
    }

    std::shared_ptr<ml::train::Layer> add1 = ml::train::layer::Addition(
      {"name=layer" + std::to_string(i) + "/add1",
       "input_layers=layer" + std::to_string(i) + "/ln_multiout1(1), layer" +
         std::to_string(i) + "/multi_head_attention"});
    model->addLayer(add1);

    std::shared_ptr<ml::train::Layer> ln_multiout2 = ml::train::layer::MultiOut(
      {"name=layer" + std::to_string(i) + "/ln_multiout2"});
    model->addLayer(ln_multiout2);

    std::shared_ptr<ml::train::Layer> ln2 =
      ml::train::layer::LayerNormalization(
        {"name=layer" + std::to_string(i) + "/ln2", "axis=3", "epsilon=1e-5"});
    model->addLayer(ln2);

    std::shared_ptr<ml::train::Layer> multiout3 = ml::train::layer::MultiOut(
      {"name=layer" + std::to_string(i) + "/multi_out3"});
    model->addLayer(multiout3);

    std::shared_ptr<ml::train::Layer> fc1 = ml::train::layer::FullyConnected(
      {"name=layer" + std::to_string(i) + "/fc1",
       "input_layers=layer" + std::to_string(i) + "/multi_out3(0)",
       "unit=" + std::to_string(FC_UNIT), "activation=gelu"});
    model->addLayer(fc1);

    std::shared_ptr<ml::train::Layer> fc2 = ml::train::layer::FullyConnected(
      {"name=layer" + std::to_string(i) + "/fc2",
       "unit=" + std::to_string(MODEL_DIM)});
    model->addLayer(fc2);

    std::shared_ptr<ml::train::Layer> add2 = ml::train::layer::Addition(
      {"name=layer" + std::to_string(i) + "/add2",
       "input_layers=layer" + std::to_string(i) + "/ln_multiout2(1), layer" +
         std::to_string(i) + "/fc2"});
    model->addLayer(add2);
  }

  std::shared_ptr<ml::train::Layer> layer_normalization =
    ml::train::layer::LayerNormalization(
      {"name=layer_normalization", "axis=3", "epsilon=1e-5"});
  model->addLayer(layer_normalization);

  model->setOptimizer(
    ml::train::createOptimizer("sgd", {"learning_rate = 0.1"}));
  model->setProperty({"input_layers=wte_input, wpe_input"});

  return model;
}

int main(int argc, char *argv[]) {

  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <text>\n";
    return 1;
  }

  const std::vector<std::string> args(argv + 1, argv + argc);
  std::string text = args[0];

  auto model = genModel();
  model->summarize(std::cout, ML_TRAIN_SUMMARY_MODEL);
  try {
    model->compile();
  } catch (const std::exception &e) {
    std::cerr << "Error during compile: " << e.what() << "\n";
    return 1;
  }

  try {
    model->initialize();
  } catch (const std::exception &e) {
    std::cerr << "Error during initialize: " << e.what() << "\n";
    return 1;
  }

  std::string weight_file_name = optimize
                                   ? "./res/app/PicoGPT/pico_gpt_124.bin"
                                   : "./res/app/PicoGPT/pico_gpt_mha_fp16.bin";
  // : "./res/app/PicoGPT/pico_gpt_124_mha.bin";

  model->load(weight_file_name, ml::train::ModelFormat::MODEL_FORMAT_BIN);

  // model->save("pico_gpt_fp16.bin");

  // exit(0);

  float *wte_input = new float[MAX_TOKEN_LEN];
  float *wpe_input = new float[MAX_TOKEN_LEN];

  std::string vocab_file_name = "../Applications/PicoGPT/jni/vocab.json";
  std::string merge_file_name = "../Applications/PicoGPT/jni/merges.txt";

  auto tokenizer = unwrap(GPT2Encoder::load(vocab_file_name, merge_file_name),
                          "Error initialising GPT2 tokenizer\n");

  auto init_input = tokenizer.encode(text);
  init_input_seq_len = init_input.size();

  for (unsigned int i = 0; i < init_input_seq_len; ++i) {
    ((uint *)(wte_input))[i] = init_input[i];
  }

  for (unsigned int i = 0; i < init_input_seq_len; ++i) {
    ((uint *)(wpe_input))[i] = i;
  }

  std::vector<float *> output_bufs;

  std::shared_ptr<ml::train::Layer> wte_embedding_layer;
  model->getLayer("wte", &wte_embedding_layer);
  const std::vector<float *> wte_weights_buf =
    wte_embedding_layer->getWeights();
  nntrainer::Tensor wte_weight =
    nntrainer::Tensor({NUM_VOCAB, MODEL_DIM}, wte_weights_buf[0]);

  for (unsigned int i = init_input_seq_len;
       i < init_input_seq_len + NUM_TOKENS_TO_GENERATE; ++i) {
    output_bufs = model->incremental_inference(
      BATCH_SIZE, {wte_input, wpe_input}, {}, init_input_seq_len, i - 1);

    nntrainer::Tensor output({BATCH_SIZE, 1, i, MODEL_DIM}, output_bufs[0]);
    nntrainer::Tensor incremented_output = output.getSharedDataTensor(
      {BATCH_SIZE, 1, 1, MODEL_DIM}, BATCH_SIZE * (i - 1) * MODEL_DIM);

    nntrainer::Tensor next = incremented_output.dot(wte_weight, false, true);

    std::vector<unsigned int> ids = next.argmax();

    ((uint *)(wte_input))[i] = ids[0];
    ((uint *)(wpe_input))[i] = i;

    std::vector<int64_t> token_ids;
    for (auto element : ids) {
      token_ids.push_back(static_cast<int64_t>(element));
    }
    auto decoded_str = tokenizer.decode(token_ids);
    std::cerr << decoded_str << " " << std::flush;
  }
  for (auto v : wte_weights_buf) {
    delete v;
  }

  std::cout << std::endl;
  return 0;
}
