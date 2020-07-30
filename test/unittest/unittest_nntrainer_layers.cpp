/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * SPDX-License-Identifier: Apache-2.0-only
 *
 * @file        unittest_nntrainer_layers.cpp
 * @date        03 June 2020
 * @brief       Unit test utility for layers.
 * @see         https://github.com/nnstreamer/nntrainer
 * @author      Jijoong Moon <jijoong.moon@samsung.com>
 * @bug         No known bugs
 */
#include <activation_layer.h>
#include <bn_layer.h>
#include <conv2d_layer.h>
#include <fc_layer.h>
#include <flatten_layer.h>
#include <fstream>
#include <input_layer.h>
#include <layer.h>
#include <loss_layer.h>
#include <nntrainer_error.h>
#include <nntrainer_test_util.h>
#include <optimizer.h>
#include <pooling2d_layer.h>
#include <util_func.h>

using nntrainer::sharedTensor;

template <typename LayerType>
class nntrainer_abstractLayer : public ::testing::Test {
protected:
  virtual void SetUp() {
    status = ML_ERROR_NONE;
    prepareLayer();
    reinitialize(last_layer);
  }

  virtual int reinitialize(bool last_layer = false) {
    int status = layer.initialize(last_layer);
    EXPECT_EQ(status, ML_ERROR_NONE);
    in = nntrainer::Tensor(layer.getInputDimension());
    out = nntrainer::Tensor(layer.getOutputDimension());
    return status;
  }

  virtual int reinitialize(const std::string str, bool last_layer = false) {
    resetLayer();
    int status = setProperty(str);
    EXPECT_EQ(status, ML_ERROR_NONE);
    status = reinitialize(last_layer);
    EXPECT_EQ(status, ML_ERROR_NONE);
    return status;
  }

  // anchor point to prepare layer
  virtual void prepareLayer(){};

  virtual void resetLayer() { layer = LayerType(); }

  virtual void setInputDim(const char *dimension) {
    nntrainer::TensorDim dim;
    int status = dim.setTensorDim(dimension);
    ASSERT_EQ(status, ML_ERROR_NONE);
    layer.setInputDimension(dim);
  }

  void matchOutput(const nntrainer::Tensor &result,
                   const nntrainer::Tensor &golden) {
    const float *out_ptr, *golden_ptr;

    out_ptr = result.getData();
    golden_ptr = golden.getData();

    for (size_t i = 0; i < result.length(); ++i) {
      EXPECT_NEAR(out_ptr[i], golden_ptr[i], tolerance);
    }
  }

  void matchOutput(const nntrainer::Tensor &result, const char *expected) {
    nntrainer::Tensor golden(result.getDim());
    loadFile(expected, golden);
    matchOutput(result, golden);
  }

  void matchOutput(const std::vector<float> result, const char *expected) {
    nntrainer::Tensor golden;
    loadFile(expected, golden);
    const float *golden_ptr = golden.getData();
    for (size_t i = 0; i < golden.length(); ++i) {
      EXPECT_NEAR(result[i], golden_ptr[i], tolerance);
    }
  }

  // setting property separated by "|"
  int setProperty(const std::string &str) {
    std::vector<std::string> input_str;
    std::regex words_regex("[^|]+");
    auto words_begin =
      std::sregex_iterator(str.begin(), str.end(), words_regex);
    auto words_end = std::sregex_iterator();
    for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
      input_str.push_back((*i).str());
    }
    int status = layer.setProperty(input_str);
    EXPECT_EQ(status, ML_ERROR_NONE);

    return status;
  }

  // setting optimizer property separated by "|"
  int setOptimizer(nntrainer::OptType type, const std::string &str = "") {
    std::vector<std::string> input_str;
    std::regex words_regex("[^|]+");
    auto words_begin =
      std::sregex_iterator(str.begin(), str.end(), words_regex);
    auto words_end = std::sregex_iterator();
    for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
      input_str.push_back((*i).str());
    }

    nntrainer::Optimizer op;
    int status = op.setType(type);
    EXPECT_EQ(status, ML_ERROR_NONE);
    status = op.setProperty(input_str);
    EXPECT_EQ(status, ML_ERROR_NONE);
    status = layer.setOptimizer(op);
    EXPECT_EQ(status, ML_ERROR_NONE);

    return status;
  }

  template <typename T> void saveFile(const char *filename, T &t) {
    std::ofstream file(filename, std::ios::out | std::ios::binary);
    if (!file.good()) {
      throw std::runtime_error("could not save file");
    }
    t.save(file);
    file.close();
  }

  template <typename T> void loadFile(const char *filename, T &t) {
    std::ifstream file(filename);
    if (!file.good()) {
      throw std::runtime_error("could not read, check filename");
    }
    t.read(file);
    file.close();
  }

  template <typename T>
  void loadFile(const char *filename, std::vector<T> &ts) {
    std::ifstream file(filename);
    if (!file.good()) {
      throw std::runtime_error("could not read, check filename");
    }
    for (auto &t : ts)
      t.read(file);
    file.close();
  }

  LayerType layer;
  int status;
  bool last_layer = false;
  nntrainer::Tensor in;
  nntrainer::Tensor out;
};

class nntrainer_InputLayer
  : public nntrainer_abstractLayer<nntrainer::InputLayer> {
protected:
  virtual void prepareLayer() { setInputDim("32:3:28:28"); }
};

/**
 * @brief Input Layer
 */
TEST_F(nntrainer_InputLayer, initialize_01_p) {
  int status = reinitialize();
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Input Layer
 */
TEST_F(nntrainer_InputLayer, setOptimizer_01_p) {
  status = setOptimizer(nntrainer::OptType::adam, "learning_rate=0.001 |"
                                                  "beta1=0.9 |"
                                                  "beta2=0.9999 |"
                                                  "epsilon=1e-7");

  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Input Layer
 */
TEST_F(nntrainer_InputLayer, setActivation_01_p) {
  int status = layer.setActivation(nntrainer::ACT_TANH);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Input Layer
 */
TEST_F(nntrainer_InputLayer, setActivation_02_n) {
  int status = layer.setActivation(nntrainer::ACT_UNKNOWN);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Input Layer
 */
TEST_F(nntrainer_InputLayer, checkValidation_01_p) {
  int status = layer.setActivation(nntrainer::ACT_TANH);
  ASSERT_EQ(status, ML_ERROR_NONE);

  status = layer.checkValidation();
  EXPECT_EQ(status, ML_ERROR_NONE);
}

class nntrainer_FullyConnectedLayer
  : public nntrainer_abstractLayer<nntrainer::FullyConnectedLayer> {
protected:
  virtual void prepareLayer() {
    setInputDim("32:1:28:28");
    setProperty("unit=1");
  }
};

/**
 * @brief Fully Connected Layer
 */
TEST_F(nntrainer_FullyConnectedLayer, initialize_01_p) {
  int status = reinitialize(false);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Fully Connected Layer without setting any parameter
 */
TEST(nntrainer_FullyConnectedLayer_n, initialize_02_n) {
  nntrainer::FullyConnectedLayer layer;
  EXPECT_THROW(layer.initialize(false), std::invalid_argument);
}

/**
 * @brief Fully Connected Layer without setting unit
 */
TEST(nntrainer_FullyConnectedLayer_n, initialize_03_n) {
  nntrainer::FullyConnectedLayer layer;
  nntrainer::TensorDim d;
  d.setTensorDim("32:1:28:28");
  layer.setInputDimension(d);

  EXPECT_THROW(layer.initialize(false), std::invalid_argument);
}

/**
 * @brief FullyConnected Layer
 */
TEST_F(nntrainer_FullyConnectedLayer, initialize_04_p) {
  std::string layer_name;

  /** Layer name can be set */
  layer_name = "FCLayer0";
  status = layer.setName(layer_name);
  EXPECT_EQ(status, ML_ERROR_NONE);
  EXPECT_EQ(layer.getName(), layer_name);

  status = layer.initialize(false);
  EXPECT_EQ(status, ML_ERROR_NONE);

  /** Layer name can be updated */
  layer_name = "FCLayer1";
  status = layer.setName(layer_name);
  EXPECT_EQ(status, ML_ERROR_NONE);
  EXPECT_EQ(layer.getName(), layer_name);
}

/**
 * @brief FullyConnected Layer
 */
TEST(nntrainer_FullyConnectedLayer_init_name, initialize_05_n) {
  int status = ML_ERROR_NONE;
  nntrainer::FullyConnectedLayer layer0, layer1;
  nntrainer::TensorDim d;
  std::string layer_name;

  /** no name is set */
  layer_name = layer0.getName();
  EXPECT_EQ(layer_name.length(), 0);

  /** Set empty name */
  status = layer0.setName(std::string());
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Fully Connected Layer
 */
TEST_F(nntrainer_FullyConnectedLayer, setOptimizer_01_p) {
  status = setOptimizer(nntrainer::OptType::adam, "learning_rate=0.001 |"
                                                  "beta1=0.9 |"
                                                  "beta2=0.9999 |"
                                                  "epsilon=1e-7");
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief FullyConnected Layer
 */
TEST_F(nntrainer_FullyConnectedLayer, setOptimizer_02_p) {
  status = setOptimizer(nntrainer::OptType::sgd, "learning_rate=0.1");
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Fully Connected Layer
 */
TEST_F(nntrainer_FullyConnectedLayer, setActivation_01_p) {
  status = layer.setActivation(nntrainer::ACT_TANH);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Fully Connected Layer
 */
TEST_F(nntrainer_FullyConnectedLayer, setActivation_02_n) {
  status = layer.setActivation(nntrainer::ACT_UNKNOWN);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief FullyConnected Layer
 */
TEST_F(nntrainer_FullyConnectedLayer, checkValidation_01_p) {
  layer.setActivation(nntrainer::ACT_RELU);
  status = layer.checkValidation();
  EXPECT_EQ(status, ML_ERROR_NONE);
}

class nntrainer_FullyConnectedLayer_TFmatch
  : public nntrainer_abstractLayer<nntrainer::FullyConnectedLayer> {

protected:
  typedef nntrainer_abstractLayer<nntrainer::FullyConnectedLayer> super;

  virtual int reinitialize(bool _last_layer = false) {
    int status = super::reinitialize(_last_layer);
    label = MAKE_SHARED_TENSOR(nntrainer::Tensor(layer.getOutputDimension()));

    loadFile("tc_fc_1_FCLayer.in", in);
    loadFile("tc_fc_1_FCKernel.in", layer);
    loadFile("tc_fc_1_FCLabel.in", label.get()[0]);
    layers.clear();

    return status;
  }

  void addActivation(nntrainer::ActiType type) {
    std::shared_ptr<nntrainer::ActivationLayer> act_layer =
      std::make_shared<nntrainer::ActivationLayer>();
    act_layer->setActivation(type);
    act_layer->setInputDimension(layer.getOutputDimension());
    status = act_layer->initialize(false);
    EXPECT_EQ(status, ML_ERROR_NONE);
    layers.push_back(act_layer);
  }

  void addLoss(nntrainer::CostType type) {
    std::shared_ptr<nntrainer::LossLayer> loss_layer =
      std::make_shared<nntrainer::LossLayer>();
    loss_layer->setInputDimension(layer.getOutputDimension());
    status = loss_layer->initialize(true);
    EXPECT_EQ(status, ML_ERROR_NONE);
    status = loss_layer->setCost(type);
    EXPECT_EQ(status, ML_ERROR_NONE);
    layers.push_back(loss_layer);
  }

  void matchForwarding(const char *file) {
    sharedTensor out;
    EXPECT_NO_THROW(out = layer.forwarding(MAKE_SHARED_TENSOR(in)));

    if (layers.size() > 0) {
      for (unsigned int idx = 0; idx < layers.size() - 1; idx++) {
        EXPECT_NO_THROW(out = layers[idx]->forwarding(out));
      }

      if (layers.back()->getType() == nntrainer::LAYER_LOSS) {
        std::shared_ptr<nntrainer::LossLayer> loss_layer =
          std::static_pointer_cast<nntrainer::LossLayer>(layers.back());
        EXPECT_NO_THROW(out = loss_layer->forwarding(out, label));
      } else {
        EXPECT_NO_THROW(out = layers.back()->forwarding(out));
      }
      EXPECT_EQ(status, ML_ERROR_NONE);
    }
    matchOutput(out.get()[0], file);
  }

  void matchLoss(const char *file) {
    nntrainer::Tensor loss;
    loadFile(file, loss);
    EXPECT_NEAR(layers.back()->getLoss(), *(loss.getData()), tolerance);
  }

  void matchBackwarding(const char *file_dx, const char *file_uw,
                        const char *file_g, const bool with_loss = false) {

    int idx = layers.size() - 1;
    sharedTensor def_derivative =
      MAKE_SHARED_TENSOR(constant(1.0, 3, 1, 1, 15));
    sharedTensor back_out;

    if (layers.size() && layers.back()->getType() == nntrainer::LAYER_LOSS) {
      if (with_loss) {
        EXPECT_NO_THROW(back_out = layers.back()->backwarding(label, 1));
      } else {
        back_out = def_derivative;
      }
      idx -= 1;
    } else {
      back_out = def_derivative;
    }

    for (; idx >= 0; --idx)
      EXPECT_NO_THROW(back_out = layers[idx]->backwarding(back_out, 1));

    EXPECT_NO_THROW(back_out = layer.backwarding(back_out, 1));
    matchOutput(*back_out.get(), file_dx);

    loadUpdatedWeightsGradients(file_uw, file_g);
    matchUpdatedWeightsGradients();
  }

  void loadUpdatedWeightsGradients(const char *file_uw, const char *file_g) {
    for (int idx = 0; idx < 2; ++idx) {
      new_w.push_back(nntrainer::Tensor(layer.paramsAt(idx).weight.getDim()));
      grad.push_back(nntrainer::Tensor(layer.paramsAt(idx).grad.getDim()));
    }

    loadFile(file_uw, new_w);
    loadFile(file_g, grad);
  }

  virtual void prepareLayer() {
    setInputDim("3:1:1:12");
    setProperty("unit=15");
    setProperty("bias_init_zero=true");
    last_layer = true;
  }

  void matchUpdatedWeightsGradients() {
    std::shared_ptr<nntrainer::UpdatableParam> params = layer.getParams();

    /** Match gradients and updated weights */
    for (int idx = 0; idx < 2; ++idx) {
      matchOutput(params.get()[idx].grad, grad[idx]);
      matchOutput(params.get()[idx].weight, new_w[idx]);
    }
  }

  sharedTensor label;
  std::vector<nntrainer::Tensor> new_w;
  std::vector<nntrainer::Tensor> grad;
  std::vector<std::shared_ptr<nntrainer::Layer>> layers;
};

/**
 * @brief Fully Connected Layer
 */
TEST_F(nntrainer_FullyConnectedLayer_TFmatch,
       DISABLED_forwarding_backwarding_00_p) {
  std::vector<float> weight_data;
  std::vector<float> bias_data;

  setOptimizer(nntrainer::OptType::adam, "learning_rate=1.0");

  sharedTensor out;

  EXPECT_NO_THROW(out = layer.forwarding(MAKE_SHARED_TENSOR(in)));

  nntrainer::Tensor derivatives(3, 1, 1, 15);

  for (unsigned int i = 0; i < derivatives.getDim().getDataLen(); ++i) {
    derivatives.getData()[i] = 1.0;
  }

  nntrainer::Tensor result;
  EXPECT_NO_THROW(
    result = layer.backwarding(MAKE_SHARED_TENSOR(derivatives), 1).get()[0]);

  matchOutput(result, "tc_fc_1_goldenFCGradientAdam.out");

  nntrainer::UpdatableParam *param_data = layer.getParams().get();

  nntrainer::UpdatableParam &param = param_data[0];
  nntrainer::Tensor &weight = param.weight;
  matchOutput(weight, "tc_fc_1_goldenFCUpdatedWeightAdam.out");

  nntrainer::UpdatableParam &bias_param = param_data[1];
  nntrainer::Tensor &bias = bias_param.weight;
  matchOutput(bias, "tc_fc_1_goldenFCUpdatedBiasAdam.out");
}

/**
 * @brief Fully Connected Layer
 */
TEST_F(nntrainer_FullyConnectedLayer_TFmatch,
       forwarding_backwarding_loss_00_p) {
  std::vector<float> weight_data;
  std::vector<float> bias_data;

  setOptimizer(nntrainer::OptType::adam, "learning_rate=0.0001");
  addLoss(nntrainer::COST_ENTROPY_SOFTMAX);

  matchForwarding("tc_fc_1_goldenFCResultSoftmaxCrossAdam.out");

  matchBackwarding("tc_fc_1_goldenFCGradientDxSoftmaxCrossAdam.out",
                   "tc_fc_1_goldenFCUpdatedWeightsSoftmaxCrossAdam.out",
                   "tc_fc_1_goldenFCGradientsSoftmaxCrossAdam.out", true);
}

/**
 * @brief Fully Connected Layer
 */
TEST_F(nntrainer_FullyConnectedLayer_TFmatch, forwarding_backwarding_01_p) {

  setOptimizer(nntrainer::OptType::sgd, "learning_rate=1.0");

  /** Verify forwarding and backwarding without loss */
  matchForwarding("tc_fc_1_goldenFCResultActNone.out");

  /** Verify backwarding without loss */
  matchBackwarding("tc_fc_1_goldenFCGradientDxActNone.out",
                   "tc_fc_1_goldenFCUpdatedWeightsActNone.out",
                   "tc_fc_1_goldenFCGradientsActNone.out");
}

/**
 * @brief Fully Connected Layer forward with MSE loss
 */
TEST_F(nntrainer_FullyConnectedLayer_TFmatch, forwarding_backwarding_02_p) {

  addActivation(nntrainer::ACT_SIGMOID);
  addLoss(nntrainer::COST_MSE);
  setOptimizer(nntrainer::OptType::sgd, "learning_rate=1.0");

  /** Verify forwarding value */
  matchForwarding("tc_fc_1_goldenFCResultSigmoidMse.out");

  /** Verify loss value */
  matchLoss("tc_fc_1_goldenFCLossSigmoidMse.out");

  /** Verify backwarding without loss */
  matchBackwarding("tc_fc_1_goldenFCGradientDxSigmoid.out",
                   "tc_fc_1_goldenFCUpdatedWeightsSigmoid.out",
                   "tc_fc_1_goldenFCGradientsSigmoid.out");
}

/**
 * @brief Fully Connected Layer forward with MSE loss
 */
TEST_F(nntrainer_FullyConnectedLayer_TFmatch, forwarding_backwarding_03_p) {

  addActivation(nntrainer::ACT_SOFTMAX);
  addLoss(nntrainer::COST_MSE);
  setOptimizer(nntrainer::OptType::sgd, "learning_rate=1.0");

  /** Verify forwarding value */
  matchForwarding("tc_fc_1_goldenFCResultSoftmaxMse.out");

  /** Verify loss value */
  matchLoss("tc_fc_1_goldenFCLossSoftmaxMse.out");

  /** Verify backwarding without loss */
  matchBackwarding("tc_fc_1_goldenFCGradientDxSoftmax.out",
                   "tc_fc_1_goldenFCUpdatedWeightsSoftmax.out",
                   "tc_fc_1_goldenFCGradientsSoftmax.out");
}

/**
 * @brief Fully Connected Layer
 */
TEST_F(nntrainer_FullyConnectedLayer_TFmatch, forwarding_backwarding_04_p) {

  addLoss(nntrainer::COST_MSE);
  setOptimizer(nntrainer::OptType::sgd, "learning_rate=1.0");

  /** Verify forwarding value */
  matchForwarding("tc_fc_1_goldenFCResultActNone.out");

  /** Verify loss value */
  matchLoss("tc_fc_1_goldenFCLossActNoneMse.out");

  /** Verify backwarding without loss */
  matchBackwarding("tc_fc_1_goldenFCGradientDxActNoneMse.out",
                   "tc_fc_1_goldenFCUpdatedWeightsActNoneMse.out",
                   "tc_fc_1_goldenFCGradientsActNoneMse.out", true);
}

/**
 * @brief Fully Connected Layer
 */
TEST_F(nntrainer_FullyConnectedLayer_TFmatch, forwarding_backwarding_05_p) {

  addActivation(nntrainer::ACT_SIGMOID);
  addLoss(nntrainer::COST_MSE);
  setOptimizer(nntrainer::OptType::sgd, "learning_rate=1.0");

  /** Verify forwarding value */
  matchForwarding("tc_fc_1_goldenFCResultSigmoidMse.out");

  /** Verify loss value */
  matchLoss("tc_fc_1_goldenFCLossSigmoidMse.out");

  /** Verify backwarding without loss */
  matchBackwarding("tc_fc_1_goldenFCGradientDxSigmoidMse.out",
                   "tc_fc_1_goldenFCUpdatedWeightsSigmoidMse.out",
                   "tc_fc_1_goldenFCGradientsSigmoidMse.out", true);
}

/**
 * @brief Fully Connected Layer
 */
TEST_F(nntrainer_FullyConnectedLayer_TFmatch, forwarding_backwarding_06_p) {

  addActivation(nntrainer::ACT_SOFTMAX);
  addLoss(nntrainer::COST_MSE);
  setOptimizer(nntrainer::OptType::sgd, "learning_rate=1.0");

  /** Verify forwarding value */
  matchForwarding("tc_fc_1_goldenFCResultSoftmaxMse.out");

  /** Verify loss value */
  matchLoss("tc_fc_1_goldenFCLossSoftmaxMse.out");

  /** Verify backwarding without loss */
  matchBackwarding("tc_fc_1_goldenFCGradientDxSoftmaxMse.out",
                   "tc_fc_1_goldenFCUpdatedWeightsSoftmaxMse.out",
                   "tc_fc_1_goldenFCGradientsSoftmaxMse.out", true);
}

/**
 * @brief Fully Connected Layer forward with Cross Entropy loss
 * @todo Upgrade this to adam to verify adam
 */
TEST_F(nntrainer_FullyConnectedLayer_TFmatch, forwarding_backwarding_07_p) {

  addLoss(nntrainer::COST_ENTROPY_SIGMOID);
  setOptimizer(nntrainer::OptType::sgd, "learning_rate=1.0");

  /** Verify forwarding value */
  matchForwarding("tc_fc_1_goldenFCResultSigmoidCross.out");

  /** Verify loss value */
  matchLoss("tc_fc_1_goldenFCLossSigmoidCross.out");

  /** Verify backwarding without loss */
  matchBackwarding("tc_fc_1_goldenFCGradientDxSigmoidCross.out",
                   "tc_fc_1_goldenFCUpdatedWeightsSigmoidCross.out",
                   "tc_fc_1_goldenFCGradientsSigmoidCross.out", true);
}

/**
 * @brief Fully Connected Layer forward with Cross Entropy loss
 * @todo Upgrade this to adam to verify adam
 */
TEST_F(nntrainer_FullyConnectedLayer_TFmatch, forwarding_backwarding_08_p) {

  addLoss(nntrainer::COST_ENTROPY_SOFTMAX);
  setOptimizer(nntrainer::OptType::sgd, "learning_rate=1.0");

  /** Verify forwarding value */
  matchForwarding("tc_fc_1_goldenFCResultSoftmaxCross.out");

  /** Verify loss value */
  matchLoss("tc_fc_1_goldenFCLossSoftmaxCross.out");

  /** Verify backwarding without loss */
  matchBackwarding("tc_fc_1_goldenFCGradientDxSoftmaxCross.out",
                   "tc_fc_1_goldenFCUpdatedWeightsSoftmaxCross.out",
                   "tc_fc_1_goldenFCGradientsSoftmaxCross.out", true);
}

class nntrainer_BatchNormalizationLayer
  : public nntrainer_abstractLayer<nntrainer::BatchNormalizationLayer> {
protected:
  typedef nntrainer_abstractLayer<nntrainer::BatchNormalizationLayer> super;

  virtual int reinitialize(bool _last_layer = false) {
    int status = super::reinitialize(_last_layer);
    loadFile("tc_bn_1_BNLayerInput.in", in);
    loadFile("tc_bn_1_BNLayerWeights.in", layer);
    return status;
  }

  virtual void prepareLayer() {
    setProperty("input_shape=3:1:4:5 | epsilon=0.001");
    setOptimizer(nntrainer::OptType::sgd, "learning_rate=1");
  }
};

/**
 * @brief Batch Normalization Layer
 */
TEST_F(nntrainer_BatchNormalizationLayer, initialize_01_p) {
  status = reinitialize();
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Batch Normalization Layer
 */
TEST_F(nntrainer_BatchNormalizationLayer, setOptimizer_01_p) {
  status = setOptimizer(
    nntrainer::OptType::adam,
    "learning_rate=0.001 | beta1=0.9 | beta2=0.9999 | epsilon=1e-7");
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Batch Normalization Layer
 */
TEST_F(nntrainer_BatchNormalizationLayer, setActivation_01_p) {
  status = layer.setActivation(nntrainer::ACT_SIGMOID);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Batch Normalization Layer
 */
TEST_F(nntrainer_BatchNormalizationLayer, setActivation_02_n) {
  status = layer.setActivation(nntrainer::ACT_UNKNOWN);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Batch Normalization Layer
 */
TEST_F(nntrainer_BatchNormalizationLayer, checkValidation_01_p) {
  status = layer.setActivation(nntrainer::ACT_RELU);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = layer.checkValidation();
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST_F(nntrainer_BatchNormalizationLayer,
       DISABLED_forward_backward_training_01_p) {
  layer.setTrainable(true);
  sharedTensor forward_result;

  EXPECT_NO_THROW(forward_result = layer.forwarding(MAKE_SHARED_TENSOR(in)));

  matchOutput(forward_result.get()[0], "tc_bn_1_goldenBNResultForward.out");

  nntrainer::Tensor backward_result;
  EXPECT_NO_THROW(
    backward_result =
      layer.backwarding(MAKE_SHARED_TENSOR(constant(1.0, 3, 1, 4, 5)), 1)
        .get()[0]);

  matchOutput(backward_result, "tc_bn_1_goldenBNLayerBackwardDx.out");
}

class nntrainer_Conv2DLayer
  : public nntrainer_abstractLayer<nntrainer::Conv2DLayer> {

protected:
  typedef nntrainer_abstractLayer<nntrainer::Conv2DLayer> super;

  virtual void prepareLayer() {
    int status =
      setProperty("input_shape=32:3:28:28 |"
                  "bias_init_zero=true |"
                  "activation=sigmoid |"
                  "weight_decay=l2norm |"
                  "weight_decay_lambda= 0.005 |"
                  "weight_ini=xavier_uniform |"
                  "normalization=true |"
                  "filter=12 | kernel_size= 5,5 | stride=3,3 | padding=1,1");

    EXPECT_EQ(status, ML_ERROR_NONE);
  }

  nntrainer::Tensor result;
};

TEST_F(nntrainer_Conv2DLayer, print_01_p) {
  setProperty("filter=3");
  reinitialize();
  unsigned int option = nntrainer::LayerPrintOption::PRINT_INST_INFO |
                        nntrainer::LayerPrintOption::PRINT_SHAPE_INFO |
                        nntrainer::LayerPrintOption::PRINT_PROP |
                        nntrainer::LayerPrintOption::PRINT_PROP_META |
                        nntrainer::LayerPrintOption::PRINT_WEIGHTS |
                        nntrainer::LayerPrintOption::PRINT_METRIC;
  layer.print(std::cerr, option);
}

/**
 * @brief Convolution 2D Layer
 */
TEST_F(nntrainer_Conv2DLayer, initialize_01_p) {
  status = reinitialize(true);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Convolution 2D Layer save and read and save
 */
TEST_F(nntrainer_Conv2DLayer, save_read_01_p) {
  saveFile("save.bin", layer);
  saveFile("save1.bin", layer);

  std::ifstream read_file("save.bin");
  ASSERT_TRUE(read_file.good());
  layer.read(read_file);

  read_file.seekg(0, std::ios::beg);

  std::ifstream read_file2("save1.bin");
  ASSERT_TRUE(read_file2.good());

  float d1, d2;

  for (int i = 0; i < (5 * 5 * 3 * 6) + 6; ++i) {
    read_file.read((char *)&d1, sizeof(float));
    read_file2.read((char *)&d2, sizeof(float));
    EXPECT_FLOAT_EQ(d1, d2);
  }

  read_file.close();
  read_file2.close();
}

/**
 * @brief Convolution 2D Layer
 */
TEST_F(nntrainer_Conv2DLayer, forwarding_01_p) {
  reinitialize("input_shape=1:3:7:7 |"
               "bias_init_zero = true |"
               "weight_ini=xavier_uniform |"
               "filter=2 | kernel_size=3,3 | stride=1, 1 | padding=0,0",
               true);

  ASSERT_EQ(in.getDim(), nntrainer::TensorDim(1, 3, 7, 7));
  ASSERT_EQ(out.getDim(), nntrainer::TensorDim(1, 2, 5, 5));

  loadFile("tc_conv2d_1_conv2DLayer.in", in);
  loadFile("tc_conv2d_1_conv2DKernel.in", layer);

  EXPECT_NO_THROW(out = layer.forwarding(MAKE_SHARED_TENSOR(in)).get()[0]);
  matchOutput(out, "tc_conv2d_1_goldenConv2DResult.out");
}

/**
 * @brief Convolution 2D Layer
 */

TEST_F(nntrainer_Conv2DLayer, forwarding_02_p) {
  reinitialize("input_shape=2:3:7:7 |"
               "bias_init_zero = true |"
               "weight_ini=xavier_uniform |"
               "filter=3 | kernel_size=3,3 | stride=1, 1 | padding=0,0",
               true);

  ASSERT_EQ(in.getDim(), nntrainer::TensorDim(2, 3, 7, 7));
  ASSERT_EQ(out.getDim(), nntrainer::TensorDim(2, 3, 5, 5));

  loadFile("tc_conv2d_2_conv2DLayer.in", in);
  loadFile("tc_conv2d_2_conv2DKernel.in", layer);

  EXPECT_NO_THROW(out = layer.forwarding(MAKE_SHARED_TENSOR(in)).get()[0]);
  matchOutput(out, "tc_conv2d_2_goldenConv2DResult.out");
}

TEST_F(nntrainer_Conv2DLayer, backwarding_01_p) {
  status = reinitialize("input_shape=1:3:7:7 |"
                        "bias_init_zero=true |"
                        "weight_ini=xavier_uniform |"
                        "filter=2 |"
                        "kernel_size= 3,3 |"
                        "stride=1, 1 |"
                        "padding=0,0",
                        true);

  setOptimizer(nntrainer::OptType::sgd, "learning_rate=1.0");
  unsigned int filter_size = 2;
  std::vector<float> grad_data;
  std::vector<float> weight_data;
  std::vector<float> bias_grad;
  std::vector<float> bias_weight;

  nntrainer::Tensor derivatives(1, 2, 5, 5);

  loadFile("tc_conv2d_1_conv2DLayer.in", in);
  loadFile("tc_conv2d_1_conv2DKernel.in", layer);

  EXPECT_NO_THROW(out = layer.forwarding(MAKE_SHARED_TENSOR(in)).get()[0]);

  for (unsigned int i = 0; i < derivatives.getDim().getDataLen(); ++i) {
    derivatives.getData()[i] = 1.0;
  }
  EXPECT_NO_THROW(
    result = layer.backwarding(MAKE_SHARED_TENSOR(derivatives), 1).get()[0]);

  nntrainer::UpdatableParam *param_data = layer.getParams().get();

  for (unsigned int i = 0; i < filter_size * 2; ++i) {
    nntrainer::UpdatableParam &param = param_data[i];
    nntrainer::Tensor &grad = param.grad;
    const float *gdata = grad.getData();
    if (i < filter_size) {
      for (unsigned int j = 0; j < grad.length(); ++j) {
        grad_data.push_back(gdata[j]);
      }
    } else {
      for (unsigned int j = 0; j < grad.length(); ++j) {
        bias_grad.push_back(gdata[j]);
      }
    }
  }

  matchOutput(grad_data, "tc_conv2d_1_goldenKernelGrad.out");

  matchOutput(result, "tc_conv2d_1_goldenInputGrad.out");

  matchOutput(bias_grad, "tc_conv2d_1_goldenBiasGrad.out");
}

TEST_F(nntrainer_Conv2DLayer, backwarding_02_p) {
  status = reinitialize("input_shape=2:3:7:7 |"
                        "bias_init_zero=true |"
                        "weight_ini=xavier_uniform |"
                        "filter=3 |"
                        "kernel_size= 3,3 |"
                        "stride=1, 1 |"
                        "padding=0,0",
                        true);

  setOptimizer(nntrainer::OptType::sgd, "learning_rate=1.0");

  unsigned int filter_size = 3;
  std::vector<float> grad_data;
  std::vector<float> weight_data;
  std::vector<float> bias_grad;
  std::vector<float> bias_weight;

  nntrainer::Tensor derivatives(2, 3, 5, 5);

  loadFile("tc_conv2d_2_conv2DLayer.in", in);
  loadFile("tc_conv2d_2_conv2DKernel.in", layer);

  EXPECT_NO_THROW(out = layer.forwarding(MAKE_SHARED_TENSOR(in)).get()[0]);

  for (unsigned int i = 0; i < derivatives.getDim().getDataLen(); ++i) {
    derivatives.getData()[i] = 1.0;
  }
  EXPECT_NO_THROW(
    result = layer.backwarding(MAKE_SHARED_TENSOR(derivatives), 1).get()[0]);
  nntrainer::UpdatableParam *param_data = layer.getParams().get();

  for (unsigned int i = 0; i < filter_size * 2; ++i) {
    nntrainer::UpdatableParam &param = param_data[i];
    nntrainer::Tensor &grad = param.grad;

    const float *gdata = grad.getData();
    if (i < filter_size) {
      for (unsigned int j = 0; j < grad.length(); ++j) {
        grad_data.push_back(gdata[j]);
      }
    } else {
      for (unsigned int j = 0; j < grad.length(); ++j) {
        bias_grad.push_back(gdata[j]);
      }
    }
  }

  matchOutput(grad_data, "tc_conv2d_2_goldenKernelGrad.out");

  matchOutput(result, "tc_conv2d_2_goldenInputGrad.out");

  matchOutput(bias_grad, "tc_conv2d_2_goldenBiasGrad.out");
}

class nntrainer_Pooling2DLayer
  : public nntrainer_abstractLayer<nntrainer::Pooling2DLayer> {
protected:
  void matchData(float *golden) {
    float *out_ptr;

    out_ptr = out.getData();

    for (size_t i = 0; i < out.getDim().getDataLen(); ++i) {
      EXPECT_NEAR(out_ptr[i], golden[i], tolerance);
    }
  }
};

TEST_F(nntrainer_Pooling2DLayer, setProperty_01_p) {
  setInputDim("2:3:5:5");
  setProperty("pooling_size=2,2 | stride=1,1 | padding=0,0 | pooling=average");
}

TEST_F(nntrainer_Pooling2DLayer, setProperty_02_n) {
  setInputDim("2:3:5:5");
  int status = layer.setProperty({"pooling_size="});
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

TEST_F(nntrainer_Pooling2DLayer, initialize_01_p) { reinitialize(); }

TEST_F(nntrainer_Pooling2DLayer, forwarding_01_p) {
  setInputDim("1:2:5:5");
  setProperty("pooling_size=2,2 | stride=1,1 | padding=0,0 | pooling=max");

  reinitialize();

  loadFile("tc_pooling2d_1.in", in);

  EXPECT_NO_THROW(out = layer.forwarding(MAKE_SHARED_TENSOR(in)).get()[0]);

  matchOutput(out, "tc_pooling2d_1_goldenPooling2Dmax.out");
}

TEST_F(nntrainer_Pooling2DLayer, forwarding_02_p) {
  setInputDim("1:2:5:5");
  setProperty("pooling_size=2,2 | stride=1,1 | padding=0,0 | pooling=average");

  reinitialize();

  loadFile("tc_pooling2d_1.in", in);

  EXPECT_NO_THROW(out = layer.forwarding(MAKE_SHARED_TENSOR(in)).get()[0]);

  matchOutput(out, "tc_pooling2d_1_goldenPooling2Daverage.out");
}

TEST_F(nntrainer_Pooling2DLayer, forwarding_03_p) {
  resetLayer();
  setInputDim("1:2:5:5");
  setProperty("pooling=global_max");
  reinitialize();

  loadFile("tc_pooling2d_1.in", in);

  EXPECT_NO_THROW(out = layer.forwarding(MAKE_SHARED_TENSOR(in)).get()[0]);

  matchOutput(out, "tc_pooling2d_1_goldenPooling2Dglobal_max.out");
}

TEST_F(nntrainer_Pooling2DLayer, forwarding_04_p) {
  resetLayer();
  setInputDim("1:2:5:5");
  setProperty("pooling=global_average");
  reinitialize();

  loadFile("tc_pooling2d_1.in", in);

  EXPECT_NO_THROW(out = layer.forwarding(MAKE_SHARED_TENSOR(in)).get()[0]);

  matchOutput(out, "tc_pooling2d_1_goldenPooling2Dglobal_average.out");
}

TEST_F(nntrainer_Pooling2DLayer, forwarding_05_p) {
  resetLayer();
  setInputDim("2:2:5:5");
  setProperty("pooling=global_max");
  reinitialize();

  loadFile("tc_pooling2d_2.in", in);
  EXPECT_NO_THROW(out = layer.forwarding(MAKE_SHARED_TENSOR(in)).get()[0]);
  matchOutput(out, "tc_pooling2d_2_goldenPooling2Dglobal_max.out");
}

TEST_F(nntrainer_Pooling2DLayer, forwarding_06_p) {
  resetLayer();
  setInputDim("2:2:5:5");
  setProperty("pooling=global_average");
  reinitialize();

  loadFile("tc_pooling2d_2.in", in);

  EXPECT_NO_THROW(out = layer.forwarding(MAKE_SHARED_TENSOR(in)).get()[0]);
  matchOutput(out, "tc_pooling2d_2_goldenPooling2Dglobal_average.out");
}

TEST_F(nntrainer_Pooling2DLayer, backwarding_01_p) {
  resetLayer();
  setInputDim("1:2:5:5");
  setProperty("pooling_size=2,2 | stride=1,1 | padding=0,0 | pooling=max");

  reinitialize();
  loadFile("tc_pooling2d_1.in", in);

  EXPECT_NO_THROW(out = layer.forwarding(MAKE_SHARED_TENSOR(in)).get()[0]);

  nntrainer::Tensor grad(out.getDim());

  for (unsigned int i = 0; i < grad.length(); ++i) {
    grad.getData()[i] = 1.0;
  }

  EXPECT_NO_THROW(in = layer.backwarding(MAKE_SHARED_TENSOR(grad), 0).get()[0]);

  matchOutput(in, "tc_pooling2d_1_goldenPooling2DmaxGrad.out");
}

TEST_F(nntrainer_Pooling2DLayer, backwarding_02_p) {
  resetLayer();
  setInputDim("1:2:5:5");
  setProperty("pooling_size=2,2 | stride=1,1 | padding=0,0 | pooling=average");
  reinitialize();
  loadFile("tc_pooling2d_1.in", in);

  EXPECT_NO_THROW(out = layer.forwarding(MAKE_SHARED_TENSOR(in)).get()[0]);

  sharedTensor grad = MAKE_SHARED_TENSOR(out.getDim());

  for (unsigned int i = 0; i < grad->length(); ++i) {
    grad->getData()[i] = 1.0;
  }

  EXPECT_NO_THROW(in = layer.backwarding(grad, 0).get()[0]);

  matchOutput(in, "tc_pooling2d_1_goldenPooling2DaverageGrad.out");
}

TEST_F(nntrainer_Pooling2DLayer, backwarding_03_p) {
  resetLayer();
  setInputDim("1:2:5:5");
  setProperty(
    "pooling_size=2,2 | stride=1,1 | padding=0,0 | pooling=global_max");
  reinitialize();

  loadFile("tc_pooling2d_1.in", in);

  EXPECT_NO_THROW(out = layer.forwarding(MAKE_SHARED_TENSOR(in)).get()[0]);

  nntrainer::Tensor grad(out.getDim());

  for (unsigned int i = 0; i < grad.getDim().getDataLen(); ++i) {
    grad.getData()[i] = 1.0;
  }

  EXPECT_NO_THROW(in = layer.backwarding(MAKE_SHARED_TENSOR(grad), 0).get()[0]);

  matchOutput(in, "tc_pooling2d_1_goldenPooling2Dglobal_maxGrad.out");
}

TEST_F(nntrainer_Pooling2DLayer, backwarding_04_p) {
  setInputDim("1:2:5:5");
  setProperty(
    "pooling_size=2,2 | stride=1,1 | padding=0,0 | pooling=global_average");
  reinitialize();
  loadFile("tc_pooling2d_1.in", in);

  EXPECT_NO_THROW(out = layer.forwarding(MAKE_SHARED_TENSOR(in)).get()[0]);

  nntrainer::Tensor grad(out.getDim());

  for (unsigned int i = 0; i < grad.getDim().getDataLen(); ++i) {
    grad.getData()[i] = 1.0;
  }

  EXPECT_NO_THROW(in = layer.backwarding(MAKE_SHARED_TENSOR(grad), 0).get()[0]);

  matchOutput(in, "tc_pooling2d_1_goldenPooling2Dglobal_averageGrad.out");
}

class nntrainer_FlattenLayer
  : public nntrainer_abstractLayer<nntrainer::FlattenLayer> {};

/**
 * @brief Flatten Layer
 */
TEST_F(nntrainer_FlattenLayer, forwarding_01_p) {
  setInputDim("1:2:4:4");
  reinitialize(false);

  EXPECT_EQ(out.getDim(), nntrainer::TensorDim(1, 1, 1, 32));

  loadFile("tc_pooling2d_1_goldenPooling2Dmax.out", in);

  EXPECT_NO_THROW(out = layer.forwarding(MAKE_SHARED_TENSOR(in)).get()[0]);

  matchOutput(out, "tc_pooling2d_1_goldenPooling2Dmax.out");
}

/**
 * @brief Flatten Layer
 */
TEST_F(nntrainer_FlattenLayer, forwarding_02_p) {
  setInputDim("2:2:4:4");
  reinitialize(false);

  EXPECT_EQ(out.getDim(), nntrainer::TensorDim(2, 1, 1, 32));

  loadFile("tc_pooling2d_2_goldenPooling2Dmax.out", in);

  EXPECT_NO_THROW(out = layer.forwarding(MAKE_SHARED_TENSOR(in)).get()[0]);

  matchOutput(out, "tc_pooling2d_2_goldenPooling2Dmax.out");
}

/**
 * @brief Flatten Layer
 */
TEST_F(nntrainer_FlattenLayer, backwarding_01_p) {
  setInputDim("1:2:4:4");
  reinitialize(false);

  EXPECT_EQ(out.getDim(), nntrainer::TensorDim(1, 1, 1, 32));

  loadFile("tc_pooling2d_1_goldenPooling2Dmax.out", out);

  EXPECT_NO_THROW(in = layer.backwarding(MAKE_SHARED_TENSOR(out), 0).get()[0]);
  EXPECT_EQ(in.getDim(), nntrainer::TensorDim(1, 2, 4, 4));

  matchOutput(in, "tc_pooling2d_1_goldenPooling2Dmax.out");
}

/**
 * @brief Flatten Layer
 */
TEST_F(nntrainer_FlattenLayer, backwarding_02_p) {
  setInputDim("2:2:4:4");
  reinitialize(false);

  EXPECT_EQ(out.getDim(), nntrainer::TensorDim(2, 1, 1, 32));

  loadFile("tc_pooling2d_2_goldenPooling2Dmax.out", out);

  EXPECT_NO_THROW(in = layer.backwarding(MAKE_SHARED_TENSOR(out), 0).get()[0]);
  EXPECT_EQ(in.getDim(), nntrainer::TensorDim(2, 2, 4, 4));

  matchOutput(in, "tc_pooling2d_2_goldenPooling2Dmax.out");
}

/**
 * @brief Loss Layer
 */
TEST(nntrainer_LossLayer, setCost_01_p) {
  int status = ML_ERROR_NONE;
  nntrainer::LossLayer layer;
  status = layer.setCost(nntrainer::COST_ENTROPY);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Loss Layer
 */
TEST(nntrainer_LossLayer, setCost_02_n) {
  int status = ML_ERROR_NONE;
  nntrainer::LossLayer layer;
  status = layer.setCost(nntrainer::COST_UNKNOWN);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Loss Layer
 */
TEST(nntrainer_LossLayer, setProperty_01_n) {
  int status = ML_ERROR_NONE;
  nntrainer::LossLayer layer;
  status = layer.setProperty({"loss=cross"});
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

TEST(nntrainer_ActivationLayer, init_01_1) {
  int status = ML_ERROR_NONE;
  nntrainer::ActivationLayer layer;
  status = layer.initialize(false);

  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST(nntrainer_ActivationLayer, setType_01_p) {
  nntrainer::ActivationLayer layer;
  EXPECT_NO_THROW(layer.setActivation(nntrainer::ACT_RELU));
  EXPECT_NO_THROW(layer.setActivation(nntrainer::ACT_SOFTMAX));
  EXPECT_NO_THROW(layer.setActivation(nntrainer::ACT_SIGMOID));
  EXPECT_NO_THROW(layer.setActivation(nntrainer::ACT_TANH));
}

TEST(nntrainer_ActivationLayer, setType_02_n) {
  nntrainer::ActivationLayer layer;
  EXPECT_THROW(layer.setActivation(nntrainer::ACT_UNKNOWN), std::runtime_error);
}

TEST(nntrainer_ActivationLayer, forward_backward_01_p) {
  int batch = 3;
  int channel = 1;
  int height = 1;
  int width = 10;

  nntrainer::ActivationLayer layer;
  layer.setActivation(nntrainer::ACT_RELU);

  nntrainer::Tensor input(batch, channel, height, width);
  GEN_TEST_INPUT(input, (l - 4) * 0.1 * (i + 1));
  nntrainer::Tensor expected(batch, channel, height, width);
  GEN_TEST_INPUT(expected,
                 nntrainer::ActivationLayer::relu((l - 4) * 0.1 * (i + 1)));
  nntrainer::Tensor result;
  EXPECT_NO_THROW(result =
                    layer.forwarding(MAKE_SHARED_TENSOR(input)).get()[0]);
  EXPECT_TRUE(result == expected);

  expected.copy(input);
  EXPECT_NO_THROW(
    result =
      layer.backwarding(MAKE_SHARED_TENSOR(constant(1.0, 3, 1, 1, 10)), 1)
        .get()[0]);
  GEN_TEST_INPUT(expected,
                 nntrainer::ActivationLayer::reluPrime(
                   nntrainer::ActivationLayer::relu((l - 4) * 0.1 * (i + 1))));
  EXPECT_TRUE(result == expected);
}

/**
 * @brief Main gtest
 */
int main(int argc, char **argv) {
  int result = -1;

  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error duing IniGoogleTest" << std::endl;
    return 0;
  }

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error duing RUN_ALL_TSETS()" << std::endl;
  }

  return result;
}
