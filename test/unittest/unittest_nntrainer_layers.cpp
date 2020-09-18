// SPDX-License-Identifier: Apache-2.0-only
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file        unittest_nntrainer_layers.cpp
 * @date        03 June 2020
 * @brief       Unit test utility for layers.
 * @see         https://github.com/nnstreamer/nntrainer
 * @author      Jijoong Moon <jijoong.moon@samsung.com>
 * @bug         No known bugs
 */
#include <fstream>

#include <activation_layer.h>
#include <addition_layer.h>
#include <bn_layer.h>
#include <conv2d_layer.h>
#include <fc_layer.h>
#include <flatten_layer.h>
#include <input_layer.h>
#include <layer.h>
#include <loss_layer.h>
#include <nntrainer_error.h>
#include <nntrainer_test_util.h>
#include <optimizer.h>
#include <pooling2d_layer.h>
#include <tensor_dim.h>
#include <util_func.h>

using nntrainer::sharedConstTensor;
using nntrainer::sharedTensor;

template <typename LayerType>
class nntrainer_abstractLayer : public ::testing::Test {
protected:
  virtual void SetUp() {
    status = ML_ERROR_NONE;
    prepareLayer();
    reinitialize();
  }

  virtual int reinitialize() {
    int status = layer.initialize();
    EXPECT_EQ(status, ML_ERROR_NONE);
    in = nntrainer::Tensor(layer.getInputDimension());
    out = nntrainer::Tensor(layer.getOutputDimension());
    return status;
  }

  virtual int reinitialize(const std::string str) {
    resetLayer();
    int status = setProperty(str);
    EXPECT_EQ(status, ML_ERROR_NONE);
    status = reinitialize();
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
      EXPECT_NEAR(out_ptr[i], golden_ptr[i], local_tolerance);
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
      EXPECT_NEAR(result[i], golden_ptr[i], local_tolerance);
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
  nntrainer::Tensor in;
  nntrainer::Tensor out;
  float local_tolerance = tolerance;
};

class nntrainer_InputLayer
  : public nntrainer_abstractLayer<nntrainer::InputLayer> {
protected:
  virtual void prepareLayer() {
    setInputDim("3:28:28");
    setProperty("batch_size=1");
  }
};

/**
 * @brief Input Layer
 */
TEST_F(nntrainer_InputLayer, initialize_01_p) {
  int status = reinitialize();
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST_F(nntrainer_InputLayer, set_property_01_n) {
  EXPECT_THROW(
    layer.setProperty(nntrainer::Layer::PropertyType::input_shape, "0:3:2:1"),
    std::invalid_argument);
}

TEST_F(nntrainer_InputLayer, set_property_02_p) {
  nntrainer::TensorDim dim;
  int status = setProperty("input_shape=3:2:1");
  EXPECT_EQ(status, ML_ERROR_NONE);

  dim = layer.getInputDimension();
  EXPECT_EQ(dim.getTensorDim(0), 1);
  EXPECT_EQ(dim.getTensorDim(1), 3);
  EXPECT_EQ(dim.getTensorDim(2), 2);
  EXPECT_EQ(dim.getTensorDim(3), 1);
}

TEST_F(nntrainer_InputLayer, set_property_03_p) {
  nntrainer::TensorDim dim;
  int status = setProperty("input_shape=1:3:2:1");
  EXPECT_EQ(status, ML_ERROR_NONE);

  dim = layer.getInputDimension();
  EXPECT_EQ(dim.getTensorDim(0), 1);
  EXPECT_EQ(dim.getTensorDim(1), 3);
  EXPECT_EQ(dim.getTensorDim(2), 2);
  EXPECT_EQ(dim.getTensorDim(3), 1);
}

TEST_F(nntrainer_InputLayer, set_property_04_p) {
  nntrainer::TensorDim dim;
  int status = setProperty("input_shape=4:3:2:1");
  EXPECT_EQ(status, ML_ERROR_NONE);

  /** Set input shape ignores batch size */
  dim = layer.getInputDimension();
  EXPECT_EQ(dim.getTensorDim(0), 1);
  EXPECT_EQ(dim.getTensorDim(1), 3);
  EXPECT_EQ(dim.getTensorDim(2), 2);
  EXPECT_EQ(dim.getTensorDim(3), 1);
}

TEST_F(nntrainer_InputLayer, set_property_05_p) {
  nntrainer::TensorDim dim;
  int status = ML_ERROR_NONE;

  status = setProperty("batch_size=5");
  EXPECT_EQ(status, ML_ERROR_NONE);

  dim = layer.getInputDimension();
  EXPECT_EQ(dim.getTensorDim(0), 5);
  EXPECT_EQ(dim.getTensorDim(1), 3);
  EXPECT_EQ(dim.getTensorDim(2), 28);
  EXPECT_EQ(dim.getTensorDim(3), 28);

  /** Original batch size is retained */
  status = setProperty("input_shape=1:3:2:1");
  EXPECT_EQ(status, ML_ERROR_NONE);

  dim = layer.getInputDimension();
  EXPECT_EQ(dim.getTensorDim(0), 5);
  EXPECT_EQ(dim.getTensorDim(1), 3);
  EXPECT_EQ(dim.getTensorDim(2), 2);
  EXPECT_EQ(dim.getTensorDim(3), 1);

  /** Original batch size is retained */
  status = setProperty("input_shape=4:3:2:1");
  EXPECT_EQ(status, ML_ERROR_NONE);

  dim = layer.getInputDimension();
  EXPECT_EQ(dim.getTensorDim(0), 5);
  EXPECT_EQ(dim.getTensorDim(1), 3);
  EXPECT_EQ(dim.getTensorDim(2), 2);
  EXPECT_EQ(dim.getTensorDim(3), 1);
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
    setInputDim("1:28:28");
    setProperty("batch_size=32");
    setProperty("unit=1");
  }
};

/**
 * @brief Fully Connected Layer
 */
TEST_F(nntrainer_FullyConnectedLayer, initialize_01_p) {
  int status = reinitialize();
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Fully Connected Layer without setting any parameter
 */
TEST(nntrainer_FullyConnectedLayer_n, initialize_02_n) {
  nntrainer::FullyConnectedLayer layer;
  EXPECT_THROW(layer.initialize(), std::invalid_argument);
}

/**
 * @brief Fully Connected Layer without setting unit
 */
TEST(nntrainer_FullyConnectedLayer_n, initialize_03_n) {
  nntrainer::FullyConnectedLayer layer;
  nntrainer::TensorDim d;
  d.setTensorDim("32:1:28:28");
  layer.setInputDimension(d);

  EXPECT_THROW(layer.initialize(), std::invalid_argument);
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

  virtual int reinitialize() {
    int status = super::reinitialize();
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
    status = act_layer->initialize();
    EXPECT_EQ(status, ML_ERROR_NONE);
    layers.push_back(act_layer);
  }

  void addLoss(nntrainer::LossType type) {
    std::shared_ptr<nntrainer::LossLayer> loss_layer =
      std::make_shared<nntrainer::LossLayer>();
    loss_layer->setInputDimension(layer.getOutputDimension());
    status = loss_layer->initialize();
    EXPECT_EQ(status, ML_ERROR_NONE);
    status = loss_layer->setLoss(type);
    EXPECT_EQ(status, ML_ERROR_NONE);
    layers.push_back(loss_layer);

    if (type == nntrainer::LossType::LOSS_ENTROPY_SOFTMAX) {
      loadFile("tc_fc_1_FCLayer_sensible.in", in);
      loadFile("tc_fc_1_FCKernel_sensible.in", layer);
      loadFile("tc_fc_1_FCLabel_sensible.in", label.get()[0]);
    }
  }

  void matchForwarding(const char *file) {
    sharedConstTensor out;
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
    nntrainer::Tensor loss(1, 1, 1, 1);
    loadFile(file, loss);
    EXPECT_NEAR(layers.back()->getLoss(), *(loss.getData()), local_tolerance);
  }

  void matchBackwarding(const char *file_dx, const char *file_uw,
                        const char *file_g, const bool with_loss = false) {

    int idx = layers.size() - 1;
    sharedTensor def_derivative =
      MAKE_SHARED_TENSOR(constant(1.0, 3, 1, 1, 15));
    sharedConstTensor back_out;

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
    setInputDim("1:1:12");
    setProperty("batch_size=3");
    setProperty("unit=15");
    setProperty("bias_initializer=zeros");
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

  sharedConstTensor out;

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
  addLoss(nntrainer::LossType::LOSS_ENTROPY_SOFTMAX);

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
  addLoss(nntrainer::LossType::LOSS_MSE);
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
  addLoss(nntrainer::LossType::LOSS_MSE);
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

  addLoss(nntrainer::LossType::LOSS_MSE);
  setOptimizer(nntrainer::OptType::sgd, "learning_rate=1.0");

  /** Verify forwarding value */
  matchForwarding("tc_fc_1_goldenFCResultActNone.out");
  matchOutput(label.get()[0], "tc_fc_1_FCLabel.in");

  /** Verify loss value */
  matchLoss("tc_fc_1_goldenFCLossActNoneMse.out");

  /**
   * This lowers the tolerance for below check. As the data values are in the
   * range [1, 10) (integer represented as floats), the values get very large
   * which leads to higher values floating point error.
   * This error exists in gradient. However, when added to weight with learning
   * rate of 1.0, this error disappears. So, for now, local tolerance just for
   * this test has been reduced to match the output.
   * Note: this issue occurs only for a single value out of matrix of 180
   * elements
   */
  local_tolerance = 1.3e-4;
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
  addLoss(nntrainer::LossType::LOSS_MSE);
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
  addLoss(nntrainer::LossType::LOSS_MSE);
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

  addLoss(nntrainer::LossType::LOSS_ENTROPY_SIGMOID);
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

  addLoss(nntrainer::LossType::LOSS_ENTROPY_SOFTMAX);
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

  virtual int reinitialize() {
    int status = super::reinitialize();
    loadFile("tc_bn_fc_1_BNLayerInput.in", in);
    loadFile("tc_bn_fc_1_BNLayerWeights.in", layer);
    return status;
  }

  virtual void prepareLayer() {
    setProperty(
      "input_shape=1:1:12 | epsilon=0.001 | batch_size=3 | momentum=0.90");
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

TEST_F(nntrainer_BatchNormalizationLayer, forward_backward_training_01_p) {
  layer.setTrainable(true);
  sharedConstTensor forward_result;

  EXPECT_NO_THROW(forward_result = layer.forwarding(MAKE_SHARED_TENSOR(in)));
  matchOutput(*forward_result, "tc_bn_fc_1_goldenBNResultForward.out");

  nntrainer::Tensor backward_in(layer.getOutputDimension());
  loadFile("tc_bn_fc_1_goldenBNLayerBackwardDxIn.out", backward_in);

  nntrainer::Tensor backward_result =
    *layer.backwarding(MAKE_SHARED_TENSOR(backward_in), 1);

  matchOutput(backward_result, "tc_bn_fc_1_goldenBNLayerBackwardDx.out");
}

class nntrainer_BatchNormalizationLayer_Conv
  : public nntrainer_abstractLayer<nntrainer::BatchNormalizationLayer> {
protected:
  typedef nntrainer_abstractLayer<nntrainer::BatchNormalizationLayer> super;

  virtual int reinitialize() {
    int status = super::reinitialize();
    loadFile("tc_bn_conv_1_BNLayerInput.in", in);
    loadFile("tc_bn_conv_1_BNLayerWeights.in", layer);
    return status;
  }

  virtual void prepareLayer() {
    setProperty(
      "input_shape=2:4:5 | epsilon=0.001 | batch_size=3 | momentum=0.90");
    setOptimizer(nntrainer::OptType::sgd, "learning_rate=1");
  }
};

TEST_F(nntrainer_BatchNormalizationLayer_Conv, forward_backward_training_01_p) {
  layer.setTrainable(true);
  sharedConstTensor forward_result;

  forward_result = layer.forwarding(MAKE_SHARED_TENSOR(in));
  matchOutput(*forward_result, "tc_bn_conv_1_goldenBNResultForward.out");

  nntrainer::Tensor backward_in(layer.getOutputDimension());
  loadFile("tc_bn_conv_1_goldenBNLayerBackwardDxIn.out", backward_in);

  nntrainer::Tensor backward_result =
    *layer.backwarding(MAKE_SHARED_TENSOR(backward_in), 1);

  matchOutput(backward_result, "tc_bn_conv_1_goldenBNLayerBackwardDx.out");
}

class nntrainer_BatchNormalizationLayer_Conv2
  : public nntrainer_abstractLayer<nntrainer::BatchNormalizationLayer> {
protected:
  typedef nntrainer_abstractLayer<nntrainer::BatchNormalizationLayer> super;

  virtual int reinitialize() {
    int status = super::reinitialize();
    loadFile("tc_bn_conv_2_BNLayerInput.in", in);
    loadFile("tc_bn_conv_2_BNLayerWeights.in", layer);
    return status;
  }

  virtual void prepareLayer() {
    setProperty(
      "input_shape=2:4:5 | epsilon=0.001 | batch_size=1 | momentum=0.90");
    setOptimizer(nntrainer::OptType::sgd, "learning_rate=1");
  }
};

TEST_F(nntrainer_BatchNormalizationLayer_Conv2,
       forward_backward_training_01_p) {
  layer.setTrainable(true);
  sharedConstTensor forward_result;

  forward_result = layer.forwarding(MAKE_SHARED_TENSOR(in));
  matchOutput(*forward_result, "tc_bn_conv_2_goldenBNResultForward.out");

  nntrainer::Tensor backward_in(layer.getOutputDimension());
  loadFile("tc_bn_conv_2_goldenBNLayerBackwardDxIn.out", backward_in);

  nntrainer::Tensor backward_result =
    *layer.backwarding(MAKE_SHARED_TENSOR(backward_in), 1);

  matchOutput(backward_result, "tc_bn_conv_2_goldenBNLayerBackwardDx.out");
}

class nntrainer_Conv2DLayer
  : public nntrainer_abstractLayer<nntrainer::Conv2DLayer> {

protected:
  typedef nntrainer_abstractLayer<nntrainer::Conv2DLayer> super;

  virtual void prepareLayer() {
    int status =
      setProperty("input_shape=3:28:28 | batch_size=32 |"
                  "bias_initializer=zeros |"
                  "activation=sigmoid |"
                  "weight_regularizer=l2norm |"
                  "weight_regularizer_constant= 0.005 |"
                  "weight_initializer=xavier_uniform |"
                  "normalization=true |"
                  "filters=12 | kernel_size= 5,5 | stride=3,3 | padding=1,1");

    EXPECT_EQ(status, ML_ERROR_NONE);
  }

  nntrainer::Tensor result;
};

TEST_F(nntrainer_Conv2DLayer, print_01_p) {
  std::stringstream ss;
  unsigned int option = nntrainer::LayerPrintOption::PRINT_INST_INFO |
                        nntrainer::LayerPrintOption::PRINT_SHAPE_INFO |
                        nntrainer::LayerPrintOption::PRINT_PROP |
                        nntrainer::LayerPrintOption::PRINT_PROP_META |
                        nntrainer::LayerPrintOption::PRINT_WEIGHTS |
                        nntrainer::LayerPrintOption::PRINT_METRIC;
  layer.print(ss, option);
  EXPECT_GT(ss.str().size(), 100);
}

/**
 * @brief Convolution 2D Layer
 */
TEST_F(nntrainer_Conv2DLayer, initialize_01_p) {
  status = reinitialize();
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
  reinitialize("input_shape=3:7:7 | batch_size=1 |"
               "bias_initializer = zeros |"
               "weight_initializer=xavier_uniform |"
               "filters=2 | kernel_size=3,3 | stride=1, 1 | padding=0,0");

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
  status =
    reinitialize("input_shape=3:7:7 | batch_size=2 |"
                 "bias_initializer = zeros |"
                 "weight_initializer=xavier_uniform |"
                 "filters=3 | kernel_size=3,3 | stride=1, 1 | padding=0,0");

  ASSERT_EQ(in.getDim(), nntrainer::TensorDim(2, 3, 7, 7));
  ASSERT_EQ(out.getDim(), nntrainer::TensorDim(2, 3, 5, 5));

  loadFile("tc_conv2d_2_conv2DLayer.in", in);
  loadFile("tc_conv2d_2_conv2DKernel.in", layer);

  EXPECT_NO_THROW(out = layer.forwarding(MAKE_SHARED_TENSOR(in)).get()[0]);
  matchOutput(out, "tc_conv2d_2_goldenConv2DResult.out");
}

TEST_F(nntrainer_Conv2DLayer, backwarding_01_p) {
  status = reinitialize("input_shape=3:7:7 | batch_size=1 |"
                        "bias_initializer=zeros |"
                        "weight_initializer=xavier_uniform |"
                        "filters=2 |"
                        "kernel_size= 3,3 |"
                        "stride=1, 1 |"
                        "padding=0,0");

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

  matchOutput(rotate_180(result), "tc_conv2d_1_goldenInputGrad.out");

  matchOutput(bias_grad, "tc_conv2d_1_goldenBiasGrad.out");
}

TEST_F(nntrainer_Conv2DLayer, backwarding_04_p) {
  status = reinitialize("input_shape=6:24:24 | batch_size=1 |"
                        "bias_initializer=zeros |"
                        "weight_initializer=xavier_uniform |"
                        "filters=12 |"
                        "kernel_size=5,5 |"
                        "stride=1,1 |"
                        "padding=0,0");

  setOptimizer(nntrainer::OptType::sgd, "learning_rate=1.0");
  unsigned int filter_size = 12;
  std::vector<float> grad_data;
  std::vector<float> weight_data;
  std::vector<float> bias_grad;
  std::vector<float> bias_weight;

  nntrainer::Tensor derivatives(1, 12, 20, 20);

  loadFile("tc_conv2d_3_conv2DLayer.in", in);
  loadFile("tc_conv2d_3_conv2DKernel.in", layer);

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

  matchOutput(grad_data, "tc_conv2d_3_goldenKernelGrad.out");

  matchOutput(rotate_180(result), "tc_conv2d_3_goldenInputGrad.out");

  matchOutput(bias_grad, "tc_conv2d_3_goldenBiasGrad.out");
}

TEST_F(nntrainer_Conv2DLayer, backwarding_02_p) {
  status = reinitialize("input_shape=3:7:7 | batch_size=2 |"
                        "bias_initializer=zeros |"
                        "weight_initializer=xavier_uniform |"
                        "filters=3 |"
                        "kernel_size= 3,3 |"
                        "stride=1, 1 |"
                        "padding=0,0");

  setOptimizer(nntrainer::OptType::sgd, "learning_rate=1.0");

  unsigned int filter_size = 3;
  std::vector<float> grad_data;
  std::vector<float> weight_data;
  std::vector<float> bias_grad;
  std::vector<float> bias_weight;
  nntrainer::UpdatableParam *param_data;

  nntrainer::Tensor derivatives(2, 3, 5, 5);

  loadFile("tc_conv2d_2_conv2DLayer.in", in);
  loadFile("tc_conv2d_2_conv2DKernel.in", layer);

  EXPECT_NO_THROW(out = layer.forwarding(MAKE_SHARED_TENSOR(in)).get()[0]);

  for (unsigned int i = 0; i < derivatives.getDim().getDataLen(); ++i) {
    derivatives.getData()[i] = 1.0;
  }
  EXPECT_NO_THROW(
    result = layer.backwarding(MAKE_SHARED_TENSOR(derivatives), 1).get()[0]);
  param_data = layer.getParams().get();

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

  matchOutput(out, "tc_conv2d_2_goldenConv2DResult.out");
  matchOutput(grad_data, "tc_conv2d_2_goldenKernelGrad.out");
  matchOutput(rotate_180(result), "tc_conv2d_2_goldenInputGrad.out");
  matchOutput(bias_grad, "tc_conv2d_2_goldenBiasGrad.out");

  for (int i = 0; i < 4; i++) {
    EXPECT_NO_THROW(out = layer.forwarding(MAKE_SHARED_TENSOR(in)).get()[0]);
    EXPECT_NO_THROW(
      result = layer.backwarding(MAKE_SHARED_TENSOR(derivatives), 1).get()[0]);
  }

  param_data = layer.getParams().get();

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

  matchOutput(out, "tc_conv2d_2_goldenConv2DResult2.out");
  matchOutput(grad_data, "tc_conv2d_2_goldenKernelGrad2.out");
  matchOutput(rotate_180(result), "tc_conv2d_2_goldenInputGrad2.out");
  matchOutput(bias_grad, "tc_conv2d_2_goldenBiasGrad2.out");
}

#ifdef USE_BLAS
TEST_F(nntrainer_Conv2DLayer, backwarding_03_p) {
  status = reinitialize("input_shape=3:28:28 | batch_size=1 |"
                        "bias_initializer=zeros |"
                        "weight_initializer=zeros |"
                        "filters=6 |"
                        "kernel_size= 5,5 |"
                        "stride=1, 1 |"
                        "padding=0, 0");

  nntrainer::Conv2DLayer layer1;
  status = layer1.setProperty(
    {"input_shape=3:28:28", "batch_size=1", "bias_initializer=zeros",
     "weight_initializer=zeros", "filters=6", "kernel_size= 5,5", "stride=1, 1",
     "padding=0, 0"});
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = layer1.initialize();
  EXPECT_EQ(status, ML_ERROR_NONE);

  nntrainer::Conv2DLayer layer2;
  status = layer2.setProperty(
    {"batch_size=1", "bias_initializer=zeros", "weight_initializer=zeros",
     "filters=12", "kernel_size= 1,1", "stride=1, 1", "padding=0, 0"});
  EXPECT_EQ(status, ML_ERROR_NONE);
  layer2.setInputDimension(layer1.getOutputDimension());
  status = layer2.initialize();
  EXPECT_EQ(status, ML_ERROR_NONE);

  nntrainer::Optimizer op;
  int status = op.setType(nntrainer::OptType::sgd);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = op.setProperty({"learning_rate=1.0"});
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = layer1.setOptimizer(op);
  EXPECT_EQ(status, ML_ERROR_NONE);

  nntrainer::Optimizer op2;
  status = op2.setType(nntrainer::OptType::sgd);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = op2.setProperty({"learning_rate=1.0"});
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = layer2.setOptimizer(op2);
  EXPECT_EQ(status, ML_ERROR_NONE);

  setOptimizer(nntrainer::OptType::sgd, "learning_rate=1.0");

  unsigned int filter_size;
  std::vector<float> grad_data;
  std::vector<float> weight_data;
  std::vector<float> bias_grad;
  std::vector<float> bias_weight;
  nntrainer::UpdatableParam *param_data;

  nntrainer::Tensor derivatives(1, 12, 24, 24);

  loadFile("tc_conv2d_int_conv2DLayer.in", in);
  loadFile("tc_conv2d_int_conv2DKernel.in", layer1);
  loadFile("tc_conv2d_int_conv2DKernel2.in", layer2);

  nntrainer::Tensor out1;
  EXPECT_NO_THROW(out1 = layer1.forwarding(MAKE_SHARED_TENSOR(in)).get()[0]);

  nntrainer::Tensor out2;

  EXPECT_NO_THROW(out2 = layer2.forwarding(MAKE_SHARED_TENSOR(out1)).get()[0]);

  matchOutput(out1, "tc_conv2d_int_goldenConv2DResult.out");
  matchOutput(out2, "tc_conv2d_int_goldenConv2DResult2.out");

  for (unsigned int i = 0; i < derivatives.getDim().getDataLen(); ++i) {
    derivatives.getData()[i] = 1.0;
  }

  nntrainer::Tensor result2;
  EXPECT_NO_THROW(
    result2 = layer2.backwarding(MAKE_SHARED_TENSOR(derivatives), 1).get()[0]);

  EXPECT_NO_THROW(
    result = layer1.backwarding(MAKE_SHARED_TENSOR(result2), 1).get()[0]);

  /** Compare second conv */
  param_data = layer2.getParams().get();
  filter_size = 12;
  grad_data.clear();
  bias_grad.clear();
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

  matchOutput(grad_data, "tc_conv2d_int_goldenKernel2Grad.out");
  matchOutput(bias_grad, "tc_conv2d_int_goldenBias2Grad.out");

  /** Compare first conv */
  param_data = layer1.getParams().get();
  filter_size = 6;
  grad_data.clear();
  bias_grad.clear();
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

  matchOutput(grad_data, "tc_conv2d_int_goldenKernelGrad.out");
  matchOutput(bias_grad, "tc_conv2d_int_goldenBiasGrad.out");

  matchOutput(rotate_180(result), "tc_conv2d_int_goldenInputGrad.out");
}
#endif

class nntrainer_Pooling2DLayer
  : public nntrainer_abstractLayer<nntrainer::Pooling2DLayer> {
protected:
  void matchData(float *golden) {
    float *out_ptr;

    out_ptr = out.getData();

    for (size_t i = 0; i < out.getDim().getDataLen(); ++i) {
      EXPECT_NEAR(out_ptr[i], golden[i], local_tolerance);
    }
  }

  virtual void prepareLayer() { setInputDim("2:3:5:5"); }
};

TEST_F(nntrainer_Pooling2DLayer, setProperty_01_p) {
  setInputDim("3:5:5");
  setProperty("batch_size=2");
  setProperty("pool_size=2,2 | stride=1,1 | padding=0,0 | pooling=average");
}

TEST_F(nntrainer_Pooling2DLayer, setProperty_02_n) {
  setInputDim("3:5:5");
  setProperty("batch_size=2");
  int status = layer.setProperty({"pool_size="});
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

TEST_F(nntrainer_Pooling2DLayer, initialize_01_p) { reinitialize(); }

TEST_F(nntrainer_Pooling2DLayer, forwarding_01_p) {
  setInputDim("2:5:5");
  setProperty("pool_size=2,2 | stride=1,1 | padding=0,0 | pooling=max");

  reinitialize();

  loadFile("tc_pooling2d_1.in", in);

  EXPECT_NO_THROW(out = layer.forwarding(MAKE_SHARED_TENSOR(in)).get()[0]);

  matchOutput(out, "tc_pooling2d_1_goldenPooling2Dmax.out");
}

TEST_F(nntrainer_Pooling2DLayer, forwarding_02_p) {
  setInputDim("2:5:5");
  setProperty("pool_size=2,2 | stride=1,1 | padding=0,0 | pooling=average");

  reinitialize();

  loadFile("tc_pooling2d_1.in", in);

  EXPECT_NO_THROW(out = layer.forwarding(MAKE_SHARED_TENSOR(in)).get()[0]);

  matchOutput(out, "tc_pooling2d_1_goldenPooling2Daverage.out");
}

TEST_F(nntrainer_Pooling2DLayer, forwarding_03_p) {
  resetLayer();
  setInputDim("2:5:5");
  setProperty("pooling=global_max");
  reinitialize();

  loadFile("tc_pooling2d_1.in", in);

  EXPECT_NO_THROW(out = layer.forwarding(MAKE_SHARED_TENSOR(in)).get()[0]);

  matchOutput(out, "tc_pooling2d_1_goldenPooling2Dglobal_max.out");
}

TEST_F(nntrainer_Pooling2DLayer, forwarding_04_p) {
  resetLayer();
  setInputDim("2:5:5");
  setProperty("pooling=global_average");
  reinitialize();

  loadFile("tc_pooling2d_1.in", in);

  EXPECT_NO_THROW(out = layer.forwarding(MAKE_SHARED_TENSOR(in)).get()[0]);

  matchOutput(out, "tc_pooling2d_1_goldenPooling2Dglobal_average.out");
}

TEST_F(nntrainer_Pooling2DLayer, forwarding_05_p) {
  resetLayer();
  setInputDim("2:5:5");
  setProperty("batch_size=2");
  setProperty("pooling=global_max");
  reinitialize();

  loadFile("tc_pooling2d_2.in", in);
  EXPECT_NO_THROW(out = layer.forwarding(MAKE_SHARED_TENSOR(in)).get()[0]);
  matchOutput(out, "tc_pooling2d_2_goldenPooling2Dglobal_max.out");
}

TEST_F(nntrainer_Pooling2DLayer, forwarding_06_p) {
  resetLayer();
  setInputDim("2:5:5");
  setProperty("batch_size=2");
  setProperty("pooling=global_average");
  reinitialize();

  loadFile("tc_pooling2d_2.in", in);

  EXPECT_NO_THROW(out = layer.forwarding(MAKE_SHARED_TENSOR(in)).get()[0]);
  matchOutput(out, "tc_pooling2d_2_goldenPooling2Dglobal_average.out");
}

TEST_F(nntrainer_Pooling2DLayer, backwarding_01_p) {
  resetLayer();
  setInputDim("2:5:5");
  setProperty("pool_size=2,2 | stride=1,1 | padding=0,0 | pooling=max");

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
  setInputDim("2:5:5");
  setProperty("pool_size=2,2 | stride=1,1 | padding=0,0 | pooling=average");
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
  setInputDim("2:5:5");
  setProperty("pool_size=2,2 | stride=1,1 | padding=0,0 | pooling=global_max");
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
  setInputDim("2:5:5");
  setProperty(
    "pool_size=2,2 | stride=1,1 | padding=0,0 | pooling=global_average");
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
  : public nntrainer_abstractLayer<nntrainer::FlattenLayer> {
protected:
  virtual void prepareLayer() {
    setInputDim("2:4:4");
    layer.setBatch(1);
  }
};

/**
 * @brief Flatten Layer
 */
TEST_F(nntrainer_FlattenLayer, forwarding_01_p) {
  reinitialize();

  EXPECT_EQ(out.getDim(), nntrainer::TensorDim(1, 1, 1, 32));

  loadFile("tc_pooling2d_1_goldenPooling2Dmax.out", in);

  EXPECT_NO_THROW(out = layer.forwarding(MAKE_SHARED_TENSOR(in)).get()[0]);

  matchOutput(out, "tc_pooling2d_1_goldenPooling2Dmax.out");
}

/**
 * @brief Flatten Layer
 */
TEST_F(nntrainer_FlattenLayer, forwarding_02_p) {
  setInputDim("2:4:4");
  layer.setBatch(2);
  reinitialize();

  EXPECT_EQ(out.getDim(), nntrainer::TensorDim(2, 1, 1, 32));

  loadFile("tc_pooling2d_2_goldenPooling2Dmax.out", in);

  EXPECT_NO_THROW(out = layer.forwarding(MAKE_SHARED_TENSOR(in)).get()[0]);

  matchOutput(out, "tc_pooling2d_2_goldenPooling2Dmax.out");
}

/**
 * @brief Flatten Layer
 */
TEST_F(nntrainer_FlattenLayer, backwarding_01_p) {
  reinitialize();

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
  setInputDim("2:4:4");
  layer.setBatch(2);
  reinitialize();

  EXPECT_EQ(out.getDim(), nntrainer::TensorDim(2, 1, 1, 32));

  loadFile("tc_pooling2d_2_goldenPooling2Dmax.out", out);

  EXPECT_NO_THROW(in = layer.backwarding(MAKE_SHARED_TENSOR(out), 0).get()[0]);
  EXPECT_EQ(in.getDim(), nntrainer::TensorDim(2, 2, 4, 4));

  matchOutput(in, "tc_pooling2d_2_goldenPooling2Dmax.out");
}

/**
 * @brief Loss Layer
 */
TEST(nntrainer_LossLayer, setLoss_01_p) {
  int status = ML_ERROR_NONE;
  nntrainer::LossLayer layer;
  status = layer.setLoss(nntrainer::LossType::LOSS_ENTROPY);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Loss Layer
 */
TEST(nntrainer_LossLayer, setLoss_02_n) {
  int status = ML_ERROR_NONE;
  nntrainer::LossLayer layer;
  status = layer.setLoss(nntrainer::LossType::LOSS_UNKNOWN);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

TEST(nntrainer_LossLayer, forward_nolabel_n) {
  nntrainer::LossLayer layer;
  nntrainer::Tensor a = constant(1.0, 1, 1, 1, 1);
  EXPECT_THROW(layer.forwarding(MAKE_SHARED_TENSOR(a)), std::runtime_error);
}

TEST(nntrainer_LossLayer, forward_loss_unknown_n) {
  nntrainer::LossLayer layer;
  nntrainer::Tensor a = constant(1.0, 1, 1, 1, 1);
  nntrainer::Tensor b = constant(1.0, 1, 1, 1, 1);
  EXPECT_THROW(layer.forwarding(MAKE_SHARED_TENSOR(a), MAKE_SHARED_TENSOR(b)),
               std::runtime_error);
}

TEST(nntrainer_LossLayer, backward_loss_unknown_n) {
  nntrainer::LossLayer layer;
  nntrainer::Tensor a = constant(1.0, 1, 1, 1, 1);
  EXPECT_THROW(layer.backwarding(MAKE_SHARED_TENSOR(a), 1), std::runtime_error);
}

TEST(nntrainer_LossLayer, forward_loss_forward_entropy_n) {
  nntrainer::LossLayer layer;
  layer.setLoss(nntrainer::LossType::LOSS_ENTROPY);
  nntrainer::Tensor a = constant(1.0, 1, 1, 1, 1);
  nntrainer::Tensor b = constant(1.0, 1, 1, 1, 1);
  EXPECT_THROW(layer.forwarding(MAKE_SHARED_TENSOR(a), MAKE_SHARED_TENSOR(b)),
               std::runtime_error);
}

TEST(nntrainer_LossLayer, backward_loss_backward_entropy_n) {
  nntrainer::LossLayer layer;
  layer.setLoss(nntrainer::LossType::LOSS_ENTROPY);
  nntrainer::Tensor a = constant(1.0, 1, 1, 1, 1);
  EXPECT_THROW(layer.backwarding(MAKE_SHARED_TENSOR(a), 1), std::runtime_error);
}

/**
 * @brief Loss Layer
 */
TEST(nntrainer_LossLayer, setProperty_through_vector_n) {
  int status = ML_ERROR_NONE;
  nntrainer::LossLayer layer;
  status = layer.setProperty({"loss=cross"});
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

TEST(nntrainer_LossLayer, setProperty_individual_n) {
  nntrainer::LossLayer layer;
  EXPECT_THROW(
    layer.setProperty(nntrainer::Layer::PropertyType::input_shape, "1:2:3:4"),
    nntrainer::exception::not_supported);
}

TEST(nntrainer_LossLayer, setProperty_individual2_n) {
  nntrainer::LossLayer layer;
  EXPECT_THROW(
    layer.setProperty(nntrainer::Layer::PropertyType::filters, "1:2"),
    nntrainer::exception::not_supported);
}

TEST(nntrainer_LossLayer, setProperty_individual3_n) {
  nntrainer::LossLayer layer;
  EXPECT_THROW(layer.setProperty(nntrainer::Layer::PropertyType::input_shape,
                                 "invalid_string"),
               nntrainer::exception::not_supported);
}

TEST(nntrainer_LossLayer, setProperty_individual4_n) {
  nntrainer::LossLayer layer;
  EXPECT_THROW(layer.setProperty(nntrainer::Layer::PropertyType::filters,
                                 "invalid_string"),
               nntrainer::exception::not_supported);
}

TEST(nntrainer_ActivationLayer, init_01_n) {
  nntrainer::ActivationLayer layer;
  EXPECT_THROW(layer.initialize(), std::invalid_argument);
}

TEST(nntrainer_ActivationLayer, init_02_p) {
  int status = ML_ERROR_NONE;
  nntrainer::ActivationLayer layer;

  layer.setInputDimension({1, 1, 1, 1});
  status = layer.initialize();
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
 * @brief Addition Layer
 */
class nntrainer_AdditionLayer
  : public nntrainer_abstractLayer<nntrainer::AdditionLayer> {
protected:
  virtual void prepareLayer() {
    setInputDim("32:3:28:28");
    setProperty("num_inputs=1");
  }
};

TEST_F(nntrainer_AdditionLayer, initialize_01_p) {
  status = reinitialize();
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST_F(nntrainer_AdditionLayer, initialize_02_n) {
  nntrainer::AdditionLayer layer;
  layer.setProperty({"input_shape=1:1:1:1"});
  status = layer.initialize();
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

TEST_F(nntrainer_AdditionLayer, checkValidation_01_p) {
  status = layer.checkValidation();
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST_F(nntrainer_AdditionLayer, setProperty_01_p) {
  setProperty("num_inputs=10");
  status = layer.initialize();
  EXPECT_EQ(status, ML_ERROR_NONE);
}

TEST_F(nntrainer_AdditionLayer, setProperty_02_n) {
  status = layer.setProperty({"num_inputs=0"});
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

TEST_F(nntrainer_AdditionLayer, forwarding_01_n) {
  setProperty("num_inputs=1");

  sharedTensor input = std::shared_ptr<nntrainer::Tensor>(
    new nntrainer::Tensor[1], std::default_delete<nntrainer::Tensor[]>());
  nntrainer::Tensor &in = input.get()[0];

  in = nntrainer::Tensor();

  EXPECT_THROW(layer.forwarding(input), std::logic_error);
}

TEST_F(nntrainer_AdditionLayer, forwarding_02_n) {
  setProperty("num_inputs=2");

  sharedTensor input = std::shared_ptr<nntrainer::Tensor>(
    new nntrainer::Tensor[1], std::default_delete<nntrainer::Tensor[]>());
  nntrainer::Tensor &in = input.get()[0];

  in = nntrainer::Tensor(layer.getInputDimension());

  EXPECT_THROW(layer.forwarding(input), std::runtime_error);
}

TEST_F(nntrainer_AdditionLayer, forwarding_03_p) {
  setProperty("num_inputs=2");

  sharedTensor input = std::shared_ptr<nntrainer::Tensor>(
    new nntrainer::Tensor[2], std::default_delete<nntrainer::Tensor[]>());
  nntrainer::Tensor &in = input.get()[0];
  in = nntrainer::Tensor(layer.getInputDimension());

  input.get()[1] = input.get()[0];

  EXPECT_NO_THROW(layer.forwarding(input));
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
