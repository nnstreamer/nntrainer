// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file fc_layer_nnapi.cpp
 * @date 10 December 2020
 * @brief This is NNAPI implementation of Fully Connected Layer
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include "fc_layer_nnapi.h"

#include <memory>
#if DEBUG
#include <cassert>
#endif
#include <sstream>
#include <sys/mman.h>

#include <android/sharedmem.h>

#include <nntrainer_log.h>
#include <profiler.h>

namespace nntrainer {

namespace {
typedef enum { FCweight = 0, FCbias = 1 } FcParams;

#define NNAPI_RETURN_STATUS(status, msg)      \
  do {                                        \
    if (status != ANEURALNETWORKS_NO_ERROR) { \
      ml_loge(msg " status: %d", status);     \
      return status;                          \
    }                                         \
  } while (0);

#define NNAPI_THROW_STATUS(status, msg)                \
  do {                                                 \
    if (status != ANEURALNETWORKS_NO_ERROR) {          \
      ml_loge(msg " status: %d", status);              \
      throw std::invalid_argument("nnapi ops failed"); \
    }                                                  \
  } while (0);

class ANeuralNetworksOperandType_Scoped {
public:
  /**
   * @brief Construct an operandType from dimension info
   *
   * @param dim_ dimension
   * @param dim_padding if this is otherten -1, dimension count is
   * explicitly padded with 1 eg) 1:1:1:1000 will be
   * 1000 with dim_padding = -1 or 1
   * will be 1:1000 with dim_padding = 2
   * @param type_ int type from nnapi header
   */
  ANeuralNetworksOperandType_Scoped(
    const TensorDim &dim_, int dim_padding = -1,
    uint32_t type_ = ANEURALNETWORKS_TENSOR_FLOAT32) :
    data(new ANeuralNetworksOperandType) {

    // for uninitialized tensor, rank is 0
    if (dim_.rank() > 0) {
      for (unsigned int i = 0; i < dim_.getNumDim(); ++i) {
        auto cur_dim = dim_.getTensorDim(i);

        if (cur_dim <= 1) {
          continue;
        }

        dim.push_back(cur_dim);
      }
    }

    int num_fill = dim_padding - dim.size();
    if (num_fill > 0) {
      for (int i = 0; i < num_fill; ++i) {
        dim.insert(dim.begin(), 1);
      }
    }

    data->type = type_;
    data->scale = 0.f;
    data->zeroPoint = 0;
    data->dimensionCount = dim.size();
    data->dimensions = dim.empty() ? nullptr : dim.data();
  }

  /**
   * @brief Construct a Type object with intializer list
   *
   * @param dim if dimension is 0, it means unspecified (later determined)
   * @param type_ type enum
   */
  ANeuralNetworksOperandType_Scoped(
    const std::initializer_list<uint32_t> &vector_dim_,
    uint32_t type_ = ANEURALNETWORKS_TENSOR_FLOAT32) :
    dim(vector_dim_),
    data(new ANeuralNetworksOperandType) {
    data->type = type_;
    data->scale = 0.f;
    data->zeroPoint = 0;
    data->dimensionCount = dim.size();
    data->dimensions = dim.empty() ? nullptr : dim.data();
  }

  ANeuralNetworksOperandType_Scoped(
    uint32_t scalartype_ = ANEURALNETWORKS_INT32) :
    ANeuralNetworksOperandType_Scoped(TensorDim(), -1, scalartype_) {}

  ANeuralNetworksOperandType *get() { return data.get(); }

  friend std::ostream &operator<<(std::ostream &out,
                                  ANeuralNetworksOperandType_Scoped &d) {
    out << "dim addr: " << d.data->dimensions << '\n';
    out << "type: " << d.data->type << '\n';
    out << "dimension count: " << d.data->dimensionCount << '\n';
    out << "dim: ";
    for (auto &i : d.dim) {
      out << i << ' ';
    }
    out << '\n';

    return out;
  }

private:
  std::vector<uint32_t> dim;
  std::unique_ptr<ANeuralNetworksOperandType> data;
};
} // namespace

const std::string FullyConnectedLayer_NNAPI::type = "fully_connected_nnapi";

FullyConnectedLayer_NNAPI::~FullyConnectedLayer_NNAPI() {
  ANeuralNetworksCompilation_free(nnapi_compilation);
  nnapi_compilation = nullptr;
  ANeuralNetworksModel_free(nnapi_model);
  nnapi_model = nullptr;
}

int FullyConnectedLayer_NNAPI::initialize(Manager &manager) {
  FullyConnectedLayer::initialize(manager);
  int status = ML_ERROR_NONE;

  status = ANeuralNetworksModel_create(&nnapi_model);
  NNAPI_RETURN_STATUS(status, "create model failed");

  /** build operations */
  /// @note leaving unspecified batch to 0 is in nnapi specification but it is
  /// not working for eden-driver, so for exinos, it is not usuable for the
  /// batched input
  ANeuralNetworksOperandType_Scoped input_type({0, input_dim[0].width()});
  ANeuralNetworksOperandType_Scoped output_type({0, output_dim[0].width()});
  ANeuralNetworksOperandType_Scoped weight_type(weights[FCweight].getDim());
  ANeuralNetworksOperandType_Scoped bias_type(weights[FCbias].getDim());
  ANeuralNetworksOperandType_Scoped activation_type(ANEURALNETWORKS_INT32);

  status = ANeuralNetworksModel_addOperand(nnapi_model,
                                           input_type.get()); // operand 0;
  NNAPI_RETURN_STATUS(status, "addOperand failed");

  status = ANeuralNetworksModel_addOperand(nnapi_model,
                                           weight_type.get()); // operand 1;
  NNAPI_RETURN_STATUS(status, "addOperand failed");

  status = ANeuralNetworksModel_addOperand(nnapi_model,
                                           bias_type.get()); // operand 2;
  NNAPI_RETURN_STATUS(status, "addOperand failed");

  status = ANeuralNetworksModel_addOperand(nnapi_model,
                                           activation_type.get()); // operand 3;
  NNAPI_RETURN_STATUS(status, "addOperand failed");
  int32_t act_none = ANEURALNETWORKS_FUSED_NONE;
  status = ANeuralNetworksModel_setOperandValue(nnapi_model, 3, &act_none,
                                                sizeof(act_none));
  NNAPI_RETURN_STATUS(status, "setting act to none failed");

  status = ANeuralNetworksModel_addOperand(nnapi_model,
                                           output_type.get()); // operand 4;
  NNAPI_RETURN_STATUS(status, "addOperand failed");

  // operand 0 -> 2D input
  // operand 1 -> 2D weight
  // operand 2 -> 1D bias
  // operand 3 -> Activation fused
  // operand 4 -> output

  uint32_t input_idxes[4] = {0, 1, 2, 3};
  uint32_t output_idxes[1] = {4};
  status = ANeuralNetworksModel_addOperation(nnapi_model,
                                             ANEURALNETWORKS_FULLY_CONNECTED, 4,
                                             input_idxes, 1, output_idxes);
  NNAPI_RETURN_STATUS(status, "addOperation failed");

  uint32_t model_input_idxes[1] = {0};
  uint32_t model_output_idxes[4] = {4};
  status = ANeuralNetworksModel_identifyInputsAndOutputs(
    nnapi_model, 1, model_input_idxes, 1, model_output_idxes);
  NNAPI_RETURN_STATUS(status, "Identify model output failed");

  return status;
}

int FullyConnectedLayer_NNAPI::postInitialize(Manager &manager) {
  int status = -1;
  Tensor &weight = weightAt(FCweight).getVariableRef();
  Tensor &bias = weightAt(FCbias).getVariableRef();

  int fd = weight.getFd();
  size_t total_size = weight.getSize() + bias.getSize();
  ml_logd("fd: %d", fd);
#if DEBUG
  cassert(fd != -1);
#endif

  uint32_t num_devices = -1;
  status = ANeuralNetworks_getDeviceCount(&num_devices);
  ml_loge("device cnt: %u, api: %d", num_devices, __ANDROID_API__);
  if (status != ANEURALNETWORKS_NO_ERROR) {
    NNAPI_RETURN_STATUS(status, "ANeuralNetworksworks_getDeviceCount failed");
  }

  ANeuralNetworksDevice *current_devices[num_devices - 1];

  /// finalize model with assigned memory
  status = ANeuralNetworksMemory_createFromFd(
    total_size, PROT_READ, fd, weight.getOffset(), &nnapi_weight_mem);
  NNAPI_RETURN_STATUS(status, "creating memory failed");

  ml_logd("offset: %d, size: %zu", 0, weight.getSize());
  status = ANeuralNetworksModel_setOperandValueFromMemory(
    nnapi_model, 1, nnapi_weight_mem, 0, weight.getSize());
  NNAPI_RETURN_STATUS(status, "setting operand value failed");

  status = ANeuralNetworksModel_setOperandValueFromMemory(
    nnapi_model, 2, nnapi_weight_mem, weight.getSize(), bias.getSize());
  NNAPI_RETURN_STATUS(status, "setting operand value faild");

  status = ANeuralNetworksModel_finish(nnapi_model);
  NNAPI_RETURN_STATUS(status, "finishing model failed");

  /// query the available devices and supported ops
  for (int i = 0; i < num_devices; ++i) {
    ANeuralNetworksDevice *device;
    status = ANeuralNetworks_getDevice(i, &device);
    if (status != ANEURALNETWORKS_NO_ERROR) {
      NNAPI_RETURN_STATUS(status, "ANeuralNetworksworks_getDevice failed");
    }
    if (i <= num_devices - 1) {
      current_devices[i] = device;
    }

    bool supported_ops[1];
    ANeuralNetworksModel_getSupportedOperationsForDevices(nnapi_model, &device,
                                                          1, supported_ops);

    int64_t feature_level;
    const char *name;
    status = ANeuralNetworksDevice_getName(device, &name);
    if (status != ANEURALNETWORKS_NO_ERROR) {
      NNAPI_RETURN_STATUS(status, "ANeuralNetworksworks_getFeaturename failed");
    }

    status = ANeuralNetworksDevice_getFeatureLevel(device, &feature_level);
    if (status != ANEURALNETWORKS_NO_ERROR) {
      NNAPI_RETURN_STATUS(status,
                          "ANeuralNetworksworks_getFeatureLevel failed");
    }

    ml_logd("device name: %s, supported feature level %ld, model supported: %d",
            name, feature_level, supported_ops[0]);
  }

  /// compile the model
  status = ANeuralNetworksCompilation_createForDevices(
    nnapi_model, current_devices, num_devices - 1, &nnapi_compilation);
  // status = ANeuralNetworksCompilation_create(nnapi_model,
  // &nnapi_compilation);
  NNAPI_RETURN_STATUS(status, "creating model compilation failed");

  status = ANeuralNetworksCompilation_setPreference(
    nnapi_compilation, ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER);
  NNAPI_RETURN_STATUS(status, "setting preference for the compilation failed");

  status = ANeuralNetworksCompilation_finish(nnapi_compilation);
  NNAPI_RETURN_STATUS(status, "finishing compilation failed");

  return status;
}

void FullyConnectedLayer_NNAPI::forwarding(sharedConstTensors in) {
  int status = -1;

  Tensor &weight = weightAt(FCweight).getVariableRef();
  Tensor &bias = weightAt(FCbias).getVariableRef();

  status = ANeuralNetworksExecution_create(nnapi_compilation, &nnapi_execution);
  NNAPI_THROW_STATUS(status, "executing compilation failed");

  Tensor &hidden_ = net_hidden[0]->var;
  Tensor &input_ = net_input[0]->var;

  ANeuralNetworksOperandType_Scoped input_type(input_dim[0], 2);
  ANeuralNetworksOperandType_Scoped output_type(output_dim[0], 2);

  int iofd = ASharedMemory_create("", input_.getSize() + hidden_.getSize());
  if (iofd < 0) {
    NNAPI_THROW_STATUS(-1, "creating shared memory failed");
  }

  // ANeuralNetworksMemory *mem2;
  // status =
  //   ANeuralNetworksMemory_createFromFd(input_.getSize() + hidden_.getSize(),
  //                                      PROT_READ | PROT_WRITE, iofd, 0,
  //                                      &mem2);
  // NNAPI_THROW_STATUS(status, "creating mem failed");

  // status = ANeuralNetworksExecution_setInputFromMemory(
  //   nnapi_execution, 0, input_type.get(), mem2, 0, input_.getSize());
  // NNAPI_THROW_STATUS(status, "set input failed");

  // status = ANeuralNetworksExecution_setOutputFromMemory(
  //   nnapi_execution, 0, output_type.get(), mem2, input_.getSize(),
  //   hidden_.getSize());
  // NNAPI_THROW_STATUS(status, "set output failed");

  // status = ANeuralNetworksExecution_setOutputFromMemory(
  //   nnapi_execution, 0, output_type.get(), mem2, input_.getSize(),
  //   hidden_.getSize());
  // NNAPI_THROW_STATUS(status, "set input failed");

  /// @todo change this to setInput/outputFromMemory
  status = ANeuralNetworksExecution_setInput(
    nnapi_execution, 0, input_type.get(), input_.getData(), input_.getSize());
  NNAPI_THROW_STATUS(status, "setting input for the execution failed");

  status =
    ANeuralNetworksExecution_setOutput(nnapi_execution, 0, output_type.get(),
                                       hidden_.getData(), hidden_.getSize());
  NNAPI_THROW_STATUS(status, "setting output for the execution failed");

  START_PROFILE(-101);
  status = ANeuralNetworksExecution_compute(nnapi_execution);
  NNAPI_THROW_STATUS(status, "computing failed for the execution failed");
  END_PROFILE(-101);

  // ANeuralNetworksMemory_free(mem2);
  // mem2 = nullptr;
  ANeuralNetworksExecution_free(nnapi_execution);
  nnapi_execution = nullptr;
}

// void FullyConnectedLayer_NNAPI::calcDerivative(sharedConstTensors in) {}

// void FullyConnectedLayer_NNAPI::calcGradient(sharedConstTensors in) {}
} // namespace nntrainer
