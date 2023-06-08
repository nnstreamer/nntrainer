// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   main.cpp
 * @date   24 Jun 2021
 * @brief  task runner for the refinedet
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */
#include "tensor.h"
#include <array>
#include <chrono>
#include <ctime>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#if defined(ENABLE_TEST)
#include <gtest/gtest.h>
#endif

#include <layer.h>
#include <model.h>
#include <optimizer.h>
#include <app_context.h>
#include <refinedet_loss.h>

#include <det_dataloader.h>

#ifdef PROFILE
#include <profiler.h>
#endif

using LayerHandle = std::shared_ptr<ml::train::Layer>;
using ModelHandle = std::unique_ptr<ml::train::Model>;

using UserDataType = std::unique_ptr<nntrainer::util::DirDataLoader>;

/** cache loss values post training for test */
float training_loss = 0.0;
float validation_loss = 0.0;

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

const std::string input_shape = "224:224:3";
const unsigned int feature_map_size1 = 28;
const unsigned int feature_map_size2 = 14;
const unsigned int feature_map_size3 = 4;
const unsigned int feature_map_size4 = 2;
const unsigned int total_feature_map = (
  feature_map_size1 * feature_map_size1 +
  feature_map_size2 * feature_map_size2 +
  feature_map_size3 * feature_map_size3 +
  feature_map_size4 * feature_map_size4);
const unsigned int num_ratios = 3;
const unsigned int num_anchors = num_ratios;
const unsigned int num_classes = 20;

/**
 * @brief Feature extractor
 *
 * @param input_name name of the input
 * @return std::vector<LayerHandle> vectors of layers
 */
std::vector<LayerHandle> featureExtractor(const std::string &input_name) {
  using ml::train::createLayer;

  auto scoped_name = [](const std::string &layer_name) {
    return "feature_extractor/" + layer_name;
  };

  auto with_name = [](const std::string &layer_name) {
    return withKey("name", "feature_extractor/" + layer_name);
  };

  auto createConv = [&with_name](const std::string &name,
                                           int kernel_size, int stride,
                                           int filters, bool use_relu,
                                           const std::string &padding,
                                           const std::string &input_layer) {
    std::vector<std::string> props{
      with_name(name),
      withKey("stride", {stride, stride}),
      withKey("filters", filters),
      withKey("kernel_size", {kernel_size, kernel_size}),
      withKey("padding", padding),
      withKey("input_layers", input_layer)};

    if (use_relu) {
      props.push_back(withKey("activation", "relu"));
    }

    return createLayer("conv2d", props);
  };

  auto createMaxpool = [&with_name](const std::string &name,
                                           int pool_size,
                                           const std::string &input_layer) {
    std::vector<std::string> props{
      with_name(name),
      withKey("pool_size", {pool_size, pool_size}),
      withKey("stride", {pool_size, pool_size}),
      withKey("pooling", "max"),
      withKey("input_layers", input_layer)};

    return createLayer("pooling2d", props);
  };

  LayerHandle permute = createLayer("permute", {
  with_name("permute"),
  withKey("direction", {3,1,2}),
  withKey("input_layers", input_name)});

  return {
    // createConv("conv1_1", 3, 1, 64, true, "same", input_name),
    // createConv("conv1_2", 3, 1, 64, true, "same", scoped_name("conv1_1")),
    // createMaxpool("pool1", 2, scoped_name("conv1_2")),
    // createConv("conv2_1", 3, 1, 128, true, "same", scoped_name("pool1")),
    // createConv("conv2_2", 3, 1, 128, true, "same", scoped_name("conv2_1")),
    // createMaxpool("pool2", 2, scoped_name("conv2_2")),
    // createConv("conv3_1", 3, 1, 256, true, "same", scoped_name("pool2")),
    // createConv("conv3_2", 3, 1, 256, true, "same", scoped_name("conv3_1")),
    // createConv("conv3_3", 3, 1, 256, true, "same", scoped_name("conv3_2")),
    // createMaxpool("pool3", 2, scoped_name("conv3_3")),
    // createConv("conv4_1", 3, 1, 512, true, "same", scoped_name("pool3")),
    // createConv("conv4_2", 3, 1, 512, true, "same", scoped_name("conv4_1")),
    // createConv("conv4_3", 3, 1, 512, true, "same", scoped_name("conv4_2")),
    // createMaxpool("pool4", 2, scoped_name("conv4_3")),
    // createConv("conv5_1", 3, 1, 512, true, "same", scoped_name("pool4")),
    // createConv("conv5_2", 3, 1, 512, true, "same", scoped_name("conv5_1")),
    // createConv("conv5_3", 3, 1, 512, true, "same", scoped_name("conv5_2")),
    permute,
    createMaxpool("pool5", 2, scoped_name("permute")),
    createLayer("conv2d", {
      with_name("conv6"),
      withKey("stride", {1, 1}),
      withKey("filters", 1024),
      withKey("kernel_size", {3, 3}),
      withKey("padding", "same"),
      withKey("input_layers", scoped_name("pool5")),
      withKey("dilation", {2, 2})
      }),

    createConv("conv7", 1, 1, 1024, true, "same", scoped_name("conv6")),
    createConv("conv8_1", 1, 1, 256, true, "same", scoped_name("conv7")),
    createConv("conv8_2", 3, 2, 512, true, "same", scoped_name("conv8_1")),
    createConv("conv9_1", 1, 1, 256, true, "same", scoped_name("conv8_2")),
    createConv("conv9_2", 3, 2, 512, true, "same", scoped_name("conv9_1")),
    createConv("conv10_1", 1, 1, 256, true, "same", scoped_name("conv9_2")),
    createConv("conv10_2", 3, 1, 256, true, "same", scoped_name("conv10_1")),
  };
  
}

/**
 * @brief ARM(Anchor Refinement Module)
 *
 * @param block_name name of the block
 * @param input_name name of the input
 * @param num_anchors number of anchors
 * @return std::vector<LayerHandle> vectors of layers
 */
std::vector<LayerHandle> ARM(const std::string &block_name,
                              const std::string &input_name,
                              bool do_permute) {
  using ml::train::createLayer;

  auto scoped_name = [block_name](const std::string &layer_name) {
    return block_name + "/" + layer_name;
  };
  auto with_name = [&scoped_name](const std::string &layer_name) {
    return withKey("name", scoped_name(layer_name));
  };

  auto createConv = [&with_name](const std::string &name,
                                           int kernel_size, int stride, int filters, 
                                           const std::string &padding,
                                           const std::string &input_layer) {
    std::vector<std::string> props{
      with_name(name),
      withKey("stride", {stride, stride}),
      withKey("filters", filters),
      withKey("kernel_size", {kernel_size, kernel_size}),
      withKey("padding", padding),
      withKey("input_layers", input_layer)};

    return createLayer("conv2d", props);
  };

  auto createBN = [&](bool use_relu, const std::string &input_layer) {
    std::vector<std::string> props{
      withKey("name", input_layer + "_bn"),
      withKey("input_layers", {input_layer})
    };

    if (use_relu) {
      props.push_back(withKey("activation", "relu"));
    }

    return createLayer("batch_normalization", props);
  };

  LayerHandle permute = createLayer("permute", {
  with_name("permute"),
  withKey("direction", {3,1,2}),
  withKey("input_layers", input_name)});

  LayerHandle conv1 = nullptr;
  if (do_permute) {
    conv1 = createConv("conv1", 3, 1, 256, "same", scoped_name("permute"));
  }
  else {
    conv1 = createConv("conv1", 3, 1, 256, "same", input_name);
  }

  std::vector<LayerHandle> ret = {
    conv1,
    createBN(true, scoped_name("conv1")),
    createConv("conv2", 3, 1, 256, "same", scoped_name("conv1_bn")),
    createBN(true, scoped_name("conv2")),
    createConv("conv3", 3, 1, 256, "same", scoped_name("conv2_bn")),
    createBN(true, scoped_name("conv3")),
    createConv("conv4", 3, 1, 256, "same", scoped_name("conv3_bn")),
    createBN(true, scoped_name("conv4")),
    createConv("ploc", 3, 1, 4 * num_anchors, "same", scoped_name("conv4_bn")),
    createBN(false, scoped_name("ploc")),
    createConv("pconf", 3, 1, 2 * num_anchors, "same", scoped_name("conv4_bn")),
    createBN(false, scoped_name("pconf")),
  };

  if (do_permute) {
    ret.push_back(permute);
  }

  return ret;
}

/**
 * @brief TCB(Transfer Connection Block)
 *
 * @param block_name name of the block
 * @param input_name name of the input
 * @param upsample_input_name name of the upsample input
 * @param upsample_input_available whether upsample input exists
 * @return std::vector<LayerHandle> vectors of layers
 */
std::vector<LayerHandle> tcbBlock(const std::string &block_name,
                                  const std::string &input_name,
                                  const std::string &upsample_input_name,
                                  bool do_permute,
                                  bool upsample_input_available) {
  using ml::train::createLayer;

  auto scoped_name = [&block_name](const std::string &layer_name) {
    return block_name + "/" + layer_name;
  };
  auto with_name = [&scoped_name](const std::string &layer_name) {
    return withKey("name", scoped_name(layer_name));
  };

  auto createConv = [&with_name](const std::string &name,
                                           int kernel_size, int stride, int filters,
                                           const std::string &padding,
                                           const std::string &input_layer) {
    std::vector<std::string> props{
      with_name(name),
      withKey("stride", {stride, stride}),
      withKey("filters", filters),
      withKey("kernel_size", {kernel_size, kernel_size}),
      withKey("padding", padding),
      withKey("input_layers", input_layer)};

    return createLayer("conv2d", props);
  };

  auto createDeconv = [&with_name](const std::string &name,
                                              int kernel_size, int stride, int filters,
                                              const std::string &input_layer) {
    std::vector<std::string> props{
      with_name(name),
      withKey("stride", {stride, stride}),
      withKey("filters", filters),
      withKey("kernel_size", {kernel_size, kernel_size}),
      withKey("padding", "same"),
      withKey("input_layers", input_layer)};

    return createLayer("convtranspose2d", props);
  };

  auto createBN = [&](bool use_relu, const std::string &input_layer) {
    std::vector<std::string> props{
      withKey("name", input_layer + "_bn"),
      withKey("input_layers", {input_layer})
    };

    if (use_relu) {
      props.push_back(withKey("activation", "relu"));
    }

    return createLayer("batch_normalization", props);
  };

  LayerHandle permute = createLayer("permute", {
  with_name("permute"),
  withKey("direction", {3,1,2}),
  withKey("input_layers", input_name)});

  // From ARM
  LayerHandle conv1 = nullptr;
  if (do_permute) {
    conv1 = createConv("conv1", 3, 1, 256, "same", scoped_name("permute"));
  }
  else {
    conv1 = createConv("conv1", 3, 1, 256, "same", input_name);
  }
  
  LayerHandle bn1 = createBN(true, scoped_name("conv1"));
  LayerHandle conv2 = createConv("conv2", 3, 1, 256, "same", scoped_name("conv1_bn"));
  LayerHandle bn2 = createBN(false, scoped_name("conv2"));
  
  LayerHandle upsample = nullptr;
  LayerHandle add = nullptr;
  LayerHandle upsample_bn = nullptr;
  std::vector<LayerHandle> ret;
  if (upsample_input_available) {
    upsample = createDeconv("upsample", 4, 2, 256, upsample_input_name);
    upsample_bn = createBN(false, scoped_name("upsample"));
    add = createLayer(
      "addition", {
        with_name("add"), 
        withKey("input_layers", {scoped_name("conv2_bn"), scoped_name("upsample_bn")}),
        withKey("activation", "relu")
        });
    LayerHandle conv3 = createConv("conv3", 3, 1, 256, "same", scoped_name("add"));
    LayerHandle bn3 = createBN(true, scoped_name("conv3"));
    ret = {conv1, bn1, conv2, bn2, conv3, bn3, add, upsample, upsample_bn};
  }
  LayerHandle conv3 = createConv("conv3", 3, 1, 256, "same", scoped_name("conv2_bn"));
  LayerHandle bn3 = createBN(true, scoped_name("conv3"));
  ret = {conv1, bn1, conv2, bn2, conv3, bn3};
  if (do_permute) {
    ret.push_back(permute);
  }

  return ret;
}

/**
 * @brief ODM(Object Detection Module)
 *
 * @param block_name name of the block
 * @param input_name name of the input
 * @param num_anchors number of anchors
 * @param num_classes number of classes
 * @return std::vector<LayerHandle> vectors of layers
 */
std::vector<LayerHandle> ODM(const std::string &block_name,
                              const std::string &input_name) {
  using ml::train::createLayer;

  auto scoped_name = [block_name](const std::string &layer_name) {
    return block_name + "/" + layer_name;
  };
  auto with_name = [&scoped_name](const std::string &layer_name) {
    return withKey("name", scoped_name(layer_name));
  };

  auto createConv = [&with_name](const std::string &name,
                                           int kernel_size, int stride, int filters,
                                           const std::string &padding,
                                           const std::string &input_layer) {
    std::vector<std::string> props{
      with_name(name),
      withKey("stride", {stride, stride}),
      withKey("filters", filters),
      withKey("kernel_size", {kernel_size, kernel_size}),
      withKey("padding", padding),
      withKey("input_layers", input_layer)};

    return createLayer("conv2d", props);
  };

  auto createBN = [&](bool use_relu, const std::string &input_layer) {
    std::vector<std::string> props{
      withKey("name", input_layer + "_bn"),
      withKey("input_layers", {input_layer})
    };

    if (use_relu) {
      props.push_back(withKey("activation", "relu"));
    }

    return createLayer("batch_normalization", props);
  };

  return {
    createConv("conv1", 3, 1, 256, "same", input_name),
    createBN(true, scoped_name("conv1")),
    createConv("conv2", 3, 1, 256, "same", scoped_name("conv1_bn")),
    createBN(true, scoped_name("conv2")),
    createConv("conv3", 3, 1, 256, "same", scoped_name("conv2_bn")),
    createBN(true, scoped_name("conv3")),
    createConv("conv4", 3, 1, 256, "same", scoped_name("conv3_bn")),
    createBN(true, scoped_name("conv4")),
    createConv("ploc", 3, 1, 4 * num_anchors, "same", scoped_name("conv4_bn")),
    createBN(false, scoped_name("ploc")),
    createConv("pconf", 3, 1, num_classes * num_anchors, "same", scoped_name("conv4_bn")),
    createBN(false, scoped_name("pconf")),
  };
}

/**
 * @brief Create RefineDet
 *
 * @return vector of layers that contain full graph of RefineDet
 */
std::vector<LayerHandle> createRefineDetGraph(const unsigned int& batch_size) {
  using ml::train::createLayer;

  std::vector<LayerHandle> layers;
  layers.push_back(createLayer(
    "input", {withKey("name", "backbone_output2"), withKey("input_shape", "14:14:512")}));
  layers.push_back(createLayer(
    "input", {withKey("name", "backbone_output1"), withKey("input_shape", "28:28:512")}));
    
  std::vector<LayerHandle> feature_extractor = featureExtractor("backbone_output2");
  layers.insert(layers.end(), feature_extractor.begin(), feature_extractor.end());

  std::vector<std::vector<LayerHandle>> armBlocks;
  armBlocks.push_back(ARM("arm1", "backbone_output1", true));
  armBlocks.push_back(ARM("arm2", "backbone_output2", true));
  armBlocks.push_back(ARM("arm3", "feature_extractor/conv8_2", false)); 
  armBlocks.push_back(ARM("arm4", "feature_extractor/conv10_2", false));

  for (auto &block : armBlocks) {
    layers.insert(layers.end(), block.begin(), block.end());
  }  

  std::vector<std::vector<LayerHandle>> tcbBlocks;
  tcbBlocks.push_back(tcbBlock("tcb4", "feature_extractor/conv10_2", "", false, false));
  tcbBlocks.push_back(tcbBlock("tcb3", "feature_extractor/conv8_2", "tcb4/conv3_bn", false, false));
  tcbBlocks.push_back(tcbBlock("tcb2", "backbone_output2", "tcb3/conv3_bn", true, true));
  tcbBlocks.push_back(tcbBlock("tcb1", "backbone_output1", "tcb2/conv3_bn", true, true));

  for (auto &block : tcbBlocks) {
    layers.insert(layers.end(), block.begin(), block.end());
  }  

  std::vector<std::vector<LayerHandle>> odmBlocks;
  odmBlocks.push_back(ODM("odm1", "tcb1/conv3_bn"));
  odmBlocks.push_back(ODM("odm2", "tcb2/conv3_bn"));
  odmBlocks.push_back(ODM("odm3", "tcb3/conv3_bn"));
  odmBlocks.push_back(ODM("odm4", "tcb4/conv3_bn"));
  
  for (auto &block : odmBlocks) {
    layers.insert(layers.end(), block.begin(), block.end());
  }

  auto permute_output = [&](const std::string& block_name, 
    const std::string& input_name) {
    return ml::train::createLayer(
      "permute", {
        withKey("name", block_name),
        withKey("input_layers", {input_name}),
        withKey("direction", {2,3,1})
      }
    );
  };

  layers.push_back(permute_output("arm1_pconf_", "arm1/pconf_bn"));
  layers.push_back(permute_output("arm2_pconf_", "arm2/pconf_bn"));
  layers.push_back(permute_output("arm3_pconf_", "arm3/pconf_bn"));
  layers.push_back(permute_output("arm4_pconf_", "arm4/pconf_bn"));

  layers.push_back(permute_output("arm1_ploc_", "arm1/ploc_bn"));
  layers.push_back(permute_output("arm2_ploc_", "arm2/ploc_bn"));
  layers.push_back(permute_output("arm3_ploc_", "arm3/ploc_bn"));
  layers.push_back(permute_output("arm4_ploc_", "arm4/ploc_bn"));

  layers.push_back(permute_output("odm1_pconf_", "odm1/pconf_bn"));
  layers.push_back(permute_output("odm2_pconf_", "odm2/pconf_bn"));
  layers.push_back(permute_output("odm3_pconf_", "odm3/pconf_bn"));
  layers.push_back(permute_output("odm4_pconf_", "odm4/pconf_bn"));

  layers.push_back(permute_output("odm1_ploc_", "odm1/ploc_bn"));
  layers.push_back(permute_output("odm2_ploc_", "odm2/ploc_bn"));
  layers.push_back(permute_output("odm3_ploc_", "odm3/ploc_bn"));
  layers.push_back(permute_output("odm4_ploc_", "odm4/ploc_bn"));

  auto reshape_output = [&](const std::string& block_name, 
    const std::string& input_name, 
    const unsigned int& batch_size,
    const unsigned int& feature_map_size,
    const unsigned int& shape) {
    using ml::train::createLayer;
 
    return createLayer(
      "reshape", {
        withKey("name", block_name),
        withKey("input_layers", {input_name}),
        withKey("target_shape", std::to_string(num_anchors * feature_map_size * feature_map_size) + ":" +
                  std::to_string(shape))
      }
    );
  };

  layers.push_back(reshape_output("arm1_pconf", "arm1_pconf_", batch_size, feature_map_size1, 2));
  layers.push_back(reshape_output("arm2_pconf", "arm2_pconf_", batch_size, feature_map_size2, 2));
  layers.push_back(reshape_output("arm3_pconf", "arm3_pconf_", batch_size, feature_map_size3, 2));
  layers.push_back(reshape_output("arm4_pconf", "arm4_pconf_", batch_size, feature_map_size4, 2));

  layers.push_back(reshape_output("arm1_ploc", "arm1_ploc_", batch_size, feature_map_size1, 4));
  layers.push_back(reshape_output("arm2_ploc", "arm2_ploc_", batch_size, feature_map_size2, 4));
  layers.push_back(reshape_output("arm3_ploc", "arm3_ploc_", batch_size, feature_map_size3, 4));
  layers.push_back(reshape_output("arm4_ploc", "arm4_ploc_", batch_size, feature_map_size4, 4));

  layers.push_back(reshape_output("odm1_pconf", "odm1_pconf_", batch_size, feature_map_size1, num_classes));
  layers.push_back(reshape_output("odm2_pconf", "odm2_pconf_", batch_size, feature_map_size2, num_classes));
  layers.push_back(reshape_output("odm3_pconf", "odm3_pconf_", batch_size, feature_map_size3, num_classes));
  layers.push_back(reshape_output("odm4_pconf", "odm4_pconf_", batch_size, feature_map_size4, num_classes));

  layers.push_back(reshape_output("odm1_ploc", "odm1_ploc_", batch_size, feature_map_size1, 4));
  layers.push_back(reshape_output("odm2_ploc", "odm2_ploc_", batch_size, feature_map_size2, 4));
  layers.push_back(reshape_output("odm3_ploc", "odm3_ploc_", batch_size, feature_map_size3, 4));
  layers.push_back(reshape_output("odm4_ploc", "odm4_ploc_", batch_size, feature_map_size4, 4));

  auto createConcat = [&](const std::string& module, const std::string& predict) {
    return createLayer("concat", {
        withKey("name", module + "_" + predict), 
        withKey("input_layers", {
          module + "1_" + predict, 
          module + "2_" + predict, 
          module + "3_" + predict, 
          module + "4_" + predict}),
        withKey("axis", 2)
      });
  };
  layers.push_back(createConcat("arm", "ploc"));
  layers.push_back(createConcat("arm", "pconf"));
  layers.push_back(createConcat("odm", "ploc"));
  layers.push_back(createConcat("odm", "pconf"));

  layers.push_back(createLayer("concat", {
    withKey("name", "output"), 
    withKey("input_layers", {"arm_ploc", "arm_pconf", "odm_ploc", "odm_pconf"}),
    withKey("axis", 3)
  }));

  layers.push_back(createLayer("refinedet_loss", {withKey("name", "refinedet_loss"),
    withKey("input_layers", {"output"})}));

  return layers;
}

ModelHandle createRefineDet(const unsigned int& batch_size) {
  ModelHandle model = ml::train::createModel(ml::train::ModelType::NEURAL_NET, {});
  for (auto &layer : createRefineDetGraph(batch_size)) {
    model->addLayer(layer);
  }
  return model;
}

int trainData_cb(float **input, float **label, bool *last, void *user_data) {
  auto data = reinterpret_cast<nntrainer::util::DirDataLoader *>(user_data);
  data->next(input, label, last);
  return 0;
}

int validData_cb(float **input, float **label, bool *last, void *user_data) {
  auto data = reinterpret_cast<nntrainer::util::DirDataLoader *>(user_data);

  data->next(input, label, last);
  return 0;
}

void createAndRun(unsigned int epochs, unsigned int batch_size,
                  UserDataType &train_user_data,
                  UserDataType &valid_user_data) {
  ModelHandle model = createRefineDet(batch_size);
  model->setProperty({withKey("batch_size", batch_size),
                      withKey("epochs", epochs),
                      withKey("save_path", "refinedet_full.bin")});

  auto optimizer = ml::train::createOptimizer("adam", {"learning_rate=0.001"});
  model->setOptimizer(std::move(optimizer));

  int status = model->compile();
  if (status) {
    throw std::invalid_argument("model compilation failed!");
  }

  status = model->initialize();
  if (status) {
    throw std::invalid_argument("model initialization failed!");
  }

  auto dataset_train = ml::train::createDataset(
    ml::train::DatasetType::GENERATOR, trainData_cb, train_user_data.get());
  auto dataset_valid = ml::train::createDataset(
    ml::train::DatasetType::GENERATOR, validData_cb, valid_user_data.get());

  model->setDataset(ml::train::DatasetModeType::MODE_TRAIN,
                    std::move(dataset_train));
  model->setDataset(ml::train::DatasetModeType::MODE_VALID,
                    std::move(dataset_valid));
  model->train();
#if defined(ENABLE_TEST)
  training_loss = model->getTrainingLoss();
  validation_loss = model->getValidationLoss();
#endif
}

std::array<UserDataType, 2> createDetDataGenerator(const char *train_dir,
                                                   const char *valid_dir,
                                                   int max_num_label, int c,
                                                   int h, int w) {
  UserDataType train_data(new nntrainer::util::DirDataLoader(
    train_dir, max_num_label, c, h, w, true));
  UserDataType valid_data(new nntrainer::util::DirDataLoader(
    valid_dir, max_num_label, c, h, w, false));

  return {std::move(train_data), std::move(valid_data)};
}

int main(int argc, char *argv[]) {
  if (argc < 5) {
    std::cerr
      << "usage: ./main [{data_directory}|\"fake\"] [batchsize] [data_split] "
         "[epoch] \n"
      << "when \"fake\" is given, original data size is assumed 512 for both "
         "train and validation\n";
    return EXIT_FAILURE;
  }

  try {
    auto &app_context = nntrainer::AppContext::Global();
    app_context.registerFactory(nntrainer::createLayer<custom::RefineDetLoss>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
    return 1;
  }


  auto start = std::chrono::system_clock::now();
  std::time_t start_time = std::chrono::system_clock::to_time_t(start);
  std::cout << "started computation at " << std::ctime(&start_time)
            << std::endl;

  std::string data_dir = argv[1];
  unsigned int batch_size = std::stoul(argv[2]);
  unsigned int data_split = std::stoul(argv[3]);
  unsigned int epoch = std::stoul(argv[4]);

  std::cout << "data_dir: " << data_dir << ' ' << "batch_size: " << batch_size
            << " data_split: " << data_split << " epoch: " << epoch
            << std::endl;

  std::array<UserDataType, 2> user_datas;

  try {
    const char *train_dir = "../Applications/RefineDet/train-small/";
    const char *valid_dir = "../Applications/RefineDet/test-small/";
    const int max_num_label = 5;
    const int channel = 3;
    const int width = 224;
    const int height = 224;
    user_datas = createDetDataGenerator(train_dir, valid_dir, max_num_label,
                                        channel, width, height);
  } catch (const std::exception &e) {
    std::cerr << "uncaught error while creating data generator! details: "
              << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  auto &[train_user_data, valid_user_data] = user_datas;

  try {
    createAndRun(epoch, batch_size, train_user_data, valid_user_data);
  } catch (const std::exception &e) {
    std::cerr << "uncaught error while running! details: " << e.what()
              << std::endl;
    return EXIT_FAILURE;
  }
  auto end = std::chrono::system_clock::now();

  std::chrono::duration<double> elapsed_seconds = end - start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);

  std::cout << "finished computation at " << std::ctime(&end_time)
            << "elapsed time: " << elapsed_seconds.count() << "s\n";

  int status = EXIT_SUCCESS;
#if defined(ENABLE_TEST)
  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error during InitGoogleTest" << std::endl;
    return EXIT_FAILURE;
  }

  try {
    status = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error during RUN_ALL_TESTS()" << std::endl;
  }
#endif

  return status;
}
