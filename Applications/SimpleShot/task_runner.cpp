// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   task_runner.cpp
 * @date   08 Jan 2021
 * @brief  task runner for adding new classes at runtime for simpleshot learning
 * demonstration
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @author Priyank Mohan Verma <p1.verma@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unistd.h>

#include <cstdio>
#include <iterator>
#include <map>
#include <stdio.h>
#include <stdlib.h>

#include <app_context.h>
#include <model.h>
#include <nntrainer-api-common.h>

#include "layers/centering.h"

namespace simpleshot {

namespace {

/**
 * @brief get backbone path from a model name
 *
 * @param model resnet50 or conv4 or facenet is supported
 *
 */
const std::string getModelFilePath(const std::string &model,
                                   const std::string &app_path) {
  const std::string resnet_model_path =
    app_path + "/backbones/resnet50_60classes.tflite";
  const std::string conv4_model_path =
    app_path + "/backbones/conv4_60classes.tflite";
  const std::string facenet_model_path = app_path + "/backbones/facenet.tflite";

  std::string model_path;

  if (model == "resnet50") {
    model_path = resnet_model_path;
  } else if (model == "conv4") {
    model_path = conv4_model_path;
  } else if (model == "facenet") {
    model_path = facenet_model_path;
  }

  std::ifstream infile(model_path);
  if (!infile.good()) {
    std::stringstream ss;
    ss << model_path << " as backbone does not exist!";
    throw std::invalid_argument(ss.str().c_str());
  }

  if (model_path.empty()) {
    std::stringstream ss;
    ss << "not supported model type given, model type: " << model;
    throw std::invalid_argument(ss.str().c_str());
  }

  return model_path;
}

const std::string getFeatureFilePath(const std::string &model,
                                     const std::string &app_path) {
  const std::string resnet_model_path =
    app_path + "/backbones/resnet50_60classes_feature_vector.bin";
  const std::string conv4_model_path =
    app_path + "/backbones/conv4_60classes_feature_vector.bin";

  std::string model_path;

  if (model == "resnet50") {
    model_path = resnet_model_path;
  } else if (model == "conv4") {
    model_path = conv4_model_path;
  }

  std::ifstream infile(model_path);
  if (!infile.good()) {
    std::stringstream ss;
    ss << model_path << " as backbone does not exist!";
    throw std::invalid_argument(ss.str().c_str());
  }

  if (model_path.empty()) {
    std::stringstream ss;
    ss << "not supported model type given, model type: " << model;
    throw std::invalid_argument(ss.str().c_str());
  }

  return model_path;
}

/**
 * @brief get current working directory by cpp string
 *
 * @return const std::string current working directory
 */
const std::string getcwd_() {
  const size_t bufsize = 4096;
  char buffer[bufsize];

  return getcwd(buffer, bufsize);
}
} // namespace

using LayerHandle = std::shared_ptr<ml::train::Layer>;

/**
 * @brief find model name by using strcmp
 * @param backbone name
 * @return std::pair<std::string,std::string> to identify the model name and
 * input_shape
 */

std::pair<std::string, std::string> findModelName(std::string backbone) {

  const char *backbone_name = backbone.c_str();

  std::pair<std::string, std::string> model_parameter;

  std::string model_name = "name=" + backbone;

  if (std::strcmp(backbone_name, "conv4") == 0 ||
      std::strcmp(backbone_name, "resnet50") == 0) {
    model_parameter = {model_name, "input_shape=32:32:3"};
  } else if (std::strcmp(backbone_name, "facenet") == 0) {
    model_parameter = {model_name, "input_shape=112:112:3"};
  } else {
    model_parameter = {"-1", "-1"};
  }

  return model_parameter;
}

int new_classes;     /**< an integer value new classes to be added */
int earlier_classes; /**< an integer value earlier classes which were present*/

/**
 * @brief Create a Model with given backbone and varient setup
 * selected
 * @param backbone either conv4 or resnet50, hardcoded tflite path will be
 * selected
 * @param app_path designated app path to search the backbone file
 * @param variant "one of UN, L2N, CL2N"
 * @param num_classes new_classes which are added
 * @return std::unique_ptr<ml::train::Model>
 */

std::unique_ptr<ml::train::Model>
createModel(const std::string &backbone, const std::string &app_path,
            const std::string &variant = "UN",
            const int num_classes = new_classes) {
  auto model = ml::train::createModel(ml::train::ModelType::NEURAL_NET,
                                      {"batch_size=1", "epochs=1"});

  /// Find backbone model name to be created

  std::pair<std::string, std::string> model_parameter = findModelName(backbone);
  if (strcmp(model_parameter.first.c_str(), "-1") == 0) {
    std::stringstream ss;
    ss << "not supported model name given, model type: " << backbone;
    throw std::invalid_argument(ss.str().c_str());
  }

  LayerHandle backbone_layer = ml::train::layer::BackboneTFLite(
    {model_parameter.first,
     "model_path=" + getModelFilePath(backbone, app_path),
     model_parameter.second, "trainable=false"});

  /// Creating Model Layers

  model->addLayer(backbone_layer);

  auto generate_knn_part = [&backbone, &app_path,
                            num_classes](const std::string &variant_) {
    std::vector<LayerHandle> v;

    const std::string num_class_prop =
      "num_class=" + std::to_string(num_classes);

    if (variant_ == "UN") {
      /// left empty intended
    } else if (variant_ == "L2N") {
      LayerHandle l2 = ml::train::createLayer(
        "preprocess_l2norm", {"name=l2norm", "trainable=false"});
      v.push_back(l2);
    } else if (variant_ == "CL2N") {
      LayerHandle centering = ml::train::createLayer(
        "centering", {"name=center",
                      "feature_path=" + getFeatureFilePath(backbone, app_path),
                      "trainable=false"});
      LayerHandle l2 =
        ml::train::createLayer("l2norm", {"name=l2norm", "trainable=false"});
      v.push_back(centering);
      v.push_back(l2);
    } else {
      std::stringstream ss;
      ss << "unsupported variant type: " << variant_;
      throw std::invalid_argument(ss.str().c_str());
    }

    LayerHandle knn = ml::train::createLayer(
      "centroid_knn", {"name=knn", num_class_prop, "trainable=false"});
    v.push_back(knn);

    return v;
  };

  auto knn_part = generate_knn_part(variant);
  for (auto &layer : knn_part) {
    model->addLayer(layer);
  }

  return model;
}
} // namespace simpleshot

/**
 * @brief Check whether file exist or not
 * @param file_name name of file
 * @return bool true if exists else false if it doesn't exists
 */

bool isFileExist(std::string file_name) {
  std::ifstream infile(file_name);
  return infile.good();
}

/**
 * @brief appends label in allLabels.txt file that maintains all added classes
 * string labels
 * @param label_path string label path of newly added classes
 * @return int 1 for new classes appended in allLabels.txt and 0 if no new
 * classes are added to allLabels.txt
 */

int appendLabel(std::string label_path) {

  int class_cnt = 0;
  int flag = 0;
  std::map<int, std::string> label_map;

  if (std::strcmp(label_path.c_str(), "null") != 0) {

    std::fstream label_file;
    label_file.open(label_path, std::ios::in);

    std::cout << "Name of new classes which is being added:\n";

    if (label_file.is_open()) { /// checking whether the file is open
      std::string tp;

      while (std::getline(label_file, tp)) {
        /// read data from file object and put it into string
        /// print the data of the string
        std::cout << tp << "\n";

        label_map.insert(std::pair<int, std::string>(class_cnt, tp));
        class_cnt++;
      }
      label_file.close();
    }

    simpleshot::new_classes = class_cnt;

    std::cout << "\n Number of new classes added : " << class_cnt << "\n \n";

  }

  else {
    flag = 1;
  }

  /// Read Create the allLabels.txt

  std::string all_class_label_path = "allLabels.txt";

  std::fstream all_labels_file;

  if (isFileExist(all_class_label_path) == false) {

    simpleshot::earlier_classes = 0;

    all_labels_file.open(all_class_label_path, std::ios::out);

    /// checking whether the file is open

    if (all_labels_file.is_open()) {

      for (auto itr = label_map.begin(); itr != label_map.end(); itr++) {
        all_labels_file << itr->second;
        all_labels_file << "\n";
      }

      /// close the file object

      all_labels_file.close();
    }

  }

  else {
    std::map<int, std::string> earlier_label_map;

    all_labels_file.open(all_class_label_path, std::ios::in);
    std::cout << "Name of Earlier Classes which were present: \n";
    class_cnt = 0;

    if (all_labels_file.is_open()) {

      // checking whether the file is open
      std::string tp;

      while (std::getline(all_labels_file, tp)) {
        /// read data from file object and put it into string
        /// print the data of the string
        std::cout << tp << "\n";
        earlier_label_map.insert(std::pair<int, std::string>(class_cnt, tp));
        class_cnt++;
      }
      all_labels_file.close();
    }

    simpleshot::earlier_classes = class_cnt;

    std::cout << "\n Number of earlier classes which were present : "
              << class_cnt << "\n \n";

    int x = std::remove(all_class_label_path.c_str());
    if (x != 0)
      std::cout << "Appending Failed\n";

    all_labels_file.open(all_class_label_path, std::ios::out);

    /// checking whether the file is open

    if (all_labels_file.is_open()) {

      for (auto itr = earlier_label_map.begin(); itr != earlier_label_map.end();
           itr++) {
        all_labels_file << itr->second;
        all_labels_file << "\n";
      }

      if (flag == 0) {
        for (auto itr = label_map.begin(); itr != label_map.end(); itr++) {

          all_labels_file << itr->second;
          all_labels_file << "\n";
        }
      }

      /// close the file object

      all_labels_file.close();
    }
  }

  std::cout << "Successfully appended the labels in allLabels.txt file\n";

  return flag;
}

/**
 * @brief main runner
 *
 * @return int
 */

int main(int argc, char **argv) {
  auto &app_context = nntrainer::AppContext::Global();

  if (argc != 7 && argc != 6) {
    std::cout
      << "usage: model method train_file validation_file app_path\n"
      << "model: are [resnet50, conv4]\n"
      << "methods: are [UN, L2N, CL2N]\n"
      << "train file: [app_path]/tasks/[train_file] is used for training\n"
      << "validation file: [app_path]/tasks/[validation_file] is used for "
         "validation\n"
      << "app_path: root path to refer to resources, if not given"
         "path is set current working directory\n";
    return 1;
  }

  for (int i = 0; i < argc; ++i) {
    if (argv[i] == nullptr) {
      std::cout
        << "usage: model method train_file_path validation_file_path app_path\n"
        << "Supported model types are [resnet50, conv4]\n"
        << "Supported methods are [UN, L2N, CL2N]\n"
        << "train file: [app_path]/tasks/[train_file] is used for training\n"
        << "validation file: [app_path]/tasks/[validation_file] is used for "
           "validation\n"
        << "app_path: root path to refer to resources, if not given"
           "path is set current working directory\n";
      return 1;
    }
  }

  std::string model_str(argv[1]);
  std::string app_path =
    argc == 7 ? std::string(argv[6]) : simpleshot::getcwd_();

  std::string method = argv[2];
  std::string train_path = app_path + "/tasks/" + argv[3];
  std::string val_path = app_path + "/tasks/" + argv[4];
  std::string label_path;

  if (std::strcmp(std::string(argv[5]).c_str(), "null") == 0) {
    label_path = argv[5];
  }

  else {
    label_path = app_path + "/tasks/" + argv[5];
  }

  /// append_label to allLabels.txt

  int flag = appendLabel(label_path);

  /// if new classes is zero so to avoid error
  if (flag == 1) {
    simpleshot::new_classes = simpleshot::earlier_classes;
  }

  try {
    app_context.registerFactory(
      nntrainer::createLayer<simpleshot::layers::CenteringLayer>);
  } catch (std::exception &e) {
    std::cerr << "registering factory failed: " << e.what();
    return 1;
  }

  std::unique_ptr<ml::train::Model> model;
  try {
    model = simpleshot::createModel(model_str, app_path, method);
    model->summarize(std::cout, ML_TRAIN_SUMMARY_MODEL);
  } catch (std::exception &e) {
    std::cerr << "creating Model failed: " << e.what();
    return 1;
  }

  std::shared_ptr<ml::train::Dataset> train_dataset, valid_dataset;
  try {
    train_dataset = ml::train::createDataset(ml::train::DatasetType::FILE,
                                             train_path.c_str());
    valid_dataset =
      ml::train::createDataset(ml::train::DatasetType::FILE, val_path.c_str());

  } catch (...) {
    std::cerr << "creating dataset failed";
    return 1;
  }

  if (model->setDataset(ml::train::DatasetModeType::MODE_TRAIN,
                        train_dataset)) {
    std::cerr << "failed to set train dataset" << std::endl;
    return 1;
  };

  if (model->setDataset(ml::train::DatasetModeType::MODE_VALID,
                        valid_dataset)) {
    std::cerr << "failed to set valid dataset" << std::endl;
    return 1;
  };

  std::shared_ptr<ml::train::Optimizer> optimizer;
  try {
    optimizer = ml::train::optimizer::SGD({"learning_rate=0.1"});
  } catch (...) {
    std::cerr << "creating optimizer failed";
    return 1;
  }

  if (model->setOptimizer(optimizer) != 0) {
    std::cerr << "failed to set optimizer" << std::endl;
    return 1;
  }

  if (model->compile() != 0) {
    std::cerr << "model compilation failed" << std::endl;
    return 1;
  }

  if (model->initialize() != 0) {
    std::cerr << "model initiation failed" << std::endl;
    return 1;
  }

  /// Getting list of Centroids and append the new classes and getting centroids
  /// of classes from getCentroids function and saving in tensor.bin

  auto save_path_centroid = "tensor.bin";
  std::string save_path_tensor = "tensor.bin";

  /// Check for Tensor.bin

  if (isFileExist(save_path_centroid) == false) {
    nntrainer::Tensor centroid_tensor = model->getCentroids();
    std::ofstream file_new(save_path_tensor, std::ios::out | std::ios::binary);
    centroid_tensor.save(file_new);
    file_new.close();
    std::cout << "Centroids Saved in tensor.bin\n";

  }

  else if (flag == 0) {
    nntrainer::Tensor new_centroid_tensor = model->getCentroids();
    nntrainer::Tensor earlier_centroid_tensor(1, 1, simpleshot::earlier_classes,
                                              192);

    std::ifstream file_read(save_path_tensor, std::ios::in | std::ios::binary);

    earlier_centroid_tensor.read(file_read);

    std::cout << "Reading Centroid Tensor of earlier classes from tensor.bin\n";

    nntrainer::Tensor all_class_centroid_tensor(
      1, 1, simpleshot::earlier_classes + simpleshot::new_classes, 192);

    /// copying old centroid

    const float *data_old = earlier_centroid_tensor.getData();

    for (int i = 0; i < 192 * simpleshot::earlier_classes; i++) {
      all_class_centroid_tensor.setValue(0, 0, i / 192, i % 192, data_old[i]);
    }

    /// copying new centroid

    const float *data_new = new_centroid_tensor.getData();

    for (int i = 0; i < 192 * simpleshot::new_classes; i++) {
      all_class_centroid_tensor.setValue(
        0, 0, simpleshot::earlier_classes + i / 192, i % 192, data_new[i]);
    }

    std::ofstream file_new(save_path_tensor, std::ios::out | std::ios::binary);
    all_class_centroid_tensor.save(file_new);
    file_new.close();

    std::cout << "All class new and old Centroids Saved in tensor.bin\n";
  }

  /// predict the class of Query images

  std::cout << "\n \n Predicting the Class of Queries\n";

  if (flag == 1)
    simpleshot::new_classes = 0;

  model->predict(simpleshot::earlier_classes,
                 simpleshot::earlier_classes + simpleshot::new_classes,
                 label_path);

  std::cout << "successfully ran" << std::endl;
  return 0;
}
