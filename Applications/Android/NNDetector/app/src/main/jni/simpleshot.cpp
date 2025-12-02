// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright 2023. Umberto Michieli <u.michieli@samsung.com>.
 * Copyright 2023. Kirill Paramonov <k.paramonov@samsung.com>.
 * Copyright 2023. Mete Ozay <m.ozay@samsung.com>.
 * Copyright 2023. JIJOONG MOON <jijoong.moon@samsung.com>.
 *
 * @file   simpleshot.cpp
 * @date   24 Oct 2023
 * @brief  image recognition/detection modules
 * @author Umberto Michieli(u.michieli@samsung.com)
 * @author Kirill Paramonov(k.paramonov@samsung.com)
 * @author Mete Ozay(m.ozay@samsung.com)
 * @author JIJOONG MOON(jijoong.moon@samsung.com)
 * @bug    No known bugs
 */
#include "simpleshot.h"

#include <sstream>

std::unique_ptr<ml::train::Model> rec_model;
std::unique_ptr<ml::train::Model> det_model;
std::string test_result = "";
std::string run_det_result = "";
std::vector<std::string> train_class_names;
std::string unknown_label = "unknown";

std::vector<float> rec_norm_ch_mean{0.472140644, 0.453308291, 0.409961280};
std::vector<float> rec_norm_ch_std{0.277183853, 0.267750409, 0.284490412};

// Hard-coded class indices from coco128
// 16 - dog class
// std::vector<unsigned int> class_indices_of_interest {16};
std::map<int, std::string> macro_class_names{
  {0, "pen"},   {1, "mug"},   {2, "bottle"},   {3, "book"},  {4, "glasses"},
  {5, "watch"}, {6, "mouse"}, {7, "keyboard"}, {8, "fruit"}, {9, "snack"}};

// Confidence level to accept detected bounding boxes in yolo.
float train_det_score_threshold = 0.3;
float test_det_score_threshold = 0.3;
float run_det_score_threshold = 0.2;

// Max number of yolo boxes to accept.
int train_det_max_bb_num = 1;
int test_det_max_bb_num = 10;
int run_det_max_bb_num = 5;

// IOU threshold to filter detection boxes in non-maximum supression.
float det_iou_threshold = 0.4;

// If the distance from the new sample to the learned prototypes
// is greater than this value, we assign 'unknown' class to it.
float unknown_class_distance_threshold = 1000;

bool use_detection_for_train_prototypes = false;
bool use_detection_for_test_prototypes = false;

std::vector<std::string> split(std::string input, char delimiter) {
  std::vector<std::string> answer;
  std::stringstream ss(input);
  std::string temp;
  while (getline(ss, temp, delimiter)) {
    answer.push_back(temp);
  }
  return answer;
}

int trainData_cb(float **input, float **label, bool *last, void *user_data) {
  auto data =
    reinterpret_cast<nntrainer::simpleshot::DirDataLoader *>(user_data);
  data->next(input, label, last);
  return 0;
}

ml::train::Model *initialize_det(int argc, char *argv[]) {

  if (argc < 3) {
    std::cerr << "usage: ./main [det backbone path] [det input image dim]\n";
    throw std::invalid_argument("wrong argument");
  }

  std::string det_backbone_path = argv[1];
  unsigned int det_input_image_dim = std::stoul(argv[2]);

  ANDROID_LOG_D("---detection backbone_path: %s", det_backbone_path.c_str());
  ANDROID_LOG_D("---input shape: %d:%d:3", det_input_image_dim,
                det_input_image_dim);

  std::string in_str = std::to_string(det_input_image_dim);

  det_model = ml::train::createModel(ml::train::ModelType::NEURAL_NET,
                                     {"batch_size=1", "epochs=1"});

  LayerHandle backbone_layer = ml::train::layer::BackboneTFLite(
    {"name=backbone", "model_path=" + det_backbone_path,
     "input_shape=" + in_str + ":" + in_str + ":3", "trainable=false"});
  det_model->addLayer(backbone_layer);

  // det_model->summarize(std::cout, ML_TRAIN_SUMMARY_MODEL);

  std::shared_ptr<ml::train::Optimizer> optimizer;
  optimizer = ml::train::optimizer::SGD({"learning_rate=0.1"});

  if (det_model->setOptimizer(optimizer)) {
    throw std::invalid_argument("failed to set optimizer");
  }
  if (det_model->compile()) {
    throw std::invalid_argument("model compilation failed");
  }
  if (det_model->initialize()) {
    throw std::invalid_argument("model initiation failed");
  }

  return det_model.get();
}

ml::train::Model *initialize_rec(int argc, char *argv[]) {

  if (argc < 5) {
    std::cerr << "usage: ./main [rec backbone path] "
                 "[rec input image dim] [num class] [knn norm variant]\n";
    throw std::invalid_argument("wrong argument");
  }

  std::string rec_backbone_path = argv[1];
  unsigned int rec_input_img_dim = std::stoul(argv[2]);
  unsigned int num_class = std::stoul(argv[3]);
  std::string knn_norm_variant = argv[4];

  ANDROID_LOG_D("---recognition backbone path: %s", rec_backbone_path.c_str());
  ANDROID_LOG_D("---num classes: %d", num_class);
  ANDROID_LOG_D("---input shape: %d:%d:3", rec_input_img_dim,
                rec_input_img_dim);
  ANDROID_LOG_D("---knn norm variant: %s", knn_norm_variant.c_str());

  rec_model = ml::train::createModel(ml::train::ModelType::NEURAL_NET,
                                     {"batch_size=1", "epochs=1"});

  LayerHandle backbone_layer = ml::train::layer::BackboneTFLite(
    {"name=backbone", "model_path=" + rec_backbone_path,
     "input_shape=" + std::to_string(rec_input_img_dim) + ":" +
       std::to_string(rec_input_img_dim) + ":3",
     "trainable=false"});
  rec_model->addLayer(backbone_layer);

  auto generate_knn_part = [num_class](const std::string &variant_) {
    std::vector<LayerHandle> v;

    const std::string num_class_prop = "num_class=" + std::to_string(num_class);

    if (variant_ == "UN") {
      /// left empty intended
    } else if (variant_ == "L2N") {
      LayerHandle l2 = ml::train::createLayer(
        "preprocess_l2norm", {"name=l2norm", "trainable=false"});
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

  auto knn_part = generate_knn_part(knn_norm_variant);
  for (auto &layer : knn_part) {
    rec_model->addLayer(layer);
  }
  // rec_model->summarize(std::cout, ML_TRAIN_SUMMARY_MODEL);

  std::shared_ptr<ml::train::Optimizer> optimizer;
  optimizer = ml::train::optimizer::SGD({"learning_rate=0.1"});

  if (rec_model->setOptimizer(optimizer)) {
    throw std::invalid_argument("failed to set optimizer");
  }
  if (rec_model->compile()) {
    throw std::invalid_argument("model compilation failed");
  }
  if (rec_model->initialize()) {
    throw std::invalid_argument("model initiation failed");
  }

  return rec_model.get();
}

void train_prototypes(int argc, char *argv[], ml::train::Model *det_model_,
                      ml::train::Model *rec_model_) {

  if (argc < 7) {
    std::cerr << "usage: ./main [train data path] [det input image dim] [det "
                 "output dim]"
                 "[det anchor num] [rec input image dim] [num class]\n";
    throw std::invalid_argument("wrong argument");
  }

  std::string train_data_path = argv[1];
  unsigned int det_input_img_dim = std::stoul(argv[2]);
  unsigned int det_output_dim = std::stoul(argv[3]);
  unsigned int det_anchor_num = std::stoul(argv[4]);
  unsigned int rec_input_img_dim = std::stoul(argv[5]);
  unsigned int num_class = std::stoul(argv[6]);

  ANDROID_LOG_D("---train_data_path: %s", train_data_path.c_str());
  ANDROID_LOG_D("---num_classes: %d", num_class);
  ANDROID_LOG_D("---recognition model input shape: %d:%d:3", rec_input_img_dim,
                rec_input_img_dim);

  UserDataType train_user_data(new nntrainer::simpleshot::DirDataLoader(
    train_data_path.c_str(), num_class, rec_input_img_dim, rec_norm_ch_mean,
    rec_norm_ch_std));

  if (use_detection_for_train_prototypes) {
    std::vector<unsigned int> class_indices_of_interest;
    for (unsigned int i = 0; i < det_output_dim - 4; i++) {
      class_indices_of_interest.push_back(i);
    }

    ANDROID_LOG_D("Running detector");
    train_user_data->runDetector(det_model_, det_input_img_dim, det_output_dim,
                                 det_anchor_num, class_indices_of_interest,
                                 train_det_score_threshold, det_iou_threshold,
                                 train_det_max_bb_num);
    ANDROID_LOG_D("Detection finished");

    std::vector<nntrainer::simpleshot::boundingBoxInfo> bb_info_list =
      train_user_data->getBBList();
    for (auto &bb_info : bb_info_list) {
      std::string file_name = bb_info.file_name;
      std::vector<int> bounding_box = bb_info.bounding_box;
      float det_score = bb_info.detection_score;
      ANDROID_LOG_D("Processing file: %s", file_name.c_str());

      float normed_xy = (float)(bounding_box[0]) / (float)(det_input_img_dim);
      std::stringstream ss;
      ss << std::fixed << std::setprecision(2) << normed_xy;
      for (unsigned int i = 1; i < 4; i++) {
        normed_xy = (float)bounding_box[i] / (float)det_input_img_dim;
        ss << " " << normed_xy;
      }

      std::string bb_string = ss.str();
      std::vector<std::string> splited_file = split(file_name, '/');
      std::string result;
      result += (splited_file.rbegin()[0]) + "\n" +
                "     : predicted bb (xyxy) " + bb_string.c_str() + "\n" +
                "     : detection score " + std::to_string(det_score) + "\n\n";

      ANDROID_LOG_D("Processing file: %s", result.c_str());
    }
  }

  auto dataset_train = ml::train::createDataset(
    ml::train::DatasetType::GENERATOR, trainData_cb, train_user_data.get());

  if (rec_model_->setDataset(ml::train::DatasetModeType::MODE_TRAIN,
                             std::move(dataset_train))) {
    throw std::invalid_argument("failed to set train dataset");
  }

  ANDROID_LOG_D("Running recognition");
  if (rec_model_->train()) {
    throw std::invalid_argument("train failed");
  }
  ANDROID_LOG_D("Recognition finished");
  train_class_names = train_user_data->getClassNames();
}

std::string test_prototypes(int argc, char *argv[],
                            ml::train::Model *det_model_,
                            ml::train::Model *rec_model_) {

  if (argc < 7) {
    std::cerr
      << "usage: ./main [test data path] [det input image dim] [det output dim]"
         "[det anchor num] [rec input image dim] [num class]\n";
    throw std::invalid_argument("wrong argument");
  }

  test_result = "";
  std::string test_data_path = argv[1];
  unsigned int det_input_img_dim = std::stoul(argv[2]);
  unsigned int det_output_dim = std::stoul(argv[3]);
  unsigned int det_anchor_num = std::stoul(argv[4]);
  unsigned int rec_input_img_dim = std::stoul(argv[5]);
  unsigned int num_class = std::stoul(argv[6]);

  UserDataType test_user_data(new nntrainer::simpleshot::DirDataLoader(
    test_data_path.c_str(), num_class, rec_input_img_dim, rec_norm_ch_mean,
    rec_norm_ch_std));

  if (use_detection_for_test_prototypes) {
    std::vector<unsigned int> class_indices_of_interest;
    for (unsigned int i = 0; i < det_output_dim - 4; i++) {
      class_indices_of_interest.push_back(i);
    }

    ANDROID_LOG_D("Running detector");
    test_user_data->runDetector(det_model_, det_input_img_dim, det_output_dim,
                                det_anchor_num, class_indices_of_interest,
                                test_det_score_threshold, det_iou_threshold,
                                test_det_max_bb_num);
    ANDROID_LOG_D("Detection finished");
  }

  ANDROID_LOG_D("Running recognition");
  float *input = new float[rec_input_img_dim * rec_input_img_dim * 3]; // HWC
  float *label = new float[num_class];

  int right = 0;
  int count = 0;
  float accuracy;
  bool last = false;

  while (!last) {
    std::vector<float *> result;
    std::vector<float *> in;
    std::vector<float *> l;
    test_user_data->next(&input, &label, &last);
    in.push_back(input);
    l.push_back(label);
    result = rec_model_->inference(1, in, l);

    int result_ans =
      nntrainer::simpleshot::argmax(result[0], train_class_names.size());
    std::vector<std::string> splited_class =
      split(train_class_names[result_ans], '/');
    std::string class_label = splited_class.rbegin()[0];
    if (-result[0][result_ans] > unknown_class_distance_threshold) {
      class_label = unknown_label;
    }

    std::string file_name = test_user_data->getCurFileName();
    std::vector<std::string> splited_file = split(file_name, '/');
    std::string last_file_name = splited_file.rbegin()[0];

    ANDROID_LOG_D("Processing file: %s", file_name.c_str());

    std::string result_string = std::to_string(-result[0][0]);
    for (unsigned int i = 1; i < num_class; i++) {
      result_string += " " + std::to_string(-result[0][i]);
    }
    ANDROID_LOG_D("Distances to prototypes are: %s", result_string.c_str());
    ANDROID_LOG_D("Predicted label: %s", class_label.c_str());

    if (use_detection_for_test_prototypes) {

      nntrainer::simpleshot::boundingBoxInfo bb_info =
        test_user_data->getCurrBBInfo();
      std::vector<int> bounding_box = bb_info.bounding_box;
      float det_score = bb_info.detection_score;

      float normed_xy = (float)(bounding_box[0]) / (float)(det_input_img_dim);
      std::stringstream ss;
      ss << std::fixed << std::setprecision(2) << normed_xy;
      for (unsigned int i = 1; i < 4; i++) {
        normed_xy = (float)bounding_box[i] / (float)det_input_img_dim;
        ss << " " << normed_xy;
      }
      std::string bb_string = ss.str();

      test_result += " " + std::to_string(count) + " ] " + last_file_name +
                     "\n" + "     : predicted bb (xyxy) " + bb_string + "\n" +
                     "     : detection score " + std::to_string(det_score) +
                     "\n" + "     : predicted label " + class_label + "\n\n";
    } else {
      test_result += " " + std::to_string(count) + " ] " + last_file_name +
                     "\n" + "     : predicted label " + class_label + "\n\n";
    }

    in.clear();
    l.clear();

    count++;
  }
  ANDROID_LOG_D("Recognition finished");

  delete[] input;
  delete[] label;

  return test_result;
}

std::string run_detector(int argc, char *argv[], ml::train::Model *det_model_) {
  if (argc < 5) {
    std::cerr << "usage: ./main [detect data path] [det input image dim] [det "
                 "output dim]"
                 "[det anchor num]\n";
    throw std::invalid_argument("wrong argument");
  }

  std::string detect_data_path = argv[1];
  unsigned int det_input_img_dim = std::stoul(argv[2]);
  unsigned int det_output_dim = std::stoul(argv[3]);
  unsigned int det_anchor_num = std::stoul(argv[4]);

  UserDataType run_det_data(new nntrainer::simpleshot::DirDataLoader(
    detect_data_path.c_str(), 0, 0, rec_norm_ch_mean, rec_norm_ch_std));

  std::vector<unsigned int> class_indices_of_interest;
  for (unsigned int i = 0; i < det_output_dim - 4; i++) {
    class_indices_of_interest.push_back(i);
  }

  run_det_data->runDetector(det_model_, det_input_img_dim, det_output_dim,
                            det_anchor_num, class_indices_of_interest,
                            run_det_score_threshold, det_iou_threshold,
                            run_det_max_bb_num);

  std::vector<nntrainer::simpleshot::boundingBoxInfo> bb_list =
    run_det_data->getBBList();
  // run_det_result = " Found objects of interest in the following files:\n";
  run_det_result = "";
  int count = 0;

  for (const auto &bb_info : bb_list) {
    count++;
    std::string file_name = bb_info.file_name;
    std::vector<int> bounding_box = bb_info.bounding_box;

    float normed_xy = (float)(bounding_box[0]) / (float)(det_input_img_dim);
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << normed_xy;
    for (unsigned int i = 1; i < 4; i++) {
      normed_xy = (float)bounding_box[i] / (float)det_input_img_dim;
      ss << " " << normed_xy;
    }
    std::string bb_string = ss.str();
    std::vector<std::string> splited_file = split(file_name, '/');
    int macro_label = bb_info.detected_macro_label;
    float det_score = bb_info.detection_score;

    // run_det_result += " " + std::to_string(count) + " ] " +
    // (splited_file.rbegin()[0]) + "\n" +
    //             "     : predicted bb (xyxy) " + bb_string.c_str() + "\n" +
    //             "     : predicted macro label " +
    //             macro_class_names[macro_label].c_str() + "\n\n";
    run_det_result += "," + bb_string + "/" +
                      macro_class_names[macro_label].c_str() + "/" +
                      std::to_string(det_score);
  }
  ANDROID_LOG_D("Result: %s", run_det_result.c_str());
  return std::to_string(count) + run_det_result;
}
