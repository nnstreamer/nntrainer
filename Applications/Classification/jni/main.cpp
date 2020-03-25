/**
 * Copyright (C) 2019 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *
 * @file	main.cpp
 * @date	04 December 2019
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 * @brief	This is Classification Example with one FC Layer
 *              The base model for feature extractor is mobilenet v2 with 1280*7*7 feature size.
 *              It read the Classification.ini in res directory and run according to the configureation.
 *
 */

#include <stdlib.h>
#include <time.h>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <sstream>
#include "bitmap_helpers.h"
#include "tensorflow/contrib/lite/interpreter.h"
#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/string_util.h"
#include "tensorflow/contrib/lite/tools/gen_op_registration.h"

#include "databuffer.h"
#include "layers.h"
#include "neuralnet.h"
#include "tensor.h"

/**
 * @brief     Data size for each category
 */
#define TOTAL_TRAIN_DATA_SIZE 100

#define TOTAL_VAL_DATA_SIZE 10

#define TOTAL_TEST_DATA_SIZE 100

#define BUFFER_SIZE 100

/**
 * @brief     Number of category : Three
 */
#define TOTAL_LABEL_SIZE 10

/**
 * @brief     Number of Test Set
 */
#define TOTAL_TEST_SIZE 8

/**
 * @brief     Max Epoch
 */
#define ITERATION 3000

#define MINI_BATCH 32

#define training true

#define FEATURE_SIZE 62720

using namespace std;

/**
 * @brief     location of resources ( ../../res/ )
 */
string data_path;

bool duplicate[TOTAL_LABEL_SIZE * TOTAL_TRAIN_DATA_SIZE];
bool valduplicate[TOTAL_LABEL_SIZE * TOTAL_VAL_DATA_SIZE];

/**
 * @brief     step function
 * @param[in] x value to be distinguished
 * @retval 0.0 or 1.0
 */
float stepFunction(float x) {
  if (x > 0.9) {
    return 1.0;
  }

  if (x < 0.1) {
    return 0.0;
  }

  return x;
}

/**
 * @brief     Generate Random integer value between min to max
 * @param[in] min : minimum value
 * @param[in] max : maximum value
 * @retval    min < random value < max
 */
static int rangeRandom(int min, int max) {
  int n = max - min + 1;
  int remainder = RAND_MAX % n;
  int x;
  do {
    x = rand();
  } while (x >= RAND_MAX - remainder);
  return min + x % n;
}

/**
 * @brief     Get Feature vector from tensorflow lite
 *            This creates interpreter & inference with ssd tflite
 * @param[in] filename input file path
 * @param[out] feature_input save output of tflite
 */
void getFeature(const string filename, vector<float> &feature_input) {
  int input_size;
  int output_size;
  int *output_idx_list;
  int *input_idx_list;
  int inputDim[4];
  int outputDim[4];
  int input_idx_list_len = 0;
  int output_idx_list_len = 0;
  std::string model_path = "../../res/mobilenetv2.tflite";
  std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());

  assert(model != NULL);
  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter);

  input_size = interpreter->inputs().size();
  output_size = interpreter->outputs().size();

  input_idx_list = new int[input_size];
  output_idx_list = new int[output_size];

  int t_size = interpreter->tensors_size();
  for (int i = 0; i < t_size; i++) {
    for (int j = 0; j < input_size; j++) {
      if (strcmp(interpreter->tensor(i)->name, interpreter->GetInputName(j)) == 0)
        input_idx_list[input_idx_list_len++] = i;
    }
    for (int j = 0; j < output_size; j++) {
      if (strcmp(interpreter->tensor(i)->name, interpreter->GetOutputName(j)) == 0)
        output_idx_list[output_idx_list_len++] = i;
    }
  }
  for (int i = 0; i < 4; i++) {
    inputDim[i] = 1;
    outputDim[i] = 1;
  }

  int len = interpreter->tensor(input_idx_list[0])->dims->size;
  std::reverse_copy(interpreter->tensor(input_idx_list[0])->dims->data,
                    interpreter->tensor(input_idx_list[0])->dims->data + len, inputDim);
  len = interpreter->tensor(output_idx_list[0])->dims->size;
  std::reverse_copy(interpreter->tensor(output_idx_list[0])->dims->data,
                    interpreter->tensor(output_idx_list[0])->dims->data + len, outputDim);

  int output_number_of_pixels = 1;
  int wanted_channels = inputDim[0];
  int wanted_height = inputDim[1];
  int wanted_width = inputDim[2];

  for (int k = 0; k < 4; k++)
    output_number_of_pixels *= inputDim[k];

  int _input = interpreter->inputs()[0];

  uint8_t *in;
  float *output;
  in = tflite::label_image::read_bmp(filename, &wanted_width, &wanted_height, &wanted_channels);
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    std::cout << "Failed to allocate tensors!" << std::endl;
    exit(0);
  }

  for (int l = 0; l < output_number_of_pixels; l++) {
    (interpreter->typed_tensor<float>(_input))[l] = ((float)in[l]) / 255.0;
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    std::cout << "Failed to invoke!" << std::endl;
    exit(0);
  }

  output = interpreter->typed_output_tensor<float>(0);

  std::cout << inputDim[0] << " " << inputDim[1] << " " << inputDim[2] << " " << inputDim[3] << std::endl;
  std::cout << outputDim[0] << " " << outputDim[1] << " " << outputDim[2] << " " << outputDim[3] << std::endl;

  for (int l = 0; l < FEATURE_SIZE; l++) {
    feature_input[l] = output[l];
  }

  delete[] in;
  delete[] input_idx_list;
  delete[] output_idx_list;
}

/**
 * @brief     Extract the features from all three categories
 * @param[in] p data path
 * @param[out] feature_input save output of tflite
 * @param[out] feature_output save label data
 */
void ExtractFeatures(std::string p, vector<vector<float>> &feature_input, vector<vector<float>> &feature_output,
                     std::string type, std::ofstream &f) {
  string total_label[10] = {"airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"};

  int data_size = TOTAL_TRAIN_DATA_SIZE;
  bool val = false;

  if (!type.compare("val")) {
    data_size = TOTAL_VAL_DATA_SIZE;
    val = true;
  } else if (!type.compare("test")) {
    data_size = TOTAL_TEST_DATA_SIZE;
    val = false;
  }

  int trainingSize = TOTAL_LABEL_SIZE * data_size;

  feature_input.resize(trainingSize);
  feature_output.resize(trainingSize);

  int count = 0;

  for (int i = 0; i < TOTAL_LABEL_SIZE; i++) {
    std::string path = p;
    if (!type.compare("val") || !type.compare("training")) {
      path += "train/" + total_label[i];
    } else if (!type.compare("test")) {
      path += "test/" + total_label[i];
    }

    for (int j = 0; j < data_size; j++) {
      std::string img = path + "/";
      std::stringstream ss;

      if (val) {
        ss << std::setw(4) << std::setfill('0') << (5000 - j);
      } else {
        ss << std::setw(4) << std::setfill('0') << (j + 1);
      }

      img += ss.str() + ".bmp";
      printf("%s\n", img.c_str());

      std::vector<float> _input, _output;
      _input.resize(FEATURE_SIZE);
      _output.resize(TOTAL_LABEL_SIZE);

      getFeature(img, _input);
      _output[i] = 1;

      for (unsigned int k = 0; k < FEATURE_SIZE; ++k)
        f.write((char *)&_input[k], sizeof(float));

      for (unsigned int k = 0; k < TOTAL_LABEL_SIZE; ++k)
        f.write((char *)&_output[k], sizeof(float));

      count++;
    }
  }
}

bool getData(std::ifstream &F, std::vector<float> &outVec, std::vector<float> &outLabel, int id) {
  long pos = F.tellg();
  F.seekg(pos + (FEATURE_SIZE + TOTAL_LABEL_SIZE) * id);
  for (int i = 0; i < FEATURE_SIZE; i++)
    F.read((char *)&outVec[i], sizeof(float));
  for (int i = 0; i < TOTAL_LABEL_SIZE; i++)
    F.read((char *)&outLabel[i], sizeof(float));

  return true;
}

bool getMiniBatch(std::vector<std::vector<std::vector<float>>> &outVec,
                  std::vector<std::vector<std::vector<float>>> &outLabel, std::string type) {
  std::vector<int> memI;
  std::vector<int> memJ;
  int count = 0;
  int data_size = TOTAL_TRAIN_DATA_SIZE;

  std::string filename = type + "Set.dat";
  std::ifstream F(filename, std::ios::in | std::ios::binary);

  for (int i = 0; i < TOTAL_LABEL_SIZE * data_size; i++) {
    if (!duplicate[i])
      count++;
  }

  if (count < MINI_BATCH)
    return false;

  count = 0;
  while (count < MINI_BATCH) {
    int nomI = rangeRandom(0, TOTAL_LABEL_SIZE * data_size - 1);
    if (!duplicate[nomI]) {
      memI.push_back(nomI);
      duplicate[nomI] = true;
      count++;
    }
  }

  for (int i = 0; i < count; i++) {
    std::vector<std::vector<float>> out;
    std::vector<std::vector<float>> outL;
    std::vector<float> o;
    std::vector<float> l;

    o.resize(FEATURE_SIZE);
    l.resize(TOTAL_LABEL_SIZE);

    getData(F, o, l, memI[i]);

    out.push_back(o);
    outL.push_back(l);

    outVec.push_back(out);
    outLabel.push_back(outL);
  }

  F.close();
  return true;
}

void save(std::vector<std::vector<float>> inVec, std::vector<std::vector<float>> inLabel, std::string type) {
  std::string file = type + "Set.dat";

  unsigned int data_size = TOTAL_TRAIN_DATA_SIZE;

  if (!type.compare("val")) {
    data_size = TOTAL_VAL_DATA_SIZE;
  } else if (!type.compare("test")) {
    data_size = TOTAL_TEST_DATA_SIZE;
  }

  std::ofstream TrainigSet(file, std::ios::out | std::ios::binary);
  for (unsigned int i = 0; i < TOTAL_LABEL_SIZE * data_size; ++i) {
    for (unsigned int j = 0; j < FEATURE_SIZE; ++j) {
      TrainigSet.write((char *)&inVec[i][j], sizeof(float));
    }
    for (unsigned int j = 0; j < TOTAL_LABEL_SIZE; ++j)
      TrainigSet.write((char *)&inLabel[i][j], sizeof(float));
  }
}

bool read(std::vector<std::vector<float>> &inVec, std::vector<std::vector<float>> &inLabel, std::string type) {
  std::string file = type + "Set.dat";

  std::ifstream TrainingSet(file, std::ios::in | std::ios::binary);

  if (!TrainingSet.good())
    return false;

  return true;
}

/**
 * @brief     create NN
 *            Get Feature from tflite & run foword & back propatation
 * @param[in]  arg 1 : configuration file path
 * @param[in]  arg 2 : resource path
 */
int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cout << "./TransferLearning Config.ini resources\n";
    exit(0);
  }
  const vector<string> args(argv + 1, argv + argc);
  std::string config = args[0];
  data_path = args[1];

  srand(time(NULL));
  std::string ini_file = data_path + "ini.bin";
  std::vector<std::vector<float>> inputVector, outputVector;
  std::vector<std::vector<float>> inputValVector, outputValVector;
  std::vector<std::vector<float>> inputTestVector, outputTestVector;

  if (!read(inputVector, outputVector, "training")) {
    /**
     * @brief     Extract Feature
     */
    std::string filename = "trainingSet.dat";
    std::ofstream f(filename, std::ios::out | std::ios::binary);
    ExtractFeatures(data_path, inputVector, outputVector, "training", f);
    f.close();
  }

  if (!read(inputValVector, outputValVector, "val")) {
    /**
     * @brief     Extract Feature
     */
    std::string filename = "valSet.dat";
    std::ofstream f(filename, std::ios::out | std::ios::binary);
    ExtractFeatures(data_path, inputValVector, outputValVector, "val", f);
    f.close();
  }

  if (!read(inputValVector, outputValVector, "test")) {
    /**
     * @brief     Extract Feature
     */
    std::string filename = "testSet.dat";
    std::ofstream f(filename, std::ios::out | std::ios::binary);
    ExtractFeatures(data_path, inputTestVector, outputTestVector, "test", f);
    f.close();
  }

  /**
   * @brief     Neural Network Create & Initialization
   */
  Network::NeuralNetwork NN;
  NN.setConfig(config);
  NN.init();
  NN.readModel();

  DataBuffer buf;

  std::ifstream train_file("trainingSet.dat", std::ios::in | std::ios::binary);
  std::ifstream val_file("valSet.dat", std::ios::in | std::ios::binary);
  std::ifstream test_file("testSet.dat", std::ios::in | std::ios::binary);

  buf.init(MINI_BATCH, BUFFER_SIZE, BUFFER_SIZE, BUFFER_SIZE, train_file, val_file, test_file,
           TOTAL_LABEL_SIZE * TOTAL_TRAIN_DATA_SIZE, TOTAL_LABEL_SIZE * TOTAL_VAL_DATA_SIZE,
           TOTAL_LABEL_SIZE * TOTAL_TEST_DATA_SIZE, FEATURE_SIZE, 10);
  buf.run(BUF_TRAIN, train_file);
  buf.run(BUF_VAL, val_file);

  /**
   * @brief     back propagation
   */
  if (training) {
    float trainingloss = 0.0;
    for (int i = 0; i < ITERATION; i++) {
      int count = 0;
      for (int j = 0; j < TOTAL_LABEL_SIZE * TOTAL_TRAIN_DATA_SIZE; j++) {
        duplicate[j] = false;
      }

      float progress = 0.0;
      int barWidth = 20;

      while (true) {
        std::vector<std::vector<std::vector<float>>> in, label;
        if (buf.getDatafromBuffer(BUF_TRAIN, in, label, MINI_BATCH, FEATURE_SIZE, 1, TOTAL_LABEL_SIZE)) {
          NN.backwarding(Tensors::Tensor(in), Tensors::Tensor(label), i);
          count++;
          progress = (((float)(count * MINI_BATCH)) / (TOTAL_LABEL_SIZE * TOTAL_TRAIN_DATA_SIZE));
          int pos = barWidth * progress;
          std::cout << "#" << i + 1 << " [ ";
          for (int l = 0; l < barWidth; ++l) {
            if (l <= pos)
              std::cout << "=";
            else
              std::cout << " ";
          }
          std::cout << " ] " << int(progress * 100.0) << "% ( Training Loss: " << NN.getLoss() << " )\r";
          std::cout.flush();
        } else {
          buf.clear(BUF_TRAIN, train_file);
          buf.run(BUF_TRAIN, train_file);
          break;
        }
      }

      trainingloss = NN.getLoss();

      int right = 0;
      float valloss = 0.0;

      while (true) {
        std::vector<std::vector<std::vector<float>>> in, label;
        // if (buf.getDatafromBuffer(BUF_VAL, in, label, MINI_BATCH, FEATURE_SIZE, 1, TOTAL_LABEL_SIZE)) {
        if (buf.getDatafromBuffer(BUF_TRAIN, in, label, MINI_BATCH, FEATURE_SIZE, 1, TOTAL_LABEL_SIZE)) {
          for (int i = 0; i < MINI_BATCH; ++i) {
            Tensors::Tensor X = Tensors::Tensor({in[i]});
            Tensors::Tensor Y2 = Tensors::Tensor({label[i]});
            Tensors::Tensor Y = NN.forwarding(X, Y2);
            if (Y.argmax() == Y2.argmax())
              right++;
            valloss += NN.getLoss();
          }
        } else {
          // buf.clear(BUF_VAL, val_file);
          // buf.run(BUF_VAL, val_file);
          buf.clear(BUF_TRAIN, train_file);
          buf.run(BUF_TRAIN, train_file);
          break;
        }
      }

      valloss = valloss / (float)(TOTAL_LABEL_SIZE * TOTAL_VAL_DATA_SIZE);

      // cout << "#" << i + 1 << "/" << ITERATION << " - Training Loss: " << trainingloss << " >> [ Accuracy: " << right
      // / (float)(TOTAL_LABEL_SIZE * TOTAL_VAL_DATA_SIZE) * 100.0
      //      << "% - Validation Loss : " << valloss << " ] " << endl;
      cout << "#" << i + 1 << "/" << ITERATION << " - Training Loss: " << trainingloss
           << " >> [ Accuracy: " << right / (float)(TOTAL_LABEL_SIZE * TOTAL_TRAIN_DATA_SIZE) * 100.0
           << "% - Validation Loss : " << valloss << " ] " << endl;

      if (training)
        NN.saveModel();
    }
    buf.clear(BUF_TRAIN, train_file);
    buf.clear(BUF_VAL, val_file);
    buf.clear(BUF_TEST, test_file);
  }

  if (!training) {
    std::string img = data_path;
    std::vector<float> featureVector, resultVector;
    featureVector.resize(FEATURE_SIZE);
    getFeature(img, featureVector);
    Tensors::Tensor X = Tensors::Tensor({featureVector});
    cout << NN.forwarding(X).apply(stepFunction) << endl;
  }
  /**
   * @brief     Finalize NN
   */
  NN.finalize();
}
