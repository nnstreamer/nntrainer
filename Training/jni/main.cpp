#include "bitmap_helpers.h"
#include "tensorflow/contrib/lite/interpreter.h"
#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/string_util.h"
#include "tensorflow/contrib/lite/tools/gen_op_registration.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <time.h>

#include "matrix.h"
#include "neuralnet.h"

#define TOTAL_DATA_SIZE 5
#define TOTAL_LABEL_SIZE 3
#define TOTAL_TEST_SIZE 8
#define ITERATION 300

using namespace std;

string data_path;

double stepFunction(double x) {
  if (x > 0.9) {
    return 1.0;
  }

  if (x < 0.1) {
    return 0.0;
  }

  return x;
}

void getFeature(const string filename, vector<double> &feature_input) {
  int input_size;
  int output_size;
  int *output_idx_list;
  int *input_idx_list;
  int inputDim[4];
  int outputDim[4];
  int input_idx_list_len = 0;
  int output_idx_list_len = 0;
  std::string model_path = data_path+"ssd_mobilenet_v2_coco_feature.tflite";
  std::unique_ptr<tflite::FlatBufferModel> model =
      tflite::FlatBufferModel::BuildFromFile(
					     model_path.c_str());

  assert(model != NULL);
  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder (*model.get(), resolver)(&interpreter);

  input_size = interpreter->inputs().size();
  output_size = interpreter->outputs().size();

  input_idx_list = new int[input_size];
  output_idx_list = new int[output_size];

  int t_size = interpreter->tensors_size();
  for (int i = 0; i < t_size; i++) {
    for (int j = 0; j < input_size; j++) {
      if (strcmp(interpreter->tensor(i)->name, interpreter->GetInputName(j)) ==
          0)
        input_idx_list[input_idx_list_len++] = i;
    }
    for (int j = 0; j < output_size; j++) {
      if (strcmp(interpreter->tensor(i)->name, interpreter->GetOutputName(j)) ==
          0)
        output_idx_list[output_idx_list_len++] = i;
    }
  }
  for (int i = 0; i < 4; i++) {
    inputDim[i] = 1;
    outputDim[i] = 1;
  }

  int len = interpreter->tensor(input_idx_list[0])->dims->size;
  std::reverse_copy(interpreter->tensor(input_idx_list[0])->dims->data,
                    interpreter->tensor(input_idx_list[0])->dims->data + len,
                    inputDim);
  len = interpreter->tensor(output_idx_list[0])->dims->size;
  std::reverse_copy(interpreter->tensor(output_idx_list[0])->dims->data,
                    interpreter->tensor(output_idx_list[0])->dims->data + len,
                    outputDim);

  int output_number_of_pixels = 1;
  int wanted_channels = inputDim[0];
  int wanted_height = inputDim[1];
  int wanted_width = inputDim[2];

  for (int k = 0; k < 4; k++)
    output_number_of_pixels *= inputDim[k];

  int _input = interpreter->inputs()[0];

  uint8_t *in;
  float *output;
  in = tflite::label_image::read_bmp(filename, &wanted_width, &wanted_height,
                                     &wanted_channels);
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    std::cout << "Failed to allocate tensors!" << std::endl;
    exit(0);
  }

  for (int l = 0; l < output_number_of_pixels; l++) {
    (interpreter->typed_tensor<float>(_input))[l] =
        ((float)in[l] - 127.5f) / 127.5f;
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    std::cout << "Failed to invoke!" << std::endl;
    exit(0);
  }

  output = interpreter->typed_output_tensor<float>(0);

  for (int l = 0; l < 128; l++) {
    feature_input[l] = output[l];
  }

  delete[] input_idx_list;
  delete[] output_idx_list;
}

void ExtractFeatures(std::string p, vector<vector<double>> &feature_input,
                     vector<vector<double>> &feature_output) {
  string total_label[TOTAL_LABEL_SIZE] = {"happy", "sad", "soso"};

  int trainingSize = TOTAL_LABEL_SIZE * TOTAL_DATA_SIZE;

  feature_input.resize(trainingSize);
  feature_output.resize(trainingSize);

  int count = 0;

  for (int i = 0; i < TOTAL_LABEL_SIZE; i++) {
    std::string path = p;
    path += total_label[i];

    for (int j = 0; j < TOTAL_DATA_SIZE; j++) {
      std::string img = path + "/";
      img += total_label[i] + std::to_string(j + 1) + ".bmp";
      printf("%s\n", img.c_str());

      feature_input[count].resize(128);

      getFeature(img, feature_input[count]);
      feature_output[count].resize(TOTAL_LABEL_SIZE);
      feature_output[count][i] = 1;
      count++;
    }
  }
}

int main(int argc, char *argv[]) {
  const vector<string> args(argv+1, argv+argc);
  data_path = args[0];
  std::string ini_file=data_path+"ini.bin";
  srand(time(NULL));

  std::vector<std::vector<double>> inputVector, outputVector;
  ExtractFeatures(data_path, inputVector, outputVector);

  Network::NeuralNetwork NN;
  Network::NeuralNetwork NN2;

  NN.init(128, 20, TOTAL_LABEL_SIZE, 0.7);
  NN2.init(128, 20, TOTAL_LABEL_SIZE, 0.7);

  // NN.saveModel(ini_file);
  NN.readModel(ini_file);

  for (int i = 0; i < ITERATION; i++) {
    for (unsigned int j = 0; j < inputVector.size(); j++) {
      NN.forwarding(inputVector[j]);
      NN.backwarding(outputVector[j]);
    }
    cout << "#" << i + 1 << "/" << ITERATION << " - Loss : " << NN.getLoss()
         << endl;
    NN.setLoss(0.0);
  }

  NN2.copy(NN);

  for (int i = 0; i < TOTAL_TEST_SIZE; i++) {
    std::string path = data_path;
    path += "testset";
    printf("\n[%s]\n", path.c_str());
    std::string img = path + "/";
    img += "test" + std::to_string(i + 1) + ".bmp";
    printf("%s\n", img.c_str());

    std::vector<double> featureVector, resultVector;
    featureVector.resize(128);
    getFeature(img, featureVector);
    cout << NN.forwarding(featureVector).applyFunction(stepFunction) << endl;
  }
}
