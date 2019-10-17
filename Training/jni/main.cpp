#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <time.h>
#include <cmath>
#include "tensorflow/contrib/lite/interpreter.h"
#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/string_util.h"
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/tools/gen_op_registration.h"
#include "bitmap_helpers.h"

#include "matrix.h"
#define TOTAL_DATA_SIZE 5
#define TOTAL_LABEL_SIZE 3
#define TOTAL_TEST_SIZE 8
#define ITERATION 300

using namespace std;

string data_path="/sdcard/Transfer-Learning/";

Matrix X, W1, H, W2, Y, B1, B2, Y2, dJdB1, dJdB2, dJdW1, dJdW2;
double learningRate;
double loss = 0.0;

double random(double x){
  return (double)(rand() %10000 +1)/10000-0.5;
}

double sigmoid(double x){
  return 1/(1+exp(-x));
}

double sigmoidePrime(double x){
  return exp(-x)/(pow(1+exp(-x),2)); 
}

double stepFunction(double x){
  if(x>0.9){
    return 1.0;
  }

  if(x<0.1){
    return 0.0;
  }

  return x;
}


void init(int inputNeuron, int hiddenNeuron, int outputNeuron, double rate){
  learningRate = rate;
  W1=Matrix(inputNeuron, hiddenNeuron);
  W2=Matrix(hiddenNeuron, outputNeuron);
  B1=Matrix(1, hiddenNeuron);
  B2=Matrix(1, outputNeuron);

  W1=W1.applyFunction(random);
  W2=W2.applyFunction(random);
  B1=B1.applyFunction(random);
  B2=B2.applyFunction(random);

}

Matrix computeOutput(vector<double> input){
  X = Matrix({input});
  H=X.dot(W1).add(B1).applyFunction(sigmoid);
  Y=H.dot(W2).add(B2).applyFunction(sigmoid);
  return Y;
}

void learn(vector<double> expectedOutput){
  Matrix Yt=Matrix({expectedOutput});
  double l = sqrt((Yt.subtract(Y)).multiply(Yt.subtract(Y)).sum())*1.0/2.0;
  if(l > loss) loss = l;
  
  Y2=Matrix({expectedOutput});

  dJdB2=Y.subtract(Y2).multiply(H.dot(W2).add(B2).applyFunction(sigmoidePrime));
  dJdB1=dJdB2.dot(W2.transpose()).multiply(X.dot(W1).add(B1).applyFunction(sigmoidePrime));
  dJdW2=H.transpose().dot(dJdB2);
  dJdW1=X.transpose().dot(dJdB1);

  W1=W1.subtract(dJdW1.multiply(learningRate));
  W2=W2.subtract(dJdW2.multiply(learningRate));
  B1=B1.subtract(dJdB1.multiply(learningRate));
  B2=B2.subtract(dJdB2.multiply(learningRate));
}

void getFeature(const string filename, vector<double>&feature_input){
  int tensor_size;
  int node_size;
  int input_size;
  int output_size;
  int *output_idx_list;
  int *input_idx_list;
  int inputDim[4];
  int outputDim[4];
  int input_idx_list_len=0;
  int output_idx_list_len=0;
  
  std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile("/sdcard/Transfer-Learning/ssd_mobilenet_v2_coco_feature.tflite");

  assert(model != NULL);
  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter);

  tensor_size=interpreter->tensors_size();
  node_size = interpreter->nodes_size();
  input_size = interpreter->inputs().size();
  output_size = interpreter->outputs().size();

  input_idx_list = new int[input_size];
  output_idx_list = new int[output_size];

  int t_size = interpreter->tensors_size();
  for(int i=0;i<t_size;i++){
    for(int j=0;j<input_size;j++){
      if(strcmp(interpreter->tensor(i)->name, interpreter->GetInputName(j)) == 0)
	input_idx_list[input_idx_list_len++]=i;
    }
    for(int j=0;j<output_size;j++){
      if(strcmp(interpreter->tensor(i)->name, interpreter->GetOutputName(j)) == 0)
	output_idx_list[output_idx_list_len++]=i;
    }    
  }
  for(int i=0;i<4;i++){
    inputDim[i]=1;
    outputDim[i]=1;
  }

  int len = interpreter->tensor(input_idx_list[0])->dims->size;
  std::reverse_copy(interpreter->tensor(input_idx_list[0])->dims->data,
		    interpreter->tensor(input_idx_list[0])->dims->data+len, inputDim);
  len = interpreter->tensor(output_idx_list[0])->dims->size;
  std::reverse_copy(interpreter->tensor(output_idx_list[0])->dims->data,
		    interpreter->tensor(output_idx_list[0])->dims->data+len, outputDim);

  int output_number_of_pixels=1;
  int wanted_channels = inputDim[0];
  int wanted_height=inputDim[1];
  int wanted_width = inputDim[2];

  for(int k=0;k<4;k++)
    output_number_of_pixels *= inputDim[k];

  int _input = interpreter->inputs()[0];
  
  uint8_t *in;
  float* output;
  in=tflite::label_image::read_bmp(filename,&wanted_width, &wanted_height, &wanted_channels);
  if(interpreter->AllocateTensors() != kTfLiteOk){
    std::cout << "Failed to allocate tensors!"<<std::endl;
    exit(0);
  }

  for(int l=0;l<output_number_of_pixels;l++){
    (interpreter->typed_tensor<float>(_input))[l] =
      ((float) in[l]-127.5f)/127.5f;
  }
      
  if(interpreter->Invoke()!=kTfLiteOk){
    std::cout <<"Failed to invoke!"<<std::endl;
    exit(0);
  }

  output = interpreter->typed_output_tensor<float>(0);

  for(int l=0;l<128;l++){
    feature_input[l]=output[l];
  }
  
  delete[] input_idx_list;
  delete[] output_idx_list;
}

void ExtractFeatures(const char* path, vector<vector<double>>&feature_input, vector<vector<double>>&feature_output){
  string total_label[TOTAL_LABEL_SIZE]={"happy", "sad", "soso"};


  int trainingSize = TOTAL_LABEL_SIZE * TOTAL_DATA_SIZE;
  
  feature_input.resize(trainingSize);
  feature_output.resize(trainingSize);

  int count=0;
  
  for(int i=0;i<TOTAL_LABEL_SIZE;i++){
    std::string path = data_path;
    path += total_label[i];

    for(int j=0;j<TOTAL_DATA_SIZE;j++){
      std::string img = path+"/";
      img += total_label[i]+std::to_string(j+1)+".bmp";
      printf("%s\n",img.c_str());
      
      feature_input[count].resize(128);
      
      getFeature(img, feature_input[count]);
      feature_output[count].resize(TOTAL_LABEL_SIZE);
      feature_output[count][i]=1;
      count++;
    }
  }

}

int main(int argc, char*argv[]){

  srand(time(NULL));

  std::vector<std::vector<double>> inputVector, outputVector;
  ExtractFeatures("/sdcard/Transfer-Learning/",inputVector, outputVector);

  init(128,20,TOTAL_LABEL_SIZE,0.7);

  for(int i=0;i<ITERATION;i++){
    for(int j=0; j<inputVector.size();j++){
      computeOutput(inputVector[j]);
      learn(outputVector[j]);
    }
    cout<<"#"<<i+1<<"/"<<ITERATION<< " - Loss : "<< loss<< endl;
    loss = 0.0;
  }

  for(int i=0;i<TOTAL_TEST_SIZE;i++){
    std::string path = data_path;
    path += "testset";
    printf("\n[%s]\n", path.c_str());
    std::string img=path+"/";
    img += "test" + std::to_string(i+1)+".bmp";
    printf("%s\n",img.c_str());

    std::vector<double> featureVector, resultVector;
    featureVector.resize(128);
    getFeature(img, featureVector);
    cout << computeOutput(featureVector).applyFunction(stepFunction)<<endl;
  }
}
