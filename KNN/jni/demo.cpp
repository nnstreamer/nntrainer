#include <stdio.h>
#include "tensorflow/contrib/lite/interpreter.h"
#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/string_util.h"
#include "tensorflow/contrib/lite/string.h"
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/tools/gen_op_registration.h"
#include "bitmap_helpers.h"
#include <iostream>
#include <fstream>
#include "math.h"

#define TOTAL_DATA_SIZE 5
#define TOTAL_LABEL_SIZE 3
#define TOTAL_TEST_SIZE 8

int KNN(float out[3][5][128], float *test){
  int ret=0;
  float dist[15];

  int count =0;
  float sum=0.0;
  float max=100000.0;
  int id = 0;
  for(int i=0;i<TOTAL_LABEL_SIZE;i++){
    for(int j=0;j<TOTAL_DATA_SIZE;j++){
      float d;
      for(int k=0;k<128;k++){
	sum += (out[i][j][k] - test[k])*(out[i][j][k] - test[k]);
      }
      d=sqrt(sum);
      dist[count++]=d;
      if(d < max){
	max=d;
	id=i;
      }
      printf("id %d, dist %f\n", id, d);
      sum=0.0;
    }
  }
  ret = id;

  return ret;
}


int main(){
  int tensor_size;
  int node_size;
  int input_size;
  int output_size;
  int *output_idx_list;
  int *input_idx_list;
  int inputDim[4];
  int outputDim[4];
  int input_idx_list_len=0;
  int output_idx_list_len = 0;
  float out[TOTAL_LABEL_SIZE][TOTAL_DATA_SIZE][128];

  char *total_label[TOTAL_LABEL_SIZE]={"happy","sad","soso"};
  char *data_path="/sdcard/Transfer-Learning/";
  
  std::unique_ptr<tflite::FlatBufferModel> model= tflite::FlatBufferModel::BuildFromFile("/sdcard/Transfer-Learning/ssd_mobilenet_v2_coco_feature.tflite");

  if(!model){
    printf("Failed to mmap mdoel\n");
    exit(0);
  }

  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter);

  tensor_size = interpreter->tensors_size();
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
      if(strcmp(interpreter->tensor(i)->name, interpreter->GetOutputName(j))==0)
	output_idx_list[output_idx_list_len++] = i;
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

  printf("input %d %d %d %d\n",inputDim[0], inputDim[1], inputDim[2], inputDim[3]);
  printf("output %d %d %d %d\n",outputDim[0], outputDim[1], outputDim[2], outputDim[3]);
  
  int output_number_of_pixels = 1;

  int wanted_channels = inputDim[0];
  int wanted_height = inputDim[1];
  int wanted_width = inputDim[2];
  
  for(int k=0;k<4;k++)
    output_number_of_pixels *= inputDim[k];

  int input=interpreter->inputs()[0];  
  
  for(int i=0;i<TOTAL_LABEL_SIZE;i++){
    std::string path = data_path;
    path += total_label[i];
    printf("\n[%s]\n", path.c_str());
    
    for(int j=0;j<TOTAL_DATA_SIZE;j++){
      std::string img=path+"/";
      img += total_label[i]+std::to_string(j+1)+".bmp";
      printf("%s\n",img.c_str());

      uint8_t *in;
      float *output;
      in = tflite::label_image::read_bmp(img, &wanted_width, &wanted_height, &wanted_channels);

      if(interpreter->AllocateTensors() != kTfLiteOk){
	std::cout << "Failed to allocate tnesors!" <<std::endl;
	return -2;
      }

      for(int l=0;l<output_number_of_pixels;l++){
	(interpreter->typed_tensor<float>(input))[l] =
	  ((float) in[l]-127.5f)/127.5f;
      }

      if(interpreter->Invoke() != kTfLiteOk) {
	std::cout << "Failed to invoke!" <<std::endl;	
	return -3;
      }

      output = interpreter->typed_output_tensor<float>(0);

      std::copy(output, output+128, out[i][j]);
      
    }
  }

  for(int i=0;i<TOTAL_LABEL_SIZE;i++){
    for(int j=0;j<TOTAL_DATA_SIZE;j++){
      std::string out_file="/sdcard/Transfer-Learning/";
      out_file += total_label[i]+std::to_string(j+1)+".txt";
      printf("%s\n",out_file.c_str());
      std::ofstream writeFile(out_file.data());
      if(writeFile.is_open()){
      	for(int k=0;k<128;k++)
      	  writeFile << out[i][j][k]<<std::endl;
      	writeFile.close();
      }
    }
  }

  
  float testout[TOTAL_TEST_SIZE][128];
  
  for(int i=0;i<TOTAL_TEST_SIZE;i++){
    std::string path = data_path;
    path += "testset";
    printf("\n[%s]\n", path.c_str());

    std::string img = path+"/";
    img += "test" + std::to_string(i+1)+".bmp";
    printf("%s\n", img.c_str());

    uint8_t *in;
    float *output;
    in=tflite::label_image::read_bmp(img, &wanted_width, &wanted_height, &wanted_channels);

    if(interpreter->AllocateTensors() != kTfLiteOk){
      std::cout << "Failed to allocate tnesors!" <<std::endl;
      return -2;
    }

    for(int l=0;l<output_number_of_pixels;l++){
      (interpreter->typed_tensor<float>(input))[l] =
	((float) in[l]-127.5f)/127.5f;
    }

    if(interpreter->Invoke() != kTfLiteOk) {
      std::cout << "Failed to invoke!" <<std::endl;
      return -3;
    }

    output = interpreter->typed_output_tensor<float>(0);
    std::copy(output, output+128, testout[i]);
    
    int ret=0;
    
    ret=KNN(out, testout[i]);
    printf("class %d\n", ret);
  }

  delete[] input_idx_list;
  delete[] output_idx_list;

  return 0;
}
