#include <iostream>
#include <string>
#include <unistd.h>

#include "nntrainer.h"
#include "nntrainer_log.h"

using namespace nntrainer;

int main() {
  ml_train_model_h model;

  chdir("/data/nntrainer2/Applications/toy_classification");
  int status = ml_train_model_construct_with_conf("model.ini", &model);
  if (status != ML_ERROR_NONE) {
    std::cout << "constructing trainer model failed %d" << status << std::endl;
    return 1;
  }

  status = ml_train_model_compile(model, NULL);
  if (status != ML_ERROR_NONE) {
    std::cout << "compile model failed " << status << std::endl;
    ml_train_model_destroy(model);
    return 2;
  }

  status = ml_train_model_run(model, NULL);
  if (status != ML_ERROR_NONE) {
    std::cout << "run model failed " << status << std::endl;
    ml_train_model_destroy(model);
    return 3;
  }

  status = ml_train_model_destroy(model);
  if (status != ML_ERROR_NONE) {
    std::cout << "Destryoing model failed " << status << std::endl;
    return 4;
  }

  return 0;
}
