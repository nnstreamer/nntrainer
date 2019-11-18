#ifndef __NEURALNET_H__
#define __NEURALNET_H__

#include "matrix.h"
#include <fstream>
#include <iostream>
#include <vector>

namespace Network {

typedef enum { OPT_SGD = 0, OPT_ADAM = 1, OPT_UNKNOWN } opt_type;

typedef struct {
  opt_type type;
  double learning_rate;
  double beta1;
  double beta2;
  double epsilon;
} Optimizer;

class NeuralNetwork {
public:
  NeuralNetwork(){};
  ~NeuralNetwork(){};

  double getLoss();
  void setLoss(double l);

  void init(int input, int hidden, int output, int batch, double rate,
            std::string acti, bool init_zero);
  Matrix forwarding(Matrix input);
  Matrix forwarding(std::vector<double> input);
  void backwarding(Matrix input, Matrix expectedOutput, int iteration);
  void saveModel(std::string model_path);
  void readModel(std::string model_path);
  void setOptimizer(std::string ty, double lr, double bt1, double bt2,
                    double ep);
  NeuralNetwork &copy(NeuralNetwork &from);

private:
  Matrix W1, W2, W3, B1, B2, B3, H1, H2;
  Matrix m1, m2, m3, v1, v2, v3;

  int inputNeuron;
  int outputNeuron;
  int hiddenNeuron;
  int batchsize;
  double (*activation)(double);
  double (*activationPrime)(double);
  double learning_rate;
  double loss;
  bool init_zero;
  Optimizer opt;
  /* bool Adam; */
};
}

#endif
