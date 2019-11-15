#ifndef __NEURALNET_H__
#define __NEURALNET_H__

#include "matrix.h"
#include <fstream>
#include <iostream>
#include <vector>

namespace Network {

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
  void backwarding(Matrix input, Matrix expectedOutput);
  void saveModel(std::string model_path);
  void readModel(std::string model_path);
  NeuralNetwork &copy(NeuralNetwork &from);

private:
  Matrix W1, W2, W3, B1, B2, B3, H1, H2;

  int inputNeuron;
  int outputNeuron;
  int hiddenNeuron;
  int batchsize;
  double (*activation)(double);
  double (*activationPrime)(double);
  double learning_rate;
  double loss;
  bool init_zero;
};
}

#endif
