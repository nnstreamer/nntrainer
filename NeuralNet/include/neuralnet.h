#ifndef __NEURALNET_H__
#define __NEURALNET_H__

#include "layers.h"
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
  void backwarding(Matrix input, Matrix expectedOutput, int iteration);
  void saveModel(std::string model_path);
  void readModel(std::string model_path);
  void setOptimizer(std::string ty, double lr, double bt1, double bt2,
                    double ep);
  NeuralNetwork &copy(NeuralNetwork &from);
  void finalize();

private:
  int inputNeuron;
  int outputNeuron;
  int hiddenNeuron;
  int batchsize;
  double (*activation)(double);
  double (*activationPrime)(double);
  double learning_rate;
  double loss;
  bool init_zero;
  Layers::Optimizer opt;
  std::vector<Layers::Layer *> layers;
};
}

#endif
