#ifndef __NEURALNET_H__
#define __NEURALNET_H__

#include "layers.h"
#include "matrix.h"
#include <fstream>
#include <iostream>
#include <vector>

namespace Network {

typedef enum { NET_KNN, NET_REG, NET_NEU, NET_UNKNOWN } net_type;
typedef enum {
  TOKEN_OPT,
  TOKEN_COST,
  TOKEN_NET,
  TOKEN_ACTI,
  TOKEN_LAYER,
  TOKEN_UNKNOWN
} input_type;

class NeuralNetwork {

public:
  NeuralNetwork(){};
  NeuralNetwork(std::string config_path);
  ~NeuralNetwork(){};

  double getLoss();
  void setLoss(double l);

  void init();
  Matrix forwarding(Matrix input);
  void backwarding(Matrix input, Matrix expectedOutput, int iteration);
  void saveModel();
  void readModel();
  void setConfig(std::string config_path);
  unsigned int getEpoch() { return epoch; };
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
  unsigned int epoch;
  double loss;
  bool init_zero;
  Layers::cost_type cost;
  std::string model;
  std::string config;
  Layers::Optimizer opt;
  net_type nettype;
  std::vector<Layers::Layer *> layers;
};
}

#endif
