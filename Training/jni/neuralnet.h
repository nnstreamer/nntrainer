#ifndef __NEURALNET_H__
#define __NEURALNET_H__

#include "matrix.h"
#include <vector>

namespace Network {

class NeuralNetwork {
public:
  NeuralNetwork(){};
  ~NeuralNetwork(){};

  double getLoss();
  void setLoss(double l);

  void init(int input, int hidden, int output, double rate);
  Matrix forwarding(std::vector<double> input);
  void backwarding(std::vector<double> expectedOutput);
  void saveModel(std::string model_path);
  void readModel(std::string model_path);  

  NeuralNetwork &copy(NeuralNetwork const &from);

private:
  Matrix X, W1, H, W2, Y, B1, B2, Y2, dJdB1, dJdB2, dJdW1, dJdW2, W3, B3, dJdB3, dJdW3, H1, H2;

  int inputNeuron;
  int outputNeuron;
  int hiddenNeuron;
  double learning_rate;
  double loss;
};
}

#endif
