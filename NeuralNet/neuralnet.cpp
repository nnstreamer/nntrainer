#include "include/neuralnet.h"
#include <assert.h>
#include <cmath>
#include <stdio.h>

namespace Network {
std::vector<std::string> Optimizer_string = {"sgd", "adam"};

void NeuralNetwork::init(int input, int hidden, int output, int batch,
                         double rate, std::string acti, bool init_zero) {
  inputNeuron = input;
  hiddenNeuron = hidden;
  outputNeuron = output;
  batchsize = batch;
  learning_rate = rate;
  loss = 100000.0;

  Layers::InputLayer *inputlayer = new (Layers::InputLayer);
  Layers::FullyConnectedLayer *fc1 = new (Layers::FullyConnectedLayer);
  Layers::FullyConnectedLayer *fc2 = new (Layers::FullyConnectedLayer);
  Layers::OutputLayer *outputlayer = new (Layers::OutputLayer);

  inputlayer->initialize(batch, 1, inputNeuron, 0, init_zero);
  fc1->initialize(batch, inputNeuron, hiddenNeuron, 1, init_zero);
  fc2->initialize(batch, hiddenNeuron, hiddenNeuron, 2, init_zero);
  outputlayer->initialize(batch, hiddenNeuron, outputNeuron, 3, init_zero);

  layers.push_back(inputlayer);
  layers.push_back(fc1);
  layers.push_back(fc2);
  layers.push_back(outputlayer);
  opt.activation = acti;
}

void NeuralNetwork::finalize() {
  for (unsigned int i = 0; i < layers.size(); i++) {
    delete layers[i];
  }
}

Matrix NeuralNetwork::forwarding(Matrix input) {

  Matrix X = input;
  for (unsigned int i = 0; i < layers.size(); i++) {
    X = layers[i]->forwarding(X);
  }
  return X;
}

void NeuralNetwork::backwarding(Matrix input, Matrix expected_output,
                                int iteration) {
  Matrix Y2 = expected_output;
  Matrix X = input;
  Matrix Y = forwarding(X);

  for (unsigned int i = layers.size() - 1; i > 0; i--) {
    Y2 = layers[i]->backwarding(Y2, i);
  }
}

double NeuralNetwork::getLoss() {
  Layers::OutputLayer *out =
      static_cast<Layers::OutputLayer *>((layers[layers.size() - 1]));
  return out->getLoss();
}
void NeuralNetwork::setLoss(double l) { loss = l; }

NeuralNetwork &NeuralNetwork::copy(NeuralNetwork &from) {
  if (this != &from) {
    inputNeuron = from.inputNeuron;
    outputNeuron = from.outputNeuron;
    hiddenNeuron = from.hiddenNeuron;
    batchsize = from.batchsize;
    learning_rate = from.learning_rate;
    loss = from.loss;
    opt = from.opt;

    for (unsigned int i = 0; i < layers.size(); i++)
      layers[i]->copy(from.layers[i]);
  }
  return *this;
}

void NeuralNetwork::saveModel(std::string model_path) {
  std::ofstream modelFile(model_path, std::ios::out | std::ios::binary);
  for (unsigned int i = 0; i < layers.size(); i++)
    layers[i]->save(modelFile);
  modelFile.close();
}

void NeuralNetwork::readModel(std::string model_path) {
  std::ifstream modelFile(model_path, std::ios::in | std::ios::binary);
  for (unsigned int i = 0; i < layers.size(); i++)
    layers[i]->read(modelFile);
  modelFile.close();
}

void NeuralNetwork::setOptimizer(std::string ty, double lr, double bt1,
                                 double bt2, double ep) {
  this->opt.type = Layers::OPT_SGD;
  this->opt.beta1 = 0.0;
  this->opt.beta2 = 0.0;
  this->opt.epsilon = 0.0;

  for (unsigned int i = 0; i < Optimizer_string.size(); i++) {
    if (Optimizer_string[i].compare(ty) == 0) {
      this->opt.type = (Layers::opt_type)i;
      break;
    }
  }

  this->opt.learning_rate = lr;
  if (bt1)
    this->opt.beta1 = bt1;
  if (bt2)
    this->opt.beta2 = bt2;
  if (ep)
    this->opt.epsilon = ep;

  for (unsigned int i = 0; i < layers.size(); i++) {
    layers[i]->setOptimizer(opt);
  }
}
}
