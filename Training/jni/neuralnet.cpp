#include "neuralnet.h"
#include <cmath>
#include <iostream>
#include <stdio.h>

double random(double x) { return (double)(rand() % 10000 + 1) / 10000 - 0.5; }

double sigmoid(double x) { return 1 / (1 + exp(-x)); }

double sigmoidePrime(double x) { return exp(-x) / (pow(1 + exp(-x), 2)); }

namespace Network {
void NeuralNetwork::init(int input, int hidden, int output, double rate) {
  inputNeuron = input;
  hiddenNeuron = hidden;
  outputNeuron = output;
  learning_rate = rate;
  loss = 0.0;
  W1 = Matrix(inputNeuron, hiddenNeuron);
  // W2 = Matrix(hiddenNeuron, outputNeuron);
  W2 = Matrix(hiddenNeuron, hiddenNeuron);
  W3 = Matrix(hiddenNeuron, outputNeuron);  
  B1 = Matrix(1, hiddenNeuron);
  B2 = Matrix(1, hiddenNeuron);  
  B3 = Matrix(1, outputNeuron);

  W1 = W1.applyFunction(random);
  W2 = W2.applyFunction(random);
  W3 = W3.applyFunction(random);  
  B1 = B1.applyFunction(random);
  B2 = B2.applyFunction(random);
  B3 = B3.applyFunction(random);
}

Matrix NeuralNetwork::forwarding(std::vector<double> input) {
  X = Matrix({input});
  // H = X.dot(W1).add(B1).applyFunction(sigmoid);
  // Y = H.dot(W2).add(B2).applyFunction(sigmoid);

  H1 = X.dot(W1).add(B1).applyFunction(sigmoid);
  H2 = H1.dot(W2).add(B2).applyFunction(sigmoid);  
  Y = H2.dot(W3).add(B3).applyFunction(sigmoid);
  
  return Y;
}

void NeuralNetwork::backwarding(std::vector<double> expectedOutput) {
  Matrix Yt = Matrix({expectedOutput});
  double l = sqrt((Yt.subtract(Y)).multiply(Yt.subtract(Y)).sum()) * 1.0 / 2.0;
  if (l > loss)
    loss = l;
  Y2 = Matrix({expectedOutput});

  dJdB3 =
      Y.subtract(Y2).multiply(H2.dot(W3).add(B3).applyFunction(sigmoidePrime));
  dJdB2 =dJdB3.dot(W3.transpose())
              .multiply(H1.dot(W2).add(B2).applyFunction(sigmoidePrime));
  dJdB1 = dJdB2.dot(W2.transpose())
              .multiply(X.dot(W1).add(B1).applyFunction(sigmoidePrime));
  dJdW3 = H2.transpose().dot(dJdB3);  
  dJdW2 = H1.transpose().dot(dJdB2);
  dJdW1 = X.transpose().dot(dJdB1);
  
  // dJdB2 =
  //     Y.subtract(Y2).multiply(H.dot(W2).add(B2).applyFunction(sigmoidePrime));
  // dJdB1 = dJdB2.dot(W2.transpose())
  //             .multiply(X.dot(W1).add(B1).applyFunction(sigmoidePrime));
  // dJdW2 = H.transpose().dot(dJdB2);
  // dJdW1 = X.transpose().dot(dJdB1);

  W1 = W1.subtract(dJdW1.multiply(learning_rate));
  W2 = W2.subtract(dJdW2.multiply(learning_rate));
  W3 = W3.subtract(dJdW3.multiply(learning_rate));  
  B1 = B1.subtract(dJdB1.multiply(learning_rate));
  B2 = B2.subtract(dJdB2.multiply(learning_rate));
  B3 = B3.subtract(dJdB3.multiply(learning_rate));  
}

double NeuralNetwork::getLoss() { return loss; }
void NeuralNetwork::setLoss(double l) { loss = l; }

NeuralNetwork &NeuralNetwork::copy(NeuralNetwork const &from) {
  if (this != &from) {
    inputNeuron = from.inputNeuron;
    outputNeuron = from.outputNeuron;
    hiddenNeuron = from.hiddenNeuron;
    learning_rate = from.learning_rate;
    loss = from.loss;

    W1.copy(from.W1);
    W2.copy(from.W2);
    B1.copy(from.B1);
    B2.copy(from.B2);
  }
  return *this;
}
void NeuralNetwork::saveModel(std::string model_path) {
  std::ofstream modelFile(model_path, std::ios::out | std::ios::binary);
    W1.save(modelFile);
    W2.save(modelFile);
    W3.save(modelFile);
    B1.save(modelFile);
    B2.save(modelFile);
    B3.save(modelFile);
    modelFile.close();
}

void NeuralNetwork::readModel(std::string model_path) {
  std::ifstream modelFile(model_path, std::ios::in | std::ios::binary);
    W1.read(modelFile);
    W2.read(modelFile);
    W3.read(modelFile);
    B1.read(modelFile);
    B2.read(modelFile);
    B3.read(modelFile);
    modelFile.close();
}  
  
}

