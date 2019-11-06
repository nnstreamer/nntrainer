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
  W2 = Matrix(hiddenNeuron, outputNeuron);
  B1 = Matrix(1, hiddenNeuron);
  B2 = Matrix(1, outputNeuron);

  W1 = W1.applyFunction(random);
  W2 = W2.applyFunction(random);
  B1 = B1.applyFunction(random);
  B2 = B2.applyFunction(random);
}

Matrix NeuralNetwork::forwarding(std::vector<double> input) {
  X = Matrix({input});
  H = X.dot(W1).add(B1).applyFunction(sigmoid);
  Y = H.dot(W2).add(B2).applyFunction(sigmoid);
  return Y;
}

void NeuralNetwork::backwarding(std::vector<double> expectedOutput) {
  Matrix Yt = Matrix({expectedOutput});
  double l = sqrt((Yt.subtract(Y)).multiply(Yt.subtract(Y)).sum()) * 1.0 / 2.0;
  if (l > loss)
    loss = l;
  Y2 = Matrix({expectedOutput});
  dJdB2 =
      Y.subtract(Y2).multiply(H.dot(W2).add(B2).applyFunction(sigmoidePrime));
  dJdB1 = dJdB2.dot(W2.transpose())
              .multiply(X.dot(W1).add(B1).applyFunction(sigmoidePrime));
  dJdW2 = H.transpose().dot(dJdB2);
  dJdW1 = X.transpose().dot(dJdB1);

  W1 = W1.subtract(dJdW1.multiply(learning_rate));
  W2 = W2.subtract(dJdW2.multiply(learning_rate));
  B1 = B1.subtract(dJdB1.multiply(learning_rate));
  B2 = B2.subtract(dJdB2.multiply(learning_rate));
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
}
