#include "neuralnet.h"
#include <assert.h>
#include <cmath>
#include <stdio.h>

// double random(double x) {
//   double min =-0.01;
//   double max = 0.01;
//   double r = (double)rand() / (double)RAND_MAX;
//   return min + r * (max - min);
// }

double random(double x) { return (double)(rand() % 10000 + 1) / 10000 - 0.5; }

double sigmoid(double x) { return 1 / (1 + exp(-x)); }

double sigmoidePrime(double x) { return exp(-x) / (pow(1 + exp(-x), 2)); }

double tanhPrime(double x) {
  double th = tanh(x);
  return 1.0 - th * th;
}

namespace Network {
void NeuralNetwork::init(int input, int hidden, int output, int batch,
                         double rate, std::string acti, bool init_zero) {
  inputNeuron = input;
  hiddenNeuron = hidden;
  outputNeuron = output;
  batchsize = batch;

  learning_rate = rate;
  loss = 100000.0;
  W1 = Matrix(inputNeuron, hiddenNeuron);
  W2 = Matrix(hiddenNeuron, hiddenNeuron);
  W3 = Matrix(hiddenNeuron, outputNeuron);
  B1 = Matrix(1, hiddenNeuron);
  B2 = Matrix(1, hiddenNeuron);
  B3 = Matrix(1, outputNeuron);

  if (init_zero) {
    W1 = W1.applyFunction(random);
    W2 = W2.applyFunction(random);
    W3 = W3.applyFunction(random);
    B1 = B1.multiply(0.0);
    B2 = B2.multiply(0.0);
    B3 = B3.multiply(0.0);
  } else {
    W1 = W1.applyFunction(random);
    W2 = W2.applyFunction(random);
    W3 = W3.applyFunction(random);
    B1 = B1.applyFunction(random);
    B2 = B2.applyFunction(random);
    B3 = B3.applyFunction(random);
  }

  if (acti.compare("tanh") == 0) {
    activation = tanh;
    activationPrime = tanhPrime;
  } else {
    activation = sigmoid;
    activationPrime = sigmoidePrime;
  }
  this->init_zero = init_zero;
}

Matrix NeuralNetwork::forwarding(std::vector<double> input) {
  assert(batchsize == 1);
  Matrix X = Matrix({input});
  H1 = X.dot(W1).add(B1).applyFunction(activation);
  H2 = H1.dot(W2).add(B2).applyFunction(activation);
  Matrix Y = H2.dot(W3).add(B3).applyFunction(activation);
  return Y;
  // return Y.softmax();
}

Matrix NeuralNetwork::forwarding(Matrix input) {
  Matrix X = input;
  H1 = X.dot(W1).add(B1).applyFunction(activation);
  H2 = H1.dot(W2).add(B2).applyFunction(activation);
  Matrix Y = H2.dot(W3).add(B3).applyFunction(activation);
  return Y;
  // return Y.softmax();
}

void NeuralNetwork::backwarding(Matrix input, Matrix expected_output) {
  double lossSum = 0.0;
  // Matrix Y2 = expected_output.softmax();
  Matrix Y2 = expected_output;
  Matrix X = input;
  Matrix Y = forwarding(X);

  Matrix sub = Y2.subtract(Y);
  Matrix l = (sub.multiply(sub)).sum().multiply(0.5);

  std::vector<double> t = l.Mat2Vec();
  for (int i = 0; i < l.getBatch(); i++) {
    lossSum += t[i];
  }

  loss = lossSum / (double)l.getBatch();

  Matrix dJdB3 = Y.subtract(Y2).multiply(
      H2.dot(W3).add(B3).applyFunction(activationPrime));
  Matrix dJdB2 =
      dJdB3.dot(W3.transpose())
          .multiply(H1.dot(W2).add(B2).applyFunction(activationPrime));
  Matrix dJdB1 =
      dJdB2.dot(W2.transpose())
          .multiply(X.dot(W1).add(B1).applyFunction(activationPrime));

  Matrix dJdW3 = H2.transpose().dot(dJdB3);
  Matrix dJdW2 = H1.transpose().dot(dJdB2);
  Matrix dJdW1 = X.transpose().dot(dJdB1);

  W1 = W1.subtract(dJdW1.average().multiply(learning_rate));
  W2 = W2.subtract(dJdW2.average().multiply(learning_rate));
  W3 = W3.subtract(dJdW3.average().multiply(learning_rate));

  if (!init_zero) {
    B1 = B1.subtract(dJdB1.average().multiply(learning_rate));
    B2 = B2.subtract(dJdB2.average().multiply(learning_rate));
    B3 = B3.subtract(dJdB3.average().multiply(learning_rate));
  }
}

double NeuralNetwork::getLoss() { return loss; }
void NeuralNetwork::setLoss(double l) { loss = l; }

NeuralNetwork &NeuralNetwork::copy(NeuralNetwork &from) {
  if (this != &from) {
    inputNeuron = from.inputNeuron;
    outputNeuron = from.outputNeuron;
    hiddenNeuron = from.hiddenNeuron;
    batchsize = from.batchsize;
    learning_rate = from.learning_rate;
    loss = from.loss;

    W1.copy(from.W1);
    W2.copy(from.W2);
    W3.copy(from.W3);

    B1.copy(from.B1);
    B2.copy(from.B2);
    B3.copy(from.B3);
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
