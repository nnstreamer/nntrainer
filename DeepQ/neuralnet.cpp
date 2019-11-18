#include "neuralnet.h"
#include <assert.h>
#include <cmath>
#include <stdio.h>

double random(double x) { return (double)(rand() % 10000 + 1) / 10000 - 0.5; }

double sigmoid(double x) { return 1 / (1 + exp(-x)); }

double sigmoidePrime(double x) { return exp(-x) / (pow(1 + exp(-x), 2)); }

double tanhPrime(double x) {
  double th = tanh(x);
  return 1.0 - th * th;
}

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

void NeuralNetwork::backwarding(Matrix input, Matrix expected_output,
                                int iteration) {
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

  switch (opt.type) {
  case OPT_SGD:
    W1 = W1.subtract(dJdW1.average().multiply(opt.learning_rate));
    W2 = W2.subtract(dJdW2.average().multiply(opt.learning_rate));
    W3 = W3.subtract(dJdW3.average().multiply(opt.learning_rate));
    break;
  case OPT_ADAM:
    m1 = m1.multiply(opt.beta1).add(dJdW1.average().multiply(1 - opt.beta1));
    v1 = v1.multiply(opt.beta2).add(
        (dJdW1.average().multiply(dJdW1.average())).multiply(1 - opt.beta2));
    W1 = W1.subtract((m1.divide(v1.applyFunction(sqrt).add(opt.epsilon)))
                         .multiply(opt.learning_rate));

    m2 = m2.multiply(opt.beta1).add(dJdW2.average().multiply(1 - opt.beta1));
    v2 = v2.multiply(opt.beta2).add(
        (dJdW2.average().multiply(dJdW2.average())).multiply(1 - opt.beta2));
    W2 = W2.subtract((m2.divide(v2.applyFunction(sqrt).add(opt.epsilon)))
                         .multiply(opt.learning_rate));

    m3 = m3.multiply(opt.beta1).add(dJdW3.average().multiply(1 - opt.beta1));
    v3 = v3.multiply(opt.beta2).add(
        (dJdW3.average().multiply(dJdW3.average())).multiply(1 - opt.beta2));
    W3 = W3.subtract((m3.divide(v3.applyFunction(sqrt).add(opt.epsilon)))
                         .multiply(opt.learning_rate));
    break;
  default:
    break;
  }

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
    opt = from.opt;
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

void NeuralNetwork::setOptimizer(std::string ty, double lr, double bt1,
                                 double bt2, double ep) {
  this->opt.type = OPT_SGD;
  this->opt.beta1 = 0.0;
  this->opt.beta2 = 0.0;
  this->opt.epsilon = 0.0;

  for (unsigned int i = 0; i < Optimizer_string.size(); i++) {
    if (Optimizer_string[i].compare(ty) == 0) {
      this->opt.type = (opt_type)i;
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

  if (opt.type == OPT_ADAM) {
    m1 = Matrix(inputNeuron, hiddenNeuron);
    m2 = Matrix(hiddenNeuron, hiddenNeuron);
    m3 = Matrix(hiddenNeuron, outputNeuron);

    v1 = Matrix(inputNeuron, hiddenNeuron);
    v2 = Matrix(hiddenNeuron, hiddenNeuron);
    v3 = Matrix(hiddenNeuron, outputNeuron);

    m1.setZero();
    m2.setZero();
    m3.setZero();

    v1.setZero();
    v2.setZero();
    v3.setZero();
  }
}
}
