#include "include/layers.h"
#include <assert.h>

double random(double x) { return (double)(rand() % 10000 + 1) / 10000 - 0.5; }

double sigmoid(double x) { return 1 / (1 + exp(-x)); }

double sigmoidePrime(double x) { return exp(-x) / (pow(1 + exp(-x), 2)); }

double tanhPrime(double x) {
  double th = tanh(x);
  return 1.0 - th * th;
}

namespace Layers {

void InputLayer::setOptimizer(Optimizer opt) {
  this->opt = opt;
  switch (opt.activation) {
  case ACT_TANH:
    activation = tanh;
    activationPrime = tanhPrime;
    break;
  case ACT_SIGMOID:
    activation = sigmoid;
    activationPrime = sigmoidePrime;
    break;
  default:
    break;
  }
}

void InputLayer::copy(Layer *l) {
  InputLayer *from = static_cast<InputLayer *>(l);
  this->opt = from->opt;
  this->index = from->index;
  this->height = from->height;
  this->width = from->width;
  this->Input.copy(from->Input);
  this->hidden.copy(from->hidden);
}

Matrix InputLayer::forwarding(Matrix input) {
  Input = input;
  return Input;
}

void InputLayer::initialize(int b, int h, int w, int id, bool init_zero) {
  batch = b;
  width = w;
  height = h;
  index = 0;
}

void FullyConnectedLayer::initialize(int b, int h, int w, int id,
                                     bool init_zero) {
  this->batch = b;
  this->width = w;
  this->height = h;
  this->index = id;
  this->init_zero = init_zero;

  Weight = Matrix(h, w);
  Bias = Matrix(1, w);

  Weight = Weight.applyFunction(random);
  if (init_zero) {
    Bias.setZero();
  } else {
    Bias = Bias.applyFunction(random);
  }
}

void FullyConnectedLayer::setOptimizer(Optimizer opt) {
  this->opt = opt;
  switch (opt.activation) {
  case ACT_TANH:
    activation = tanh;
    activationPrime = tanhPrime;
    break;
  case ACT_SIGMOID:
    activation = sigmoid;
    activationPrime = sigmoidePrime;
    break;
  default:
    break;
  }
  if (opt.type == OPT_ADAM) {
    M = Matrix(height, width);
    V = Matrix(height, width);
    M.setZero();
    V.setZero();
  }
}

Matrix FullyConnectedLayer::forwarding(Matrix input) {
  Input = input;
  hidden = Input.dot(Weight).add(Bias).applyFunction(activation);
  return hidden;
}

void FullyConnectedLayer::read(std::ifstream &file) {
  Weight.read(file);
  Bias.read(file);
}

void FullyConnectedLayer::save(std::ofstream &file) {
  Weight.save(file);
  Bias.save(file);
}

void FullyConnectedLayer::copy(Layer *l) {
  FullyConnectedLayer *from = static_cast<FullyConnectedLayer *>(l);
  this->opt = from->opt;
  this->index = from->index;
  this->height = from->height;
  this->width = from->width;
  this->Input.copy(from->Input);
  this->hidden.copy(from->hidden);
  this->Weight.copy(from->Weight);
  this->Bias.copy(from->Bias);
}

Matrix FullyConnectedLayer::backwarding(Matrix derivative, int iteration) {
  Matrix dJdB = derivative.multiply(
      Input.dot(Weight).add(Bias).applyFunction(activationPrime));
  Matrix dJdW = Input.transpose().dot(dJdB);
  Matrix ret = dJdB.dot(Weight.transpose());

  switch (opt.type) {
  case OPT_SGD:
    Weight = Weight.subtract(dJdW.average().multiply(opt.learning_rate));
    break;
  case OPT_ADAM:
    M = M.multiply(opt.beta1).add(dJdW.average().multiply(1 - opt.beta1));
    V = V.multiply(opt.beta2).add(
        (dJdW.average().multiply(dJdW.average())).multiply(1 - opt.beta2));
    M.divide(1 - pow(opt.beta1, iteration + 1));
    V.divide(1 - pow(opt.beta2, iteration + 1));
    Weight = Weight.subtract((M.divide(V.applyFunction(sqrt).add(opt.epsilon)))
                                 .multiply(opt.learning_rate));
    break;
  default:
    break;
  }

  if (!this->init_zero) {
    Bias = Bias.subtract(dJdB.average().multiply(opt.learning_rate));
  }

  return ret;
}

void OutputLayer::initialize(int b, int h, int w, int id, bool init_zero) {
  this->batch = b;
  this->width = w;
  this->height = h;
  this->index = id;
  this->init_zero = init_zero;
  Weight = Matrix(h, w);
  Bias = Matrix(1, w);
  this->cost = cost;

  Weight = Weight.applyFunction(random);
  if (init_zero) {
    Bias.setZero();
  } else {
    Bias = Bias.applyFunction(random);
  }
}

Matrix OutputLayer::forwarding(Matrix input) {
  Input = input;
  if (cost == COST_LOGISTIC)
    hidden = input.dot(Weight).applyFunction(activation);
  else
    hidden = input.dot(Weight).add(Bias).applyFunction(activation);
  return hidden;
}

void OutputLayer::read(std::ifstream &file) {
  Weight.read(file);
  Bias.read(file);
}

void OutputLayer::save(std::ofstream &file) {
  Weight.save(file);
  Bias.save(file);
}

void OutputLayer::copy(Layer *l) {
  OutputLayer *from = static_cast<OutputLayer *>(l);
  this->opt = from->opt;
  this->index = from->index;
  this->height = from->height;
  this->width = from->width;
  this->Input.copy(from->Input);
  this->hidden.copy(from->hidden);
  this->Weight.copy(from->Weight);
  this->Bias.copy(from->Bias);
  this->loss = from->loss;
}

void OutputLayer::setOptimizer(Optimizer opt) {
  this->opt = opt;
  switch (opt.activation) {
  case ACT_TANH:
    activation = tanh;
    activationPrime = tanhPrime;
    break;
  case ACT_SIGMOID:
    activation = sigmoid;
    activationPrime = sigmoidePrime;
    break;
  default:
    break;
  }

  if (opt.type == OPT_ADAM) {
    M = Matrix(height, width);
    V = Matrix(height, width);
    M.setZero();
    V.setZero();
  }
}

Matrix OutputLayer::backwarding(Matrix label, int iteration) {
  double lossSum = 0.0;
  Matrix Y2 = label;
  Matrix Y = hidden;
  Matrix ret;
  Matrix dJdB;

  if (cost == COST_LOGISTIC) {
    dJdB = Y.subtract(Y2);
    Matrix temp =
        ((Y2.multiply(-1.0).transpose().dot(Y.add(1e-5).applyFunction(log)))
             .subtract(Y2.multiply(-1.0).add(1.0).transpose().dot(
                 Y.multiply(-1.0).add(1.0).add(1e-5).applyFunction(log))));
    loss = (1.0 / Y.Mat2Vec().size()) * temp.Mat2Vec()[0];
  } else {
    Matrix sub = Y2.subtract(Y);
    Matrix l = (sub.multiply(sub)).sum().multiply(0.5);
    std::vector<double> t = l.Mat2Vec();
    for (int i = 0; i < l.getBatch(); i++) {
      lossSum += t[i];
    }

    loss = lossSum / (double)l.getBatch();

    dJdB = Y.subtract(Y2).multiply(
        Input.dot(Weight).add(Bias).applyFunction(activationPrime));
  }

  Matrix dJdW = Input.transpose().dot(dJdB);
  ret = dJdB.dot(Weight.transpose());

  switch (opt.type) {
  case Layers::OPT_SGD:
    Weight = Weight.subtract(dJdW.average().multiply(opt.learning_rate));
    break;
  case Layers::OPT_ADAM:
    M = M.multiply(opt.beta1).add(dJdW.average().multiply(1 - opt.beta1));
    V = V.multiply(opt.beta2).add(
        (dJdW.average().multiply(dJdW.average())).multiply(1 - opt.beta2));
    M.divide(1 - pow(opt.beta1, iteration + 1));
    V.divide(1 - pow(opt.beta2, iteration + 1));
    Weight = Weight.subtract((M.divide(V.applyFunction(sqrt).add(opt.epsilon)))
                                 .multiply(opt.learning_rate));
    break;
  default:
    break;
  }

  if (!this->init_zero) {
    Bias = Bias.subtract(dJdB.average().multiply(opt.learning_rate));
  }

  return ret;
}
}
