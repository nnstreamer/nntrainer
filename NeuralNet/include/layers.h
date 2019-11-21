#ifndef __LAYERS_H__
#define __LAYERS_H__

#include "matrix.h"
#include <fstream>
#include <iostream>
#include <vector>

namespace Layers {

typedef enum { OPT_SGD = 0, OPT_ADAM = 1, OPT_UNKNOWN } opt_type;

typedef struct {
  opt_type type;
  double learning_rate;
  double beta1;
  double beta2;
  double epsilon;
  std::string activation;
} Optimizer;

class Layer {
public:
  virtual ~Layer(){};
  virtual Matrix forwarding(Matrix input) = 0;
  virtual Matrix backwarding(Matrix input, int iteration) = 0;
  virtual void initialize(int w, int h, int batch, int id, bool init_zero) = 0;
  virtual void read(std::ifstream &file) = 0;
  virtual void save(std::ofstream &file) = 0;
  virtual void setOptimizer(Optimizer opt) = 0;
  virtual void copy(Layer *l) = 0;

  Matrix Input;
  Matrix hidden;
  unsigned int index;
  unsigned int batch;
  unsigned int width;
  unsigned int height;
  Optimizer opt;
  bool init_zero;
  double (*activation)(double);
  double (*activationPrime)(double);
};

class InputLayer : public Layer {
public:
  InputLayer(){};
  ~InputLayer(){};
  void read(std::ifstream &file){};
  void save(std::ofstream &file){};
  Matrix backwarding(Matrix input, int iteration) { return Input; };
  Matrix forwarding(Matrix input);
  void setOptimizer(Optimizer opt);
  void initialize(int b, int w, int h, int id, bool init_zero);
  void copy(Layer *l);
};

class FullyConnectedLayer : public Layer {
public:
  FullyConnectedLayer(){};
  ~FullyConnectedLayer(){};
  void read(std::ifstream &file);
  void save(std::ofstream &file);
  Matrix forwarding(Matrix input);
  Matrix backwarding(Matrix input, int iteration);
  void setOptimizer(Optimizer opt);
  void copy(Layer *l);
  void initialize(int b, int w, int h, int id, bool init_zero);

private:
  Matrix Weight;
  Matrix Bias;
  Matrix M;
  Matrix V;
};

class OutputLayer : public Layer {
public:
  OutputLayer(){};
  ~OutputLayer(){};
  void read(std::ifstream &file);
  void save(std::ofstream &flle);
  Matrix forwarding(Matrix input);
  Matrix backwarding(Matrix label, int iteration);
  void setOptimizer(Optimizer opt);
  void initialize(int b, int w, int h, int id, bool init_zero);
  double getLoss() { return loss; }
  void copy(Layer *l);

private:
  Matrix Weight;
  Matrix Bias;
  Matrix M;
  Matrix V;
  double loss;
};
}

#endif
