/**
 * @file	layers.h
 * @date	04 December 2019
 * @brief	This is Layer classes of Neural Network
 * @see		https://github.sec.samsung.net/jijoong-moon/Transfer-Learning.git
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */
#ifndef __LAYERS_H__
#define __LAYERS_H__

#include <fstream>
#include <iostream>
#include <vector>
#include "matrix.h"

/**
 * @Namespace   Namespace of Layers
 * @brief       Namespace for Layers
 */
namespace Layers {

/**
 * @brief     Enumeration of optimizer type
 *            0. SGD ( Stocastic Gradient Descent )
 *            1. ADAM ( Adaptive Moment Estimation )
 *            2. Unknown
 */
typedef enum { OPT_SGD, OPT_ADAM, OPT_UNKNOWN } opt_type;

/**
 * @brief     Enumeration of cost(loss) function type
 *            0. MSR ( Mean Squared Roots )
 *            1. ENTROPY ( Categorical Cross Entropy )
 *            2. Unknown
 */
typedef enum { COST_MSR, COST_ENTROPY, COST_UNKNOWN } cost_type;

/**
 * @brief     Enumeration of activation function type
 *            0. tanh
 *            1. sigmoid
 *            2. Unknown
 */
typedef enum { ACT_TANH, ACT_SIGMOID, ACT_UNKNOWN } acti_type;

/**
 * @brief     Enumeration of layer type
 *            0. Input Layer type
 *            1. Fully Connected Layer type
 *            2. Output Layer type
 *            3. Unknown
 */
typedef enum { LAYER_IN, LAYER_FC, LAYER_OUT, LAYER_UNKNOWN } layer_type;

/**
 * @brief     type for the Optimizor to save hyper-parameter
 */
typedef struct {
  opt_type type;
  double learning_rate;
  double beta1;
  double beta2;
  double epsilon;
  acti_type activation;
} Optimizer;

/**
 * @class   Layer Base class for layers
 * @brief   Base class for all layers
 */
class Layer {
 public:
  /**
   * @brief     Destructor of Layer Class
   */
  virtual ~Layer(){};

  /**
   * @brief     Forward Propation of neural Network
   * @param[in] input Input Matrix taken by upper layer
   * @retval    Output Matrix
   */
  virtual Matrix forwarding(Matrix input) = 0;

  /**
   * @brief     Back Propation of neural Network
   * @param[in] input Input Matrix taken by lower layer
   * @param[in] iteration Epoch value for the ADAM Optimizer
   * @retval    Output Matrix
   */
  virtual Matrix backwarding(Matrix input, int iteration) = 0;

  /**
   * @brief     Initialize the layer
   *            - Weight(Height, Width), Bias(1, Width)
   * @param[in] b batch
   * @param[in] h Height
   * @param[in] w Width
   * @param[in] id index of this layer
   * @param[in] init_zero Bias initialization with zero
   */
  virtual void initialize(int b, int h, int w, int id, bool init_zero) = 0;

  /**
   * @brief     read layer Weight & Bias data from file
   * @param[in] file input file stream
   */
  virtual void read(std::ifstream &file) = 0;

  /**
   * @brief     save layer Weight & Bias data from file
   * @param[in] file output file stream
   */
  virtual void save(std::ofstream &file) = 0;

  /**
   * @brief     Optimizer Setter
   * @param[in] opt Optimizer
   */
  virtual void setOptimizer(Optimizer opt) = 0;

  /**
   * @brief     Layer type Setter
   * @param[in] type layer type
   */
  void setType(layer_type type) { this->type = type; }

  /**
   * @brief     Copy Layer
   * @param[in] l Layer to be copied
   */
  virtual void copy(Layer *l) = 0;

  /**
   * @brief     Input Matrix
   */
  Matrix Input;

  /**
   * @brief     Hidden Layer Matrix which store the
   *            forwading result
   */
  Matrix hidden;

  /**
   * @brief     Layer index
   */
  unsigned int index;

  /**
   * @brief     batch size of Weight Data
   */
  unsigned int batch;

  /**
   * @brief     width size of Weight Data
   */
  unsigned int width;

  /**
   * @brief     height size of Weight Data
   */
  unsigned int height;

  /**
   * @brief     Optimizer for this layer
   */
  Optimizer opt;

  /**
   * @brief     Boolean for the Bias to set zero
   */
  bool init_zero;

  /**
   * @brief     Layer type
   */
  layer_type type;

  /**
   * @brief     Activation function pointer
   */
  double (*activation)(double);

  /**
   * @brief     Activation Derivative function pointer
   */
  double (*activationPrime)(double);
};

/**
 * @class   Input Layer
 * @brief   Just Handle the Input of Network
 */
class InputLayer : public Layer {
 public:
  /**
   * @brief     Constructor of InputLayer
   */
  InputLayer(){};

  /**
   * @brief     Destructor of InputLayer
   */
  ~InputLayer(){};

  /**
   * @brief     No Weight data for this Input Layer
   */
  void read(std::ifstream &file){};

  /**
   * @brief     No Weight data for this Input Layer
   */
  void save(std::ofstream &file){};

  /**
   * @brief     It is back propagation of input layer.
   *            It return Input as it is.
   * @param[in] input input Matrix from lower layer.
   * @param[in] iteration Epoch Number for ADAM
   * @retval
   */
  Matrix backwarding(Matrix input, int iteration) { return Input; };

  /**
   * @brief     foward propagation : return Input Matrix
   *            It return Input as it is.
   * @param[in] input input Matrix from lower layer.
   * @retval    return Input Matrix
   */
  Matrix forwarding(Matrix input);

  /**
   * @brief     Set Optimizer
   * @param[in] opt optimizer
   */
  void setOptimizer(Optimizer opt);

  /**
   * @brief     Initializer of Input Layer
   * @param[in] b batch size
   * @param[in] h height
   * @param[in] w width
   * @param[in] id index of this layer
   * @param[in] init_zero boolean to set Bias zero
   */
  void initialize(int b, int h, int w, int id, bool init_zero);

  /**
   * @brief     Copy Layer
   * @param[in] l layer to copy
   */
  void copy(Layer *l);
};

/**
 * @class   FullyConnecedLayer
 * @brief   fully connected layer
 */
class FullyConnectedLayer : public Layer {
 public:
  /**
   * @brief     Constructor of Fully Connected Layer
   */
  FullyConnectedLayer(){};

  /**
   * @brief     Destructor of Fully Connected Layer
   */
  ~FullyConnectedLayer(){};

  /**
   * @brief     Read Weight & Bias Data from file
   * @param[in] file input stream file
   */
  void read(std::ifstream &file);

  /**
   * @brief     Save Weight & Bias Data to file
   * @param[in] file output stream file
   */
  void save(std::ofstream &file);

  /**
   * @brief     forward propagation with input
   * @param[in] input Input Matrix from upper layer
   * @retval    Activation(W x input + B)
   */
  Matrix forwarding(Matrix input);

  /**
   * @brief     back propagation
   *            Calculate dJdB & dJdW & Update W & B
   * @param[in] input Input Matrix from lower layer
   * @param[in] iteration Number of Epoch for ADAM
   * @retval    dJdB x W Matrix
   */
  Matrix backwarding(Matrix input, int iteration);

  /**
   * @brief     set optimizer
   * @param[in] opt Optimizer
   */
  void setOptimizer(Optimizer opt);

  /**
   * @brief     copy layer
   * @param[in] l layer to copy
   */
  void copy(Layer *l);

  /**
   * @brief     initialize layer
   * @param[in] b batch size
   * @param[in] h height
   * @param[in] w width
   * @param[in] id layer index
   * @param[in] init_zero boolean to set Bias zero
   */
  void initialize(int b, int h, int w, int id, bool init_zero);

 private:
  Matrix Weight;
  Matrix Bias;

  /**
   * @brief     First Momentum Matrix for the ADAM
   */
  Matrix M;

  /**
   * @brief     Second Momentum Matrix for the ADAM
   */
  Matrix V;
};

/**
 * @class   OutputLayer
 * @brief   OutputLayer (has Cost Function & Weight, Bias)
 */
class OutputLayer : public Layer {
 public:
  /**
   * @brief     Constructor of OutputLayer
   */
  OutputLayer(){};

  /**
   * @brief     Destructor of OutputLayer
   */
  ~OutputLayer(){};

  /**
   * @brief     Read Weight & Bias Data from file
   * @param[in] file input stream file
   */
  void read(std::ifstream &file);

  /**
   * @brief     Save Weight & Bias Data to file
   * @param[in] file output stream file
   */
  void save(std::ofstream &flle);

  /**
   * @brief     forward propagation with input
   * @param[in] input Input Matrix from upper layer
   * @retval    Activation(W x input + B)
   */
  Matrix forwarding(Matrix input);

  /**
   * @brief     back propagation
   *            Calculate dJdB & dJdW & Update W & B
   * @param[in] input Input Matrix from lower layer
   * @param[in] iteration Number of Epoch for ADAM
   * @retval    dJdB x W Matrix
   */
  Matrix backwarding(Matrix label, int iteration);

  /**
   * @brief     set optimizer
   * @param[in] opt Optimizer
   */
  void setOptimizer(Optimizer opt);

  /**
   * @brief     initialize layer
   * @param[in] b batch size
   * @param[in] h height
   * @param[in] w width
   * @param[in] id layer index
   * @param[in] init_zero boolean to set Bias zero
   */
  void initialize(int b, int w, int h, int id, bool init_zero);

  /**
   * @brief     get Loss value
   */
  double getLoss() { return loss; }

  /**
   * @brief     set cost function
   * @param[in] c cost function type
   */
  void setCost(cost_type c) { this->cost = c; };

  /**
   * @brief     copy layer
   * @param[in] l layer to copy
   */
  void copy(Layer *l);

 private:
  Matrix Weight;
  Matrix Bias;
  Matrix M;
  Matrix V;
  double loss;
  cost_type cost;
};
}

#endif
