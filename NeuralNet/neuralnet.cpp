#include "include/neuralnet.h"
#include "iniparser.h"
#include <assert.h>
#include <cmath>
#include <sstream>
#include <stdio.h>

namespace Network {

std::vector<std::string> Optimizer_string = {"sgd", "adam"};
std::vector<std::string> Cost_string = {"msr", "logistic"};
std::vector<std::string> NetworkType_string = {"knn", "regression",
                                               "neuralnet"};
std::vector<std::string> activation_string = {"tanh", "sigmoid"};
std::vector<std::string> layer_string = {"InputLayer", "FullyConnectedLayer",
                                         "OutputLayer"};

static bool is_file_exist(std::string filename) {
  std::ifstream infile(filename);
  return infile.good();
}

std::vector<std::string> parseLayerName(std::string ll) {
  std::vector<std::string> ret;
  std::istringstream ss(ll);
  do {
    std::string word;
    ss >> word;
    if (word.compare("") != 0)
      ret.push_back(word);
  } while (ss);

  return ret;
}

unsigned int parseType(std::string ll, input_type t) {
  int ret;
  unsigned int i;

  switch (t) {
  case TOKEN_OPT:
    for (i = 0; i < Optimizer_string.size(); i++) {
      if (Optimizer_string[i].compare(ll) == 0) {
        return (i);
      }
    }
    ret = i - 1;
    break;
  case TOKEN_COST:
    for (i = 0; i < Cost_string.size(); i++) {
      if (Cost_string[i].compare(ll) == 0) {
        return (i);
      }
    }
    ret = i - 1;
    break;
  case TOKEN_NET:
    for (i = 0; i < NetworkType_string.size(); i++) {
      if (NetworkType_string[i].compare(ll) == 0) {
        return (i);
      }
    }
    ret = i - 1;
    break;
  case TOKEN_ACTI:
    for (i = 0; i < activation_string.size(); i++) {
      if (activation_string[i].compare(ll) == 0) {
        return (i);
      }
    }
    ret = i - 1;
    break;
  case TOKEN_LAYER:
    for (i = 0; i < layer_string.size(); i++) {
      if (layer_string[i].compare(ll) == 0) {
        return (i);
      }
    }
    ret = i - 1;
    break;
  case TOKEN_UNKNOWN:
  default:
    ret = 3;
    break;
  }
  return ret;
}

NeuralNetwork::NeuralNetwork(std::string config) { this->config = config; }

void NeuralNetwork::setConfig(std::string config) { this->config = config; }

void NeuralNetwork::init() {
  int w, h, id;
  bool b_zero;
  std::string l_type;
  Layers::layer_type t;
  std::string inifile = config;
  dictionary *ini = iniparser_load(inifile.c_str());

  if (ini == NULL) {
    fprintf(stderr, "cannot parse file: %s\n", inifile.c_str());
  }

  nettype = (Network::net_type)parseType(
      iniparser_getstring(ini, "Network:Type", NULL), TOKEN_NET);
  std::vector<std::string> layers_name =
      parseLayerName(iniparser_getstring(ini, "Network:Layers", NULL));
  learning_rate = iniparser_getdouble(ini, "Network:Learning_rate", 0.0);
  opt.learning_rate = learning_rate;
  epoch = iniparser_getint(ini, "Network:Epoch", 100);
  opt.type = (Layers::opt_type)parseType(
      iniparser_getstring(ini, "Network:Optimizer", NULL), TOKEN_OPT);
  opt.activation = (Layers::acti_type)parseType(
      iniparser_getstring(ini, "Network:Activation", NULL), TOKEN_ACTI);
  cost = (Layers::cost_type)parseType(
      iniparser_getstring(ini, "Network:Cost", NULL), TOKEN_COST);

  model = iniparser_getstring(ini, "Network:Model", "model.bin");
  batchsize = iniparser_getint(ini, "Network:minibatch", 1);

  opt.beta1 = iniparser_getdouble(ini, "Network:beta1", 0.0);
  opt.beta2 = iniparser_getdouble(ini, "Network:beta2", 0.0);
  opt.epsilon = iniparser_getdouble(ini, "Network:epsilon", 0.0);

  for (unsigned int i = 0; i < layers_name.size(); i++)
    std::cout << layers_name[i] << std::endl;

  // std::cout << learning_rate<< " " << epoch << " " << opt.type<< " " <<
  // opt.activation<< " " << cost << " " << model << " " << batchsize<< " \n";

  loss = 100000.0;

  for (unsigned int i = 0; i < layers_name.size(); i++) {
    l_type = iniparser_getstring(ini, (layers_name[i] + ":Type").c_str(), NULL);
    t = (Layers::layer_type)parseType(l_type, TOKEN_LAYER);
    w = iniparser_getint(ini, (layers_name[i] + ":Width").c_str(), 1);
    h = iniparser_getint(ini, (layers_name[i] + ":Height").c_str(), 1);
    id = iniparser_getint(ini, (layers_name[i] + ":Id").c_str(), 0);
    b_zero = iniparser_getboolean(ini, (layers_name[i] + ":Bias_zero").c_str(),
                                  true);
    std::cout << l_type << " " << t << " " << w << " " << b_zero << " " << id
              << std::endl;
    switch (t) {
    case Layers::LAYER_IN: {
      Layers::InputLayer *inputlayer = new (Layers::InputLayer);
      inputlayer->setType(t);
      inputlayer->initialize(batchsize, h, w, id, b_zero);
      inputlayer->setOptimizer(opt);
      layers.push_back(inputlayer);
    } break;
    case Layers::LAYER_FC: {
      Layers::FullyConnectedLayer *fclayer = new (Layers::FullyConnectedLayer);
      fclayer->setType(t);
      fclayer->initialize(batchsize, h, w, id, b_zero);
      fclayer->setOptimizer(opt);
      layers.push_back(fclayer);
    } break;
    case Layers::LAYER_OUT: {
      Layers::OutputLayer *outputlayer = new (Layers::OutputLayer);
      outputlayer->setType(t);
      outputlayer->initialize(batchsize, h, w, id, b_zero);
      outputlayer->setOptimizer(opt);
      layers.push_back(outputlayer);
    } break;
    case Layers::LAYER_UNKNOWN:
      break;
    default:
      break;
    }
  }

  iniparser_freedict(ini);
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

void NeuralNetwork::saveModel() {
  std::ofstream modelFile(model, std::ios::out | std::ios::binary);
  for (unsigned int i = 0; i < layers.size(); i++)
    layers[i]->save(modelFile);
  modelFile.close();
}

void NeuralNetwork::readModel() {
  if (!is_file_exist(model))
    return;
  std::ifstream modelFile(model, std::ios::in | std::ios::binary);
  for (unsigned int i = 0; i < layers.size(); i++)
    layers[i]->read(modelFile);
  modelFile.close();
  std::cout << "read model file \n";
}
}
