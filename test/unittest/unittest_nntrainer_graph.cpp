#include <functional>
#include <iostream>
#include <memory>
#include <neuralnet.h>
#include <string>
#include <unistd.h>
#include <vector>

using namespace nntrainer;

class Node {
public:
  Node() {}

  Node(std::shared_ptr<Layer> layer) :
    l(layer),
    input_dim(layer->getInputDimension()),
    output_dim(layer->getOutputDimension()) {
    int param_size = layer->getParamsize();

    weight_dim.resize(param_size);
    expected_weight.resize(param_size);
    expected_grad.resize(param_size);

    for (int i = 0; i < param_size; ++i) {
      weight_dim[i] = layer->paramsAt(i).weight.getDim();
    }

    expected_input = Tensor(input_dim);
    for (unsigned int i = 0; i < weight_dim.size(); ++i) {
      expected_weight[i] = Tensor(weight_dim[i]);
      expected_grad[i] = Tensor(weight_dim[i]);
    }
    expected_output = Tensor(output_dim);
    expected_dx = Tensor(input_dim);
  }

  void read(std::ifstream &in) {
    expected_input.read(in);
    for (unsigned int i = 0; i < weight_dim.size(); ++i) {
      expected_weight[i].read(in);
    }
    for (unsigned int i = 0; i < weight_dim.size(); ++i) {
      expected_grad[i].read(in);
    }
    expected_output.read(in);
    expected_dx.read(in);
  }

  void prepare_weights() {
    for (int i = 0; i < l->getParamsize(); ++i) {
      l->paramsAt(i).weight = expected_weight[i].clone();
    }
    l->setTrainable(false);
  }

  void verify(Tensor a, Tensor b, const std::string &error_msg) {
    if (a != b) {
      std::cout
        << "============================================================\n";
      std::cout << "current " << a << "expected " << b;
      throw std::invalid_argument(error_msg.c_str());
    }
  }

  void verify_weight(const std::string &error_msg) {
    if (expected_weight.size() != (unsigned int)l->getParamsize())
      throw std::invalid_argument(error_msg.c_str());

    for (int i = 0; i < l->getParamsize(); ++i) {
      verify(l->paramsAt(i).weight, expected_weight[i],
             error_msg + " " + l->paramsAt(i).name + "weight");
    }
  }

  void verify_grad(const std::string &error_msg) {
    if (expected_weight.size() != (unsigned int)l->getParamsize())
      throw std::invalid_argument(error_msg.c_str());

    for (int i = 0; i < l->getParamsize(); ++i) {
      verify(l->paramsAt(i).grad, expected_grad[i],
             error_msg + " " + l->paramsAt(i).name + "grad");
    }
  }

  sharedConstTensor forward(sharedConstTensor in, int iteration) {
    std::stringstream ss;
    ss << "forward failed at " << l->getName() << " at iteration " << iteration;
    std::string err_msg = ss.str();

    verify(*in, expected_input, err_msg + " at intput ");
    verify_weight(err_msg);
    sharedConstTensor out = l->forwarding(in);
    verify_weight(err_msg);
    verify(*out, expected_output, err_msg + " at output ");
    return out;
  }

  sharedConstTensor loss_forward(sharedConstTensor pred,
                                 sharedConstTensor answer, int iteration) {
    std::stringstream ss;
    ss << "loss failed at " << l->getName() << " at iteration " << iteration;
    std::string err_msg = ss.str();

    verify(*pred, expected_input, err_msg);
    sharedConstTensor out =
      std::static_pointer_cast<LossLayer>(l)->forwarding(pred, answer);
    verify(*out, expected_output, err_msg);

    return out;
  }

  float getLoss() { return l->getLoss(); }

  sharedConstTensor backward(sharedConstTensor deriv, int iteration) {
    std::stringstream ss;
    ss << "backward failed at " << l->getName() << " at iteration "
       << iteration;
    std::string err_msg = ss.str();

    sharedConstTensor out = l->backwarding(deriv, iteration);
    verify(*out, expected_dx, err_msg);
    verify_grad(err_msg);

    l->getOptimizer().apply_gradients(l->getParams(), l->getParamsize(),
                                      iteration);
    return out;
  }

  std::shared_ptr<Layer> l;
  std::vector<Tensor> expected_weight;
  std::vector<TensorDim> weight_dim;
  std::vector<Tensor> expected_grad;
  TensorDim input_dim;
  TensorDim output_dim;
  Tensor expected_dx;
  Tensor expected_input;
  Tensor expected_output;
};

class Graph {
public:
  Graph(const std::string &file_path) {
    file.open(file_path, std::ios_base::in | std::ios_base::binary);
  }

  ~Graph() { file.close(); }

  void init(const std::string &config, TensorDim labelDim) {
    NeuralNetwork nn(config);
    nn.loadFromConfig();
    nn.init();

    initial_input = Tensor(nn.getGraph().front()->getInputDimension());
    label = Tensor(labelDim);

    initial_input.read(file);
    label.read(file);

    for (auto i : nn.getGraph()) {
      Node node(i);
      nodes.push_back(node);
    }
  }

  void read_epoch() {
    for (auto &item : nodes) {
      item.read(file);
    }
    file.read((char *)&expected_loss, sizeof(float));
  }

  void start_validation(int epochs) {

    read_epoch();
    for (auto &item : nodes) {
      item.prepare_weights();
    }

    for (int iteration = 0; iteration < epochs; ++iteration) {
      if (iteration != 0)
        read_epoch();
      sharedConstTensor input = MAKE_SHARED_TENSOR(initial_input);
      sharedConstTensor lb = MAKE_SHARED_TENSOR(label);

      for (unsigned int i = 0; i < nodes.size() - 1; i++)
        input = nodes[i].forward(input, iteration);

      input = nodes.back().loss_forward(input, lb, iteration);

      if (fabs(expected_loss - nodes.back().getLoss()) > Tensor::epsilon) {
        std::stringstream ss;
        ss << "validation failed at " << iteration
           << " reason: loss does not match";
        std::cout << "current : " << nodes.back().getLoss() << "expected "
                  << expected_loss << std::endl;
        throw std::invalid_argument(ss.str().c_str());
      }

      sharedConstTensor output = lb;
      for (unsigned int i = nodes.size() - 1; i > 0; i--)
        output = nodes[i].backward(output, iteration);
    }
  }

  std::ifstream file;
  Tensor label;
  Tensor initial_input;

  float expected_loss;

  NeuralNetwork network;
  std::vector<Node> nodes;
};

int main() {
  // please change this path
  chdir("/data/nntrainer/test/unittest");

  // please change this path
  Graph g("/data/nntrainer/test/input_gen/a.info");
  g.init("model.ini", TensorDim(3, 1, 1, 10));

  g.start_validation(2);

  std::cout << "hello world" << std::endl;
}
