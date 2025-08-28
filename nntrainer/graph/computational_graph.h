#ifndef __NNTRAINER_COMPUTATIONAL_GRAPH_H__
#define __NNTRAINER_COMPUTATIONAL_GRAPH_H__

#include <network_graph.h>

namespace nntrainer {

struct ComputationalGraphNode {
  LayerNode *node = nullptr;
  std::vector<ComputationalGraphNode *> inputs;
  std::vector<ComputationalGraphNode *> outputs;
  bool evaluated = false;
};

class ComputationalGraph {
public:
  void initialize(const NetworkGraph &network_graph);

  void serialize(const std::string &file_name);

private:
  void evaluateNode(ComputationalGraphNode *node,
                    ComputationalGraphNode *input_node);

  std::map<std::string, ComputationalGraphNode *> nodes_map_;
  std::vector<ComputationalGraphNode> nodes_;
  std::vector<ComputationalGraphNode *> input_nodes_;
};

} // namespace nntrainer

#endif // __NNTRAINER_COMPUTATIONAL_GRAPH_H__
