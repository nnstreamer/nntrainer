#ifndef __NNTRAINER_COMPUTATIONAL_GRAPH_H__
#define __NNTRAINER_COMPUTATIONAL_GRAPH_H__

#include <network_graph.h>

namespace nntrainer {

struct ComputationalGraphNode {
  std::shared_ptr<LayerNode> node = nullptr;
  std::set<ComputationalGraphNode *> inputs;
  std::set<ComputationalGraphNode *> outputs;
  bool evaluated = false;
  std::vector<int> in_orders;
  int order = 0;
};

class ComputationalGraph {
public:
  void initialize(const NetworkGraph &network_graph);

  void serialize(const std::string &file_name);

  void topologicalSort();

  sharedConstTensors forwarding(
    bool training,
    std::function<void(std::shared_ptr<LayerNode>, bool, SynchronizationInfo *)>
      forwarding_op);

private:
  void evaluateNode(ComputationalGraphNode *node,
                    ComputationalGraphNode *input_node);

  std::map<std::string, ComputationalGraphNode *> nodes_map_;
  std::vector<ComputationalGraphNode> nodes_;
  std::vector<ComputationalGraphNode *> input_nodes_;
  std::vector<ComputationalGraphNode *> output_nodes_;
  std::vector<ComputationalGraphNode *> sorted_nodes_;
};

} // namespace nntrainer

#endif // __NNTRAINER_COMPUTATIONAL_GRAPH_H__
