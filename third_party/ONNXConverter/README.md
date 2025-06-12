
# ONNX Converter

## Essential background knowledge for developing ONNX Converter

The graph compilation-related parts can be found in the `compile` function of both `nntrainer/models/neuralnet.cpp` and `nntrainer/graph/network_graph.cpp` scripts.

The compilation process is divided into two main stages, which we'll refer to as `Stage 1` and `Stage 2`.

### Stage 1 (compile function in the neuralnet.cpp script)

#### ExecutionMode & ExecutionOrder
```cpp
int NeuralNetwork::compile(ExecutionMode mode) {
    exec_mode = mode;
    ...
}
```
The ExecutionMode parameter distinguishes between `train` and `inference` modes. This information is later used to determine memory allocation (e.g., whether to allocate space for gradients) and graph execution order (e.g., whether to execute the backward pass `[Training] forwarding of Layer1 → forwarding of Layer2 → Loss calculation → backwarding of Layer2 → backwarding of Layer1, [Inference] forwarding of Layer1 → forwarding of Layer2`).

#### Realizer
```cpp
std::vector<std::unique_ptr<GraphRealizer>> realizers;

realizers.emplace_back(new PreviousInputRealizer(
  std::vector<Connection>(input_conn.begin(), input_conn.end())));
realizers.emplace_back(new MultioutRealizer());
realizers.emplace_back(new FlattenRealizer());
realizers.emplace_back(new ActivationRealizer());

for (auto &realizer : realizers) {
  graph_representation = realizer->realize(graph_representation);
}
```
Layers can be created by receiving necessary meta information (properties) from users, but certain meta information needs to be finalized considering the connection relationships within the graph. For example (not real API):
```cpp
creatLayer(type="fc layer", name="fc1");
creatLayer(type="fc layer", name="fc2");
```
Here, since the user didn't specify the input layer for "fc2", during compilation, the system determines that "fc1" is the input layer for "fc2" based on the creation order. This role is performed by `PreviousInputRealizer`. 

#### FSU
```cpp
bool fsu = std::get<props::Fsu>(model_flex_props);
const std::string fsu_path = std::get<props::FsuPath>(model_flex_props);
unsigned int lookahead = std::get<props::FsuLookahead>(model_flex_props);
```
FSU optimizes memory usage by not uploading all model weights to memory at once, but rather uploading only the necessary portions sequentially. Beyond simply loading the weights of the current layer, it pre-loads the weights of the next N(lookahead) layers to minimize speed degradation while conserving memory.

#### Tensor format & Tensor type
```cpp
const std::string tensor_format =
  to_string(std::get<props::TensorFormat>(model_flex_props));

const std::string tensor_type =
  to_string(std::get<props::ModelTensorDataType>(model_flex_props));
```
The tensor format determines wheter tensors are processed in `NCHW` or `NHWC` format. ModelTensorDataType receives weight and activation types as paired value (e.g., "FP16-FP32").

Based on this information, a network_graph object is created. The process then moves to `Stage 2` by calling the compile function of the network_graph object.
### Stage 2 (compile function in the network_graph.cpp script)
#### Realizer
```cpp
  try {
    setOutputConnections();
  } catch (std::exception &e) {
    ml_loge("setting output layer failed, reason: %s", e.what());
    return ML_ERROR_INVALID_PARAMETER;
  }

  graph.realizeInputOutputNode();
  ```
  Unlike Stage 1, this stage doesn't use the Realizer class but still performs realization tasks. Using the input information from `Stage 1`, each layer could identify its output layer as well. This completes the mutual connection between input and output layers.

#### Graph Sorting & Execution Order
```cpp
graph.topologicalSort();
setExecutionOrder();
forward_iter_end = (*(cend() - 1)).get();
```

Then we performs topological sorting on the connected graph. The actual execution order of the graph is determined at this stage.
Subsequently, we calculates the execution order for `forward` and `backward` passes(specifically, breaking down the steps of `calc derivative`, `calc gradient` and `apply gradient`) using the sorted graph information to optimize memory management.

#### Inplace
```cpp
inPlaceOptimize();
```
Finally, if the Inplace option is enabled by the user and the operation supports Inplace operation, we configure the operation accordingly. This involves the case where multiple layers are connected after a single layer, to determine Inplace feasibility.

---
After these stages, the primary graph compilation process is complete. However, in partice, each layer's implementation includes a `finalize` function that performs additional tasks, such as verifying input/output layer dimensions and calculating size of weights. All such elements must be considered to complete the graph information for execution. Therefore, these factors should be accounted for in the Offline ONNX Converter's design to properly handle all pre-compilation process to handle all pre-calculable and optimizable elements.
Further details will be continuously updated in this document.