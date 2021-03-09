---
title: Memory Management
...

# Memory Management

Memory management for a model consists of multiple parts including:

- **Weights** which make the model
- **Input, Output, and Label tensors** for the model
- **Variables** required to run the model like intermediate inputs and outputs, derivatives, and gradients.

The management for the above 3 kinds of memory is done depending on the mode of execution of the model, namely `inference` or `training` mode. The memory management of these tensors is performed by the `Manager` inside `NNTrainer`.

The memory for the `weights` are allocated at the time of initializing the model, where the `weights` are either initialized using the provided initializer or loaded from the saved model file. The memory allocated for `weights` is freed
upon `destruction` of the model object containing the `weights`.

The memory for the `Input, Output and Label` tensors and `Variables` is allocated lazily, and de-allocated once its usage is finished.

- During `training`, the memory is allocated lazily at the first iteration and then reused for the rest of the training. This memory is freed once the training finishes. The next subsequent call to training will re-allocate the memory again.
- During `inference`, the memory is allocated lazily right before the inference. The memory can be retained for the next iteration (for burst inference) and then freed later, or freed at the end of the inference with the arguments passed to the `inference`.

The next sections discuss the optimizations performed to reduce the memory requirements of various parts of the model. Most of the memory optimizations have no effect on the performance of the model but reduces the debug information available from the model. These optimizations can be turned off for debugging purposes.

## Weights

The memory for the weights is allocated at the time of initialization of the model. The weight memory is available once the graph of the model is made and the nodes (representing layer operations) in the graph are finalized. The memory required by the nodes (layer) to represent their parameters is allocated as the memory for weights. The weights for each node (layer) are allocated separately.

The memory allocation for weights is done independently of the mode of execution for the model. The total memory allocated for weights will be equal or smaller than the saved model binary size (as saved model binary size can include training-related parameters).

## Input, Output, and Label Tensors

The input, output, and label tensors of the model are managed differently based on the mode of execution of the model.

- In `training` mode, `Manager` allocates the memory of the input, output, and the label tensors preemptively. These pre-allocated tensors are connected with the dataset and reused in each iteration for memory-efficient training.
- In `inference` mode, as both input and label are allocated and provided by the user, `Manager` preemptively allocates the memory only for output tensors of the model. The input and label tensors provided by the users are directly used for inference. Note that the input and label provided by the user must remain valid during the `inference` call to ensure correctness of the code.

## Gradients, Intermediates, and Derivatives

The variables required for executing the model include intermediate inputs and outputs between the layers, and gradients/derivatives depending on the mode of execution. The optimization strategy for allocating these variables also depends on the mode of the execution.

### Gradients

Gradients are allocated only while `training` the model. The memory required by the gradients is optimized by the `Manager` by using a shared tensor. A shared tensor with the size of the maximum memory required by a weight is allocated, and all the gradients use this shared tensor for their memory requirements.

This optimization comes with the cost that each weight must be updated by its gradient before calculating the gradient for the next weight. Further, the gradient of an individual weight is not available after an iteration.

If multiple weights in a layer require all their gradients to be calculated before being applied, the optimization can increase the size of the shared tensor to the size of maximum memory required by all the weights of a layer in the model (This is not yet supported).

### Intermediate Inputs and Outputs

The allocations for input and output tensors between layers are optimized depending on the execution mode of the model.

- For `inference`, the total amount of memory allocated is limited to the maximum amount of memory required to execute a single layer. This includes the combined memory requirement for all the input(s) and output(s) of a layer. The total memory requirement of the model is reduced to the memory requirement of the largest layer. Note that with this optimization enabled, the results of intermediate layers are not available.

- For `training`, the output(s) of a layer `i` and the input(s) of the next connected layer(s) `i+1` share the same memory. This adds a constraint on the layer to not modify their inputs (this might impact performance in some scenarios but we did not observe any).

### Derivatives

Derivatives are allocated only while `training` the model. The memory allocated for the derivative is dependent on the type of the layer:

- For layers excluding the activation layers, the memory for the `derivatives` is shared with the memory allocated for intermediate inputs/outputs. This is based on the observation that the back-propagation of most of the layers requires only the inputs and the incoming derivatives from the next layers. So, sharing the memory of the incoming derivative and output of this layer overwrites the output of the current layer and reduces the memory requirements of a layer.

- For activation layers, the memory required by the `derivatives` is optimized by using a shared tensor (just like Gradients). The size of the shared tensor is the size of the maximum memory required by the derivative of any activation layer. This optimization comes with the cost that the derivative of activation layers is not available for debugging at the end of an iteration.

Sharing the memory of the derivatives and intermediate inputs/outputs can also be performed for activation layers but it affects their runtime performance as the back-propagation operation for most of the activation layers can be speedup by using the output of the layer than the input of the layer.

# Memory Optimizations using In-Place Layer Operations

Certain layers can operate in-place, allowing more memory optimizations to be performed. Layers performing in-place operations do not require memory for storing their input(s), and their output(s) overwrite their inputs. The back-propagation operation for such layers also must not require their inputs.

- Batch Normalization layer can operate in-place. BN layer stores processed input rather than the original input to achieve higher performance. This allows the back-propagation operation to work without inputs, and for BN layer to work in-place.

- Various activation Layers can also operate in-place (limited to `ReLU`, `Sigmoid`, `TanH`, etc. operations that work on individual elements and do not depend on their neighboring values). These activation layers do not need to store their inputs and rather depend on their outputs for faster performance. Such layers are optimized to work in-place. Note that derivatives for activation layers do not overwrite their outputs as explained in the previous section.

- Flatten layer also operates in-place as it does not process the data but only changes the representation of the data by modifying its shape.

The in-place optimization has a limitation on the locations where it can be applied. Consecutive layers cannot be optimized to work in-place.
