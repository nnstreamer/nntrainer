# Multi_Input example

- This example demonstrates how to use the `multi_input` layer. 
- The NNTrainer supports a network that takes multiple tensors as inputs. 
- Users can create multiple `input` layers for the network with their own names and build the network accordingly. 
- This code includes an example of training with...

```
                       +-----------+
                       |  output   |
                       +-----------+
                              |                  
    +---------------------------------------------------+  
    |                      flatten                      |
    +---------------------------------------------------+  
                              |                   
    +---------------------------------------------------+  
    |                      concat0                      |
    +---------------------------------------------------+  
        |                     |                  |
    +-----------+       +-----------+       +-----------+  
    |  input 2  |       |  input 1  |       |  input 0  |  
    +-----------+       +-----------+       +-----------+   

```

- **[Note]** Users should feed the multi-input in reverse order because the model is structured in a reversed manner internally. This is a known issue for us, and we plan to address it soon.