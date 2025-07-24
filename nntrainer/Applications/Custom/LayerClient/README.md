# LayerClient Example

This application demonstrates how to create a custom layer and register inside client-side code

# How to

1. build with ninja

2. run with ini path or argument `model`

```bash
$ Applications/Custom/LayerClient/jni/layer_client model
pow layer forward is called
input: <N9nntrainer6TensorE at 0x55d4521c75e0>
Shape: 10:1:1:100
[2 2 2 ... 2 2 2]
output: <N9nntrainer6TensorE at 0x55d4521cb7f0>
Shape: 10:1:1:100
[8 8 8 ... 8 8 8]
Press enter key to continue...
pow layer backward is called
input: <N9nntrainer6TensorE at 0x55d4521cb7f0>
Shape: 10:1:1:100
[0.00556926 0.000204645 -0.00607976 ... 0.022363 0.020753 0.00855796]
output: <N9nntrainer6TensorE at 0x55d4521c75e0>
Shape: 10:1:1:100
[0.0167078 0.000613935 -0.0182393 ... 0.067089 0.0622591 0.0256739]
Press enter key to continue...
pow layer forward is called ]  ( Training Loss: 0 )
input: <N9nntrainer6TensorE at 0x55d4521c75e0>
Shape: 10:1:1:100
[2 2 2 ... 2 2 2]
output: <N9nntrainer6TensorE at 0x55d4521cb7f0>
Shape: 10:1:1:100
[8 8 8 ... 8 8 8]
Press enter key to continue...

pow layer backward is called
input: <N9nntrainer6TensorE at 0x55d4521cb7f0>
Shape: 10:1:1:100
[0.0045707 -0.00549962 0.014237 ... -0.0178485 -0.0163203 -0.00993024]
output: <N9nntrainer6TensorE at 0x55d4521c75e0>
Shape: 10:1:1:100
[0.0137121 -0.0164989 0.0427111 ... -0.0535455 -0.0489608 -0.0297907]
Press enter key to continue...

pow layer forward is called ]  ( Training Loss: 0 )
input: <N9nntrainer6TensorE at 0x55d4521c75e0>
Shape: 10:1:1:100
[2 2 2 ... 2 2 2]
output: <N9nntrainer6TensorE at 0x55d4521cb7f0>
Shape: 10:1:1:100
[8 8 8 ... 8 8 8]
Press enter key to continue...
```
