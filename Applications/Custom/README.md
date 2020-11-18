# Application with Custom Object

Applications inside this folder are dedicated to demonstrate how to create custom layers, optimizers or other supported objects.

There are two ways to apply custom object to the code.

1. Create an object as a part of client code and register it to NNTrainer on the client code.
  Easy to write, less portable
2. Create an object as a dynamic library and NNTrainer load the dynamic library.
  Portable, more configurations


## Structure of the folder

`*Client` demonstrates how to generate an object inside a client code and register on the fly.
For example, `LayerClient` will demo about how to write a custom layer and register it.

`*Plugin` demonstrates how to generate an object inside a plugin code and register on the fly.
For example, `OptimizerPlugin` will demo about how to create a custom optimizer as a pluggable library.
