# Application with Custom Object

Applications inside this folder are dedicated to demonstrate how to create custom layers, optimizers or other supported objects.

There are two ways to apply custom object to the code.

1. **Client**: Create an object as a part of client code and register it to NNTrainer on the client code.
  Easy to write, less portable
2. **Plugin**: Create an object as a dynamic library and NNTrainer load the dynamic library.
  Portable, more configurations

## List of Available Examples

### Layer

#### Related Folders:

1. `LayerClients/` has a demo about how to write a custom layer and register it.
2. `LayerPlugin/` contains test and build scripts to show how to build a custom layer and object as a seperate library(`.so`) and register it.

#### Related Custom Objects:


1. **Pow Layer**: A custom layer object which get x -> returns x^(exponent), exponent can be set by `layer.setProperty({"exponent={n}"});`.
2. **Mean Absoulute Error(MAE) Loss Layer**: A loss layer object which calculates MAE.
