// Layer.h (ccapi)
class Layer {
  public:
    template <typename T>
    LayerImpl(T x) { impl = std::make_unique<T>(x); }
    foo() {impl->foo();}
    bar() {impl->bar();}

  protected:
    // Note: different from option1 where this is private
    std::unique_ptr<T> impl;
};

// layer_factory.h (ccapi) / (internal)
std::unique_ptr<Layer> createLayer(string layer_type, std::vector<string> properties);

// layer_factory.cpp (ccapi) / (internal)
#include <fclayer.h>
#include <conv2dlayer.h>
std::unique_ptr<Layer> createLayer(string layer_type, std::vector<string> properties) {
  // This can be case-insensitive comparison
  if (layer_type == "FullyConnected") {
    FCLayer fclayer = FCLayer();
    fclayer->setProperty(properties);
    return std::make_unique<Layer>(std::move(fclayer));
  } else if (layer_type == "Conv2D") {
    Conv2DLayer convlayer = Conv2DLayer();
    convlayer->setProperty(properties);
    return std::make_unique<Layer>(std::move(convlayer));
  } else
    throw std::invalid_argument("Unknown type for the layer");
  }
}

// LayerInternal.h (internal) - must have same interface as Layer.h
class LayerInternal {
  public:
    LayerInternal(...);
    virtual foo() { /** can do some work */ }
    virtual bar() = 0;

  private:
    // Actual data members
    // weights, etc
}

// FcLayer.h (internal)
class FCLayer : public LayerInternal {
  public:
    FCLayer() : LayerInternal(LayerType::LAYER_FC) {}
    foo() { /** do actual work */ }
    bar() { /** do actual work */ }

  private:
    // Actual data members
    // weights, etc
}

// app main.cpp
// custom layer by the user - must follow the signature of Layer
class CustomLayer {
  public:
    template <typename... Args>
    CustomLayer(unsigned int custom = 0, Args... args) :
        custom(custom) {}
    foo() { /** do actual work */ }
    bar() { /** do actual work */ }

  private:
    // Actual data members
    // weights, etc
}

int main() {
  Model model;
  std::unique_ptr<Layer> fc = createLayer("FullyConnected", {"unit=3"});
  fc->foo();
  fc->bar();

  model.addLayer(fc);
  model.addLayer(createLayer("FullyConnected", {"unit=3"}));
  model.addLayer(std::make_unique<Layer>(CustomLayer(2)));
}

// Note: different layers MUST have same interface

