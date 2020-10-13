// Layer.h (ccapi)
class Layer {
  public:
    template <typename... Args>
    Layer(Args... args) { std::make_unique<LayerImpl>(args...); }
    virtual foo() {impl->foo();}
    virtual bar() = 0;

  protected:
    // Note: different from option1 where this is private
    // Do not need to include LayerImpl for in this file
    class LayerImpl;
    std::unique_ptr<LayerImpl> impl;
};

// LayerImpl.h (internal)
class LayerImpl {
  public:
    LayerImpl(....);
    foo() { /** do actual work */ }

  private:
    // Actual data members
    // weights, etc
}

// layer_factory.h (ccapi)
#include <fclayer.h>
#include <conv2dlayer.h>
template <typename... Args>
std::unique_ptr<Layer> createLayer(LayerType type, Args... args) {
  switch (type) {
  case LayerType::LAYER_FC:
    return std::make_unique<FullyConnectedLayer>(args...);
  case LayerType::LAYER_CONV2D:
    return std::make_unique<Conv2DLayer>(args...);
  default:
    throw std::invalid_argument("Unknown type for the layer");
  }
}

// FcLayer.h (internal)
class FCLayer : public Layer {
  public:
    template <typename... Args>
    FCLayer(unsigned int unit_ = 0, Args... args) :
        Layer(LayerType::LAYER_FC, args...),
        unit(unit_) {}
    foo() { /** do actual work */ }
    bar() { /** do actual work */ }

  private:
    // Actual data members
    // weights, etc
}

// app main.cpp
// custom layer by the user - must follow the signature of Layer
class CustomLayer : public Layer {
  public:
    template <typename... Args>
    CustomLayer(unsigned int custom = 0, Args... args) :
        Layer(LayerType::LAYER_CUSTOM, args...),
        custom(custom) {}
    foo() { /** do actual work */ }
    bar() { /** do actual work */ }

  private:
    // Actual data members
    // weights, etc
}

int main() {
  Model model;
  std::unique_ptr<Layer> fc = createLayer(LayerType::LAYER_FC, 3);
  fc->foo();
  fc->bar();

  model.addLayer(fc);
  model.addLayer(createLayer(LayerType::LAYER_FC, 3));
  model.addLayer(std::make_unique<CustomLayer>(2));
}

// Note: different layers MUST have same interface
