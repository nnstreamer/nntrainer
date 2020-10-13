// Layer.h (ccapi)
class Layer {
  public:
    Layer(....);
    virtual foo();
    virtual bar() = 0;

  private:
    // Actual data members
    // weights, etc
};

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
int main() {
  Model model;
  std::unique_ptr<Layer> fc = createLayer(LayerType::LAYER_FC, 3);
  fc->foo();
  fc->bar();

  model.addLayer(fc);
  model.addLayer(createLayer(LayerType::LAYER_FC, 3));
}

// Note: different layers MUST have same interface
