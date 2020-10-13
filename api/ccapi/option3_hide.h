// Layer.h (ccapi)
class Layer {
  public:
    Layer() { std::make_unique<LayerImpl>(); }
    virtual foo() {impl->foo();}
    virtual bar() {impl->foo()};

  protected:
    // Note: different from option1 where this is private
    // Do not need to include LayerImpl for in this file
    class LayerImpl;
    std::unique_ptr<LayerImpl> impl;
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

// LayerImpl.h (internal)
class LayerImpl {
  public:
    LayerImpl();
    virtual foo() { /** implement */ }
    virtual bar() = 0;

  private:
    // Actual data members
    // weights, etc
}

// FcLayer.h (internal)
class FCLayer : public LayerImpl {
  public:
    FCLayer() : Layer(LayerType::LAYER_FC) {}
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
    CustomLayer() : Layer(LayerType::LAYER_CUSTOM) {}
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

  CustomLayer customlayer = CustomLayer();
  customlayer->setProperty({"custom=2"});
  model.addLayer(std::make_unique<Layer>(std::move(customlayer)));
}

// Note: different layers MUST have same interface

