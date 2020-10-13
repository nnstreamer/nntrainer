// Layer.h (ccapi)
class Layer {
  public:
    Layer();
    virtual foo() = 0;
    virtual bar() = 0;
};

// LayerInternal.h (internal)
class LayerInternal : public Layer {
  public:
    LayerInternal();
    virtual foo() { /** do some work */ }

  private:
    // Actual data members
    // weights, etc
};

// layer_factory.h (ccapi)
std::unique_ptr<Layer> createLayer(string layer_type, std::vector<string> properties);

// layer_factory.cpp (internal)
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


// FcLayer.h (internal)
class FCLayer : public LayerInternal {
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


