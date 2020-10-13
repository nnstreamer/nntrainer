// Layer.h (ccapi) - pure virtual
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

// layer_factory.h (ccapi)
#include <fclayer.h>
#include <conv2dlayer.h>
template <typename... Args>
std::unique_ptr<Layer> createLayer(LayerType type, Args... args) {
  switch (type) {
  case LayerType::LAYER_FC:
    return std::make_unique<Layer>(FClayer(args...));
  case LayerType::LAYER_CONV2D:
    return std::make_unique<Layer>(Conv2Dlayer(args...));
  default:
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
    template <typename... Args>
    FCLayer(unsigned int unit_ = 0, Args... args) :
        LayerInternal(LayerType::LAYER_FC, args...),
        unit(unit_) {}
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
  std::unique_ptr<Layer> fc = CreateLayer(LayerType::LAYER_FC, 3);
  fc->foo();
  fc->bar();

  model.addLayer(fc);
  model.addLayer(createLayer(LayerType::LAYER_FC, 3));
  model.addLayer(std::make_unique<Layer>(CustomLayer(2)));
}

// Note: different layers MUST have same interface
