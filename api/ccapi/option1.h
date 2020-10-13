// Layer.h (ccapi)
class Layer {
  public:
    template <typename... Args>
    Layer(Args... args) { impl = std::make_unique<LayerImpl>(args...); }
    virtual foo() {impl->foo()};
    virtual bar() = 0;

  private:
    std::unique_ptr<LayerImpl> impl;
};

// LayerImpl.h (internal)
class LayerImpl {
  public:
    LayerImpl(...);
    foo() { /** do actual work */ }

  private:
    // Actual data members
    // weights, etc
}

// FcLayer.h (ccapi)
class FCLayer : public Layer{
  public:
    template <typename... Args>
    FCLayer(Args... args) { impl = std::make_unique<FcLayerImpl>(args...); }
    foo() {impl->foo()}; // overloaded
    bar() {impl->bar()};

  private:
    std::unique_ptr<FCLayerImpl> impl;
}

// FcLayerImpl.h (internal)
class FcLayerImpl {
  public:
    template <typename... Args>
    FCLayerImpl(unsigned int unit_ = 0, Args... args) :
        LayerImpl(LayerType::LAYER_FC, args...),
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
  FClayer fc = FCLayer(3);
  fc->foo();
  fc->bar();

  model.addLayer(fc);
  model.addLayer(FCLayer(4));
}

// Note: different layers can have different interfaces
// Although I dont think its useful
