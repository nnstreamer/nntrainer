// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   time_dist.h
 * @date   01 April 2021
 * @brief  This is Time Distributed Layer Class of Neural Network
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __TIME_DIST_H__
#define __TIME_DIST_H__
#ifdef __cplusplus

#include <layer_devel.h>
#include <weight.h>

namespace nntrainer {

/**
 * @class   TimeDistLayer
 * @brief   Time Distribution Layer
 */
class TimeDistLayer : public Layer {
public:
  /**
   * @brief     Constructor of Time Distribution Layer
   */
  TimeDistLayer() : Layer() {
    for (unsigned int i = 0; i < 4; ++i) {
      positions[i] = nullptr;
    }
  }

  /**
   * @brief     Destructor of Time Distributed Layer
   */
  ~TimeDistLayer() = default;

  /**
   *  @brief  Move constructor.
   *  @param[in] TimeDistLayer &&
   */
  TimeDistLayer(TimeDistLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs TimeDistLayer to be moved.
   */
  TimeDistLayer &operator=(TimeDistLayer &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(RunLayerContext &context) override;

  /**
   * @copydoc Layer::calcGradient(RunLayerContext &context)
   */
  void calcGradient(RunLayerContext &context) override;

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override {
    dist_layer->exportTo(exporter, method);
  }

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return TimeDistLayer::type; };

  /**
   * @copydoc Layer::supportBackwarding()
   */
  bool supportBackwarding() const override {
    return dist_layer->supportBackwarding();
  }

  /**
   * @copydoc Layer::setBatch(RunLayerContext &context, unsigned int batch)
   */
  void setBatch(RunLayerContext &context, unsigned int batch) override;

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  void setProperty(const std::vector<std::string> &values) override {
    /**
     * @note assumption: name of the dist_layer is set via setName() and not
     * with setProperty()
     */
    if (!values.empty())
      dist_layer->setProperty(values);
  }

  /**
   * @copydoc Layer::supportInPlace()
   */
  virtual bool supportInPlace() const override { return false; }

  /**
   * @copydoc Layer::requireLabel()
   */
  virtual bool requireLabel() const override {
    return dist_layer->requireLabel();
  }

  /**
   * @brief     set distribute layer
   * @param[in] l layer to distribute along time
   */
  void setDistLayer(std::unique_ptr<Layer> &&l) { dist_layer = std::move(l); }

  /**
   * @brief     get distribute layer
   * @retval dist_layer std::shared_ptr<Layer>
   */
  Layer *getDistLayer() { return dist_layer.get(); };

  /**
   * @brief     get distribute layer
   * @retval dist_layer std::shared_ptr<Layer>
   */
  const Layer *getDistLayer() const { return dist_layer.get(); };

  inline static const std::string type = "time_dist";

private:
  /**
   * @brief Layer to be distributed through time
   */
  std::unique_ptr<Layer> dist_layer;
  std::vector<Weight> weights_wrapper;
  std::vector<Var_Grad> tensors_wrapper;

  /**
   * @brief pointer value of each input/output tensors to compare position
   */
  float *positions[4];

  /**
   * @brief  Transpose Input and Output Tensors to avoid duplicatation becuase
   * of memory optimization
   * It transpose the net_input.getVariableRef, net_input.getGradientRef,
   * net_hidden.getVariableRef and net_hidden.getGradientRef.
   *
   * @param context Run layer context
   */
  void transposeInOut(RunLayerContext &context);

  /**
   * @brief     get transposed Tensor according to time iteration axis
   *            [b, 1, h, w] to [h, 1, b, w]
   * @param[in] m Tensor
   * @retval Tensor transposed Tensor
   */
  static Tensor transposeTensor(Tensor &m);

  /**
   * @brief  calculate the pointer of each input and output tensors
   *
   * @param context Run layer context
   */
  void setPosition(RunLayerContext &context);

  /**
   * @brief Fill weights from the given context
   *
   * @param context The given context
   */
  void fillWeightsFromContext(RunLayerContext &context);

  /**
   * @brief Get the Weights for Context object
   *
   * @return std::vector<Weight *> The list of weights
   */
  std::vector<Weight *> getWeightsForContext();

  /**
   * @brief Fill tensors from the given context
   *
   * @param context The given context
   */
  void fillTensorsFromContext(RunLayerContext &context);

  /**
   * @brief Get the Tensors for Context object
   *
   * @return std::vector<Var_Grad *> The list of tensors
   */
  std::vector<Var_Grad *> getTensorsForContext();

  /**
   * @brief Clean the values filled from context
   *
   * @note This is necessary to ensure that all the references to the stored
   * tensors are cleared for the memory to be released after run is complete.
   *
   */
  void clearFromContext() {
    weights_wrapper.clear();
    tensors_wrapper.clear();
  }

  /**
   * @brief Fill init context from the given dist context
   *
   * @param context context to be set/filled
   * @param dist_context context from which to be filled
   */
  void fillLayerInitContext(InitLayerContext &context,
                            const InitLayerContext &dist_context);
};
} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __TIME_DIST_H__ */
