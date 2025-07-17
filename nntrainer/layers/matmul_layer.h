// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   matmul_layer.h
 * @date   26 March 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is matmul layer class (operation layer)
 */

#ifndef __MATMUL_LAYER_H__
#define __MATMUL_LAYER_H__
#ifdef __cplusplus

#include <common_properties.h>
#include <layer_devel.h>
#include <operation_layer.h>

namespace nntrainer {

/**
 * @class MatMul Layer
 * @brief MatMul Layer
 */
class MatMulLayer : public BinaryOperationLayer {
public:
  /**
   * @brief Constructor of MatMul Layer
   */
  NNTR_API MatMulLayer() :
    BinaryOperationLayer(),
    matmul_props(props::Print(), props::InPlaceProp()),
    support_backwarding(true) {}

  /**
   * @brief Destructor of MatMul Layer
   */
  NNTR_API ~MatMulLayer(){};

  /**
   *  @brief  Move constructor of MatMul Layer.
   *  @param[in] MatMulLayer &&
   */
  NNTR_API MatMulLayer(MatMulLayer &&rhs) noexcept = default;

  /**
   * @brief Move assignment operator.
   * @parma[in] rhs MatMulLayer to be moved.
   */
  NNTR_API MatMulLayer &operator=(MatMulLayer &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  NNTR_API void finalize(InitLayerContext &context) final;

  /**
   * @brief forwarding operation for matmul
   *
   * @param input1 first input tensor
   * @param input2 second input tensor
   * @param hidden tensor to store the result value
   */
  NNTR_API void forwarding_operation(const Tensor &input1, const Tensor &input2,
                                     Tensor &hidden) final;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  NNTR_API void calcDerivative(RunLayerContext &context) final;

  /**
   * @copydoc bool supportBackwarding() const
   */
  NNTR_API bool supportBackwarding() const final {
    return support_backwarding;
  };

  /**
   * @brief Initialize the in-place settings of the layer
   * @return InPlaceType
   */
  NNTR_API InPlaceType initializeInPlace() final {
    is_inplace = !std::get<props::InPlaceProp>(matmul_props).empty() &&
                 std::get<props::InPlaceProp>(matmul_props).get();
    support_backwarding = !is_inplace;

    if (!supportInPlace())
      return InPlaceType::NONE;
    else
      return InPlaceType::NON_RESTRICTING;
  }

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  NNTR_API void exportTo(Exporter &exporter,
                         const ml::train::ExportMethods &method) const final {}

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  NNTR_API void setProperty(const std::vector<std::string> &values) final;

  /**
   * @copydoc Layer::getType()
   */
  NNTR_API const std::string getType() const final {
    return MatMulLayer::type;
  };

  std::tuple<props::Print, props::InPlaceProp> matmul_props;
  bool support_backwarding;

  inline static const std::string type = "matmul";
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __MATMUL_LAYER_H__ */
