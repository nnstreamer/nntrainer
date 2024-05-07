// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file   fc_layer_cl.h
 * @date   7 May 2024
 * @brief  This is Fully Connected Layer Class of Neural Network with OpenCl
 * implementation
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Debadri Samaddar <s.debadri@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __FC_LAYER_CL_H__
#define __FC_LAYER_CL_H__
#ifdef __cplusplus

#include <common_properties.h>
#include <layer_impl.h>
#include <opencl_buffer.h>
#include <opencl_kernel.h>

#define CREATE_IF_EMPTY_DIMS(tensor, ...) \
  do {                                    \
    if (tensor.empty())                   \
      tensor = Tensor(__VA_ARGS__);       \
  } while (0);

namespace nntrainer {

/**
 * @class   FullyConnecedLayer
 * @brief   fully connected layer
 */
class FullyConnectedLayerCl : public LayerImpl {
public:
  /**
   * @brief     Constructor of Fully Connected Layer
   */
  FullyConnectedLayerCl();

  /**
   * @brief     Destructor of Fully Connected Layer
   */
  ~FullyConnectedLayerCl() = default;

  /**
   *  @brief  Move constructor.
   *  @param[in] FullyConnected &&
   */
  FullyConnectedLayerCl(FullyConnectedLayerCl &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs FullyConnectedLayer to be moved.
   */
  FullyConnectedLayerCl &operator=(FullyConnectedLayerCl &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(RunLayerContext &context, bool training) override;

  /**
￼   * @copydoc Layer::incremental_forwarding(RunLayerContext &context, unsigned
￼   * int from, unsigned int to, bool training)
￼   */
  void incremental_forwarding(RunLayerContext &context, unsigned int from,
                              unsigned int to, bool training) override;

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
                const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override {
    return FullyConnectedLayerCl::type;
  };

  /**
   * @brief declaring static kernel objects
   */
  static opencl::Kernel kernel_sgemv;
  static opencl::Kernel kernel_sgemm;
  static opencl::Kernel kernel_dot;

  /**
   * @brief Process data and dimensions for dot operation used in fc_layer
   * @param[in] input Tensor
   * @param[in] weight Tensor
   * @param[in] result Tensor
   * @param[in] RunLayerContext reference
   */
  void fcDotProcess(Tensor const &input, Tensor const &weight, Tensor &result,
                    RunLayerContext &context);

  /**
   * @brief     sgemv computation : Y = A*X + Y
   * @param[in] matAdata float * for Matrix A
   * @param[in] vecXdata float * for Vector X
   * @param[in] vecYdata float * for Vector Y
   * @param[in] dim1 number of A's row
   * @param[in] dim2 number of X's columns
   * @param[in] lda number of X's columns
   * @param[in] context RunLayerContext reference
   */
  void fc_sgemv_cl(const float *matAdata, const float *vecXdata,
                   float *vecYdata, unsigned int dim1, unsigned int dim2,
                   unsigned int lda, RunLayerContext &context);

  /**
   * @brief     dot computation : sum of all X * Y
   * @param[in] matAdata float * for Vector A
   * @param[in] vecXdata float * for Vector X
   * @param[in] dim1 number of elements in both input vectors
   * @param[in] context RunLayerContext reference
   */
  float fc_dot_cl(const float *matAdata, const float *vecXdata,
                  unsigned int dim1, RunLayerContext &context);

  /**
   * @brief     sgemm computation : Y = op(A)*op(B) + C,
   * where op(X) is one of X or X**T
   * @param[in] A float * for Matrix A
   * @param[in] B float * for Matrix B
   * @param[in] C float * for Matrix C
   * @param[in] M number of op(A)'s and C's row
   * @param[in] N number of op(B)'s and C's columns
   * @param[in] K number of op(A)'s and columns and op(B)'s rows
   * @param[in] context RunLayerContext reference
   */
  void fc_sgemm_cl(const float *A, const float *B, float *C, unsigned int M,
                   unsigned int N, unsigned int K, unsigned int lda,
                   unsigned int ldb, unsigned int ldc,
                   RunLayerContext &context);

  /**
   * @copydoc Layer::supportBackwarding()
   */
  bool supportBackwarding() const override { return true; }

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  void setProperty(const std::vector<std::string> &values) override;

  inline static const std::string type = "fully_connected";

private:
  std::tuple<props::Unit>
    fc_props; /**< fc layer properties : unit - number of output neurons */
  std::array<unsigned int, 2> weight_idx; /**< indices of the weights */
};
} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __FC_LAYER_CL__ */
