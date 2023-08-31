// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2022 hyeonseok Lee <hs89.lee@samsung.com>
 *
 * @file   multi_head_attention_layer.h
 * @date   08 July 2022
 * @see    https://github.com/nnstreamer/nntrainer
 *         https://arxiv.org/abs/1706.03762
 * @author hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is MultiHeadAttention Layer Class for Neural Network
 *
 */

#ifndef __MULTI_HEAD_ATTENTION_LAYER_H__
#define __MULTI_HEAD_ATTENTION_LAYER_H__
#ifdef __cplusplus

#include <complex>

#include <acti_func.h>
#include <complex>
#include <layer_impl.h>
#include <utility>

#include <iostream>

namespace nntrainer {

/**
 * @class   Multi Head Attention Layer
 * @brief   Implementation of multi head attention which is described in paper
 * "Attention is all you need"
 */
class MultiHeadAttentionLayer : public LayerImpl {
public:
  /**
   * @brief     Constructor of MultiHeadAttention Layer
   */
  MultiHeadAttentionLayer();

  /**
   * @brief     Destructor of MultiHeadAttention Layer
   */
  ~MultiHeadAttentionLayer();

  /**
   *  @brief  Move constructor of MultiHeadAttentionLayer.
   *  @param[in] MultiHeadAttentionLayer &&
   */
  MultiHeadAttentionLayer(MultiHeadAttentionLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs MultiHeadAttentionLayer to be moved.
   */
  MultiHeadAttentionLayer &operator=(MultiHeadAttentionLayer &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::incremental_forwarding(RunLayerContext &context, unsigned
   * int from, unsigned int to, bool training)
   */
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
   * @copydoc bool supportBackwarding() const
   */
  bool supportBackwarding() const override { return true; };

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override {
    return MultiHeadAttentionLayer::type;
  };

  /**
   * @copydoc Layer::setBatch(RunLayerContext &context, unsigned int batch)
   */
  void setBatch(RunLayerContext &context, unsigned int batch) override;

  inline static const std::string type = "multi_head_attention";

private:
  std::tuple<props::NumHeads, props::ProjectedKeyDim, props::ProjectedValueDim,
             props::OutputShape, props::DropOutRate,
             props::ReturnAttentionWeight, props::AverageAttentionWeight,
             props::MaxTimestep>
    multi_head_attention_props; /**< multi_head_attention layer properties */

  ActiFunc sm; /** softmax activation operation */
  std::array<unsigned int, 16>
    weight_idx; /**< indices of the weights and tensors */

  /**
   * @brief     to protect overflow
   */
  float epsilon;

  inline static std::vector<std::vector<std::complex<float>>> *freqs_cis = {};

  template <typename T = float>
  void precompute_freqs_cis(int dim, int seq_len, float theta = 10000.0) {
    if (freqs_cis == nullptr) {
      std::cerr << "hello\n";
      std::vector<float> freqs(dim / 2);
      for (int i = 0; i < dim / 2; ++i) {
        freqs[i] = 1.0 / (std::pow(theta, (2 * i) / static_cast<float>(dim)));
      }

      auto cis = new std::vector<std::vector<std::complex<T>>>();
      cis->assign(1024, std::vector<std::complex<T>>(dim / 2, 0));

      for (int i = 0; i < seq_len; ++i) {
        for (int j = 0; j < dim / 2; ++j) {
          float angle = i * freqs[j];
          (*cis)[i][j] = static_cast<std::complex<T>>(std::polar(1.0f, angle));
        }
      }

      freqs_cis = cis;
    }
  }

  template <typename T = float>
  std::tuple<T, T> apply_rotary_emb(T real, T imag, int i, int j) {
    std::complex<float> input_complex(static_cast<float>(real),
                                      static_cast<float>(imag));
    std::complex<float> output_complex =
      input_complex * (*freqs_cis)[i][(int)j / 2];
    return std::make_tuple(static_cast<T>(output_complex.real()),
                           static_cast<T>(output_complex.imag()));
  }

  template <typename T = float>
  Tensor apply_rotary_emb_tensor(Tensor in, unsigned int dim,
                                 unsigned int from) {
    Tensor out(in.getDim());
    for (int b = 0; b < (int)in.batch(); b++) {
      for (int c = 0; c < (int)in.channel(); c++) {
        for (int h = 0; h < (int)in.height(); h++) {
          for (int w = 0; w < (int)in.width(); w = w + 2) {
#ifdef ENABLE_FP16
            _FP16 real = in.getValue<_FP16>(b, c, h, w);
            _FP16 imag = in.getValue<_FP16>(b, c, h, w + 1);
            std::tie(real, imag) =
              apply_rotary_emb<_FP16>(real, imag, from + h, w % dim);
#else
            float real = in.getValue(b, c, h, w);
            float imag = in.getValue(b, c, h, w + 1);
            std::tie(real, imag) =
              apply_rotary_emb<float>(real, imag, from + h, w % dim);
#endif
            out.setValue(b, c, h, w, real);
            out.setValue(b, c, h, w + 1, imag);
          }
        }
      }
    }
    return out;
  }

  /**
   * @brief calculate common derivative
   * @param context Context of the layer
   */
  void calcCommonDerivative(RunLayerContext &context);
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __MULTI_HEAD_ATTENTION_LAYER_H__ */
