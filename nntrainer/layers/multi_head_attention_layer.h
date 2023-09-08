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
  void initial_incremental_forwarding(RunLayerContext &context,
                                      unsigned int from, unsigned int to,
                                      bool training);

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

  unsigned int cache_index;

  inline static unsigned int layer_progress;

  inline static std::vector<std::vector<float>> *freqs_cos = {};
  inline static std::vector<std::vector<float>> *freqs_sin = {};
  inline static std::vector<float> freqs;

  /**
   * @brief     compute frequency for rotary embedding
   * @param[in] dim hidden dim size
   * @param[in] seq_len sequency length
   * @param[in] theta rotary angle
   */
  void precompute_freqs(int dim, unsigned int seq_len, float theta = 10000.0) {
    if (freqs_cos == nullptr) {
      unsigned int half_ = dim / 2;
      for (unsigned int i = 0; i < half_; ++i) {
        freqs.push_back(1.0 /
                        (std::pow(theta, (2 * i) / static_cast<float>(dim))));
      }

      auto cos = new std::vector<std::vector<float>>();
      cos->assign(seq_len, std::vector<float>(dim, 0));

      auto sin = new std::vector<std::vector<float>>();
      sin->assign(seq_len, std::vector<float>(dim, 0));

      for (unsigned int i = 0; i < seq_len; ++i) {
        for (unsigned int j = 0; j < half_; ++j) {
          float angle = i * freqs[j];
          (*cos)[i][j] = std::cos(angle);
          (*cos)[i][j + half_] = std::cos(angle); // repeated 2 times

          (*sin)[i][j] = std::sin(angle);
          (*sin)[i][j + half_] = std::sin(angle); // repeated 2 times
        }
      }

      freqs_cos = cos;
      freqs_sin = sin;
    }
  }

  /**
   * @brief     apply rotary embedding
   * @param[in] in input tensor
   * @param[in] dim hidden dim size
   * @param[in] from sequence order
   */
  void apply_rotary_emb_tensor(Tensor &in, unsigned int dim,
                               unsigned int from) {
    Tensor out(in.getDim());
    float value = 0;
    float transformed_value = 0.0;
    unsigned int half_ = dim / 2;
    unsigned int max_timestep =
      std::get<props::MaxTimestep>(multi_head_attention_props).get();

    std::vector<float> *cos_;
    std::vector<float> *sin_;

    if (from >= max_timestep) {
      std::cout << from << " " << max_timestep << std::endl;
      cos_ = new std::vector<float>(dim);
      sin_ = new std::vector<float>(dim);

      for (unsigned int i = 0; i < half_; ++i) {
        float angle = from * freqs[i];
        (*cos_)[i] = std::cos(angle);
        (*cos_)[i + half_] = std::cos(angle); // repeated 2 times

        (*sin_)[i] = std::sin(angle);
        (*sin_)[i + half_] = std::sin(angle); // repeated 2 times
      }
    }

    if (in.getDataType() == ml::train::TensorDim::DataType::FP32) {
      for (unsigned int b = 0; b < in.batch(); b++) {
        for (unsigned int c = 0; c < in.channel(); c++) {
          for (unsigned int h = 0; h < in.height(); h++) {
            if (from < max_timestep) {
              cos_ = &(*freqs_cos)[from + h];
              sin_ = &(*freqs_sin)[from + h];
            }

            for (unsigned int w = 0; w < in.width(); w = w + dim) {
              for (unsigned int k = 0; k < dim; k++) {
                unsigned int span = w + k;
                value = in.getValue<float>(b, c, h, span);

                if (k < half_) {
                  transformed_value =
                    -1.0 * in.getValue<float>(b, c, h, span + half_);
                } else {
                  transformed_value = in.getValue<float>(b, c, h, span - half_);
                }
                value = value * (*freqs_cos)[from][k] +
                        transformed_value * (*freqs_sin)[from][k];

                out.setValue(b, c, h, span, value);
              }
            }
          }
        }
      }
    } else if (in.getDataType() == ml::train::TensorDim::DataType::FP16) {

      for (unsigned int b = 0; b < in.batch(); b++) {
        for (unsigned int c = 0; c < in.channel(); c++) {
          for (unsigned int h = 0; h < in.height(); h++) {
            if (from < max_timestep) {
              cos_ = &(*freqs_cos)[from + h];
              sin_ = &(*freqs_sin)[from + h];
            }

            for (unsigned int w = 0; w < in.width(); w = w + dim) {
              for (unsigned int k = 0; k < dim; k++) {
#ifdef ENABLE_FP16
                unsigned int span = w + k;
                value = static_cast<float>(in.getValue<_FP16>(b, c, h, span));

                if (k < half_) {
                  transformed_value =
                    -1.0 * static_cast<float>(
                             in.getValue<_FP16>(b, c, h, half_ + span));
                } else {
                  transformed_value = static_cast<float>(
                    in.getValue<_FP16>(b, c, h, span - half_));
                }
                out.setValue(b, c, h, span,
                             static_cast<_FP16>(value * (*freqs_cos)[from][k] +
                                                transformed_value *
                                                  (*freqs_sin)[from][k]));

#else
                throw std::invalid_argument(
                  "Error: enable-fp16 is not enabled");
#endif
              }
            }
          }
        }
      }
    }
    in.copy(out);
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
