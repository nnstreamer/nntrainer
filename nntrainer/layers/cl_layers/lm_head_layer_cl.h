// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Yash Singh <yash.singh@samsung.com>
 *
 * @file   lm_head_layer_cl.h
 * @date   1 Oct 2024
 * @brief  Implementation of custom lm head layer for GPU
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Yash Singh <yash.singh@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Implementation of custom lm head layer for GPU
 */

#ifndef __CUSTOM_LM_HEAD_LAYER_H__
#define __CUSTOM_LM_HEAD_LAYER_H__

#include <custom_vocab_selection.h>
#include <layer_context.h>
#include <layer_impl.h>
#include <node_exporter.h>
#include <utility>

namespace nntrainer {

namespace props {

/**
 * @brief indicated whether do vocab selection or not
 *
 */
class UseVocabSelection : public nntrainer::Property<bool> {
public:
  /**
   * @brief Construct a new UseVocabSelection object
   *
   */
  UseVocabSelection(bool value = false) { set(value); }
  static constexpr const char *key = "use_vocab_selection";
  using prop_tag = nntrainer::bool_prop_tag;
};

/**
 * @brief LshChoices property, indicate how many words will be choose
 *
 */
class LshChoices : public nntrainer::PositiveIntegerProperty {
public:
  /**
   * @brief Construct a new LshChoices object with a default value 1
   *
   */
  LshChoices(unsigned int value = 1) { set(value); };
  static constexpr const char *key = "lsh_choices"; /**< unique key to access */
  using prop_tag = nntrainer::uint_prop_tag;        /**< property type */
};

} // namespace props

/**
 * @brief A Custom LM Head layer for llama.
 *
 */
class CustomLMHeadLayerCl : public LayerImpl {
public:
  /**
   * @brief Construct a new Custom LM Head layer object
   *
   */
  CustomLMHeadLayerCl();

  /**
   * @brief Destroy the Custom LM Head layer object
   *
   */
  ~CustomLMHeadLayerCl() {}

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(nntrainer::InitLayerContext &context) override;

  /**
   * @copydoc Layer::intialize(RunLayerContext &context)
   */
  void initialize(nntrainer::RunLayerContext &context) {
    auto use_vocab_selection =
      std::get<props::UseVocabSelection>(custom_lm_head_props).get();

    if (use_vocab_selection) {
      auto lsh_choices =
        std::get<props::LshChoices>(custom_lm_head_props).get();
      initVocabSelection(LshType::ORTHOSIMHASH, lsh_choices, context);
    }
  }

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(nntrainer::RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::incremental_forwarding(RunLayerContext &context, unsigned
   * int from, unsigned int to, bool training)
   */
  void incremental_forwarding(nntrainer::RunLayerContext &context,
                              unsigned int from, unsigned int to,
                              bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(nntrainer::RunLayerContext &context) override;

  /**
   * @copydoc Layer::calcGradient(RunLayerContext &context)
   */
  //   void calcGradient(nntrainer::RunLayerContext &context) override;

  /**
   * @copydoc bool supportBackwarding() const
   */
  bool supportBackwarding() const override { return false; };

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ExportMethods method)
   */
  void exportTo(nntrainer::Exporter &exporter,
                const ml::train::ExportMethods &method) const override{};

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override {
    return CustomLMHeadLayerCl::type;
  };

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  /**
   * @brief Enum class for LSH Type
   */
  void initVocabSelection(LshType lshType, int lshChoices,
                          nntrainer::RunLayerContext &context);

  inline static const std::string type = "custom_lm_head";

  std::shared_ptr<VocabSelection> vocabSelection;

private:
  std::tuple<nntrainer::props::Unit, props::UseVocabSelection,
             props::LshChoices>
    custom_lm_head_props;
  std::array<unsigned int, 4> weight_idx; /**< indices of the weights */
  std::unique_ptr<nntrainer::Tensor>
    weight_T; // temporary weight. will be removed
};
} // namespace nntrainer

#endif /* __LM_HEAD_LAYER_CL_H__ */
