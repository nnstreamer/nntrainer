#ifndef __CAUSALLM_COMMON_PROPERTIES_H__
#define __CAUSALLM_COMMON_PROPERTIES_H__

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#include <base_properties.h>
#include <tensor.h>
#include <utility>

namespace causallm {
namespace props {

/**
 * @brief unit property, unit is used to measure how many weights are there
 *
 */
class FeatureSize : public nntrainer::PositiveIntegerProperty {
public:
  static constexpr const char *key =
    "feature_size";                          /**< unique key to access */
  using prop_tag = nntrainer::uint_prop_tag; /**< property type */
};

/**
 * @brief RMS_NORM_GAMMA_INIT Initialization Enumeration Information
 *
 */
WIN_EXPORT class RMS_NORM_GAMMA_INIT final
  : public nntrainer::EnumProperty<nntrainer::props::InitializerInfo> {
public:
  /**
   * @brief Construct a CUSTOM_RMS_NORM_GAMMA_INIT object
   */
  WIN_EXPORT RMS_NORM_GAMMA_INIT(
    nntrainer::Initializer value = nntrainer::Initializer::ONES) {
    set(value);
  };

  using prop_tag = nntrainer::enum_class_prop_tag;
  static constexpr const char *key = "gamma_initializer";
};
}; // namespace props

WIN_EXPORT enum RMSParams { gamma };

} // namespace causallm

#endif
