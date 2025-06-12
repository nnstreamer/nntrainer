// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Sehyeon Park <shlion@snu.ac.kr>
 *
 * @file   groupconv_layer.h
 * @date   27 April 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sehyeon Park <shlion@snu.ac.kr>
 * @bug    No known bugs except for NYI items
 * @brief  This is Group Convolution Layer Class for Neural Network, Based on conv2d layer.
 *
 */

 #ifndef __GROUPCONV_LAYER_H_
 #define __GROUPCONV_LAYER_H_
 #ifdef __cplusplus
 
 #include <memory.h>
 
 #include <common_properties.h>
 #include <layer_impl.h>
 
 namespace nntrainer { // TODO : CHANGE ALL FUNCTIONS FROM CONV2D FOR GROUP CONVOLUTION
 
 constexpr const unsigned int GROUPCONV_DIM = 2;
 
 /**
  * @class   Group Convolution Layer
  * @brief   Group Convolution Layer
  */
 class GroupConvLayer : public LayerImpl {
 public:
   /**
    * @brief     Constructor of Group Conv Layer
    */
   GroupConvLayer(const std::array<unsigned int, GROUPCONV_DIM * 2> &padding_ = {
                 0, 0, 0, 0});
 
   /**
    * @brief     Destructor of Group Conv Layer
    */
   ~GroupConvLayer() = default;
 
   /**
    *  @brief  Move constructor of Group Conv Layer.
    *  @param[in] GroupConvLayer &&
    */
   GroupConvLayer(GroupConvLayer &&rhs) noexcept = default;
 
   /**
    * @brief  Move assignment operator.
    * @param[in] rhs GroupConvLayer to be moved.
    */
   GroupConvLayer &operator=(GroupConvLayer &&rhs) = default;
 
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
                 const ml::train::ExportMethods &method) const override;
 
   /**
    * @copydoc Layer::getType()
    */
   const std::string getType() const override { return GroupConvLayer::type; };
 
   /**
    * @copydoc Layer::supportBackwarding()
    */
   bool supportBackwarding() const override { return true; }
 
   using Layer::setProperty;
 
   /**
    * @copydoc Layer::setProperty(const PropertyType type, const std::string
    * &value)
    */
   void setProperty(const std::vector<std::string> &values) override;
 
   static constexpr const char *type = "group_convolution";
 
 private:
   std::array<unsigned int, GROUPCONV_DIM * 2> padding;
   std::tuple<props::FilterSize, std::array<props::KernelSize, GROUPCONV_DIM>,
              std::array<props::Stride, GROUPCONV_DIM>, props::Padding2D,
              std::array<props::Dilation, GROUPCONV_DIM>, props::SplitNumber>
     conv_props;
 
   std::array<unsigned int, 5> wt_idx; /**< indices of the weights and tensors */
 };
 
 } // namespace nntrainer
 
 #endif /* __cplusplus */
 #endif /* __GROUPCONV_LAYER_H_ */
 
