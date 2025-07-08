// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   unsqueeze_layer.cpp
 * @date   08 July 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sachin Singh <sachin.3@samsung.com>
 * @author Abhinav Dwivedi <abhinav.d@samsung.com>
 * @bug	   No known bugs except for NYI items
 * @brief  This is Unsqueeze Layer Class for Neural Network
 * @todo   Input axis can't be negative. Have to add support for negative axis input. Negative axis handling in unsqueeze layer is implemented
 * 
 */

 #include <layer_context.h>
 #include <nntrainer_error.h>
 #include <nntrainer_log.h>
 #include <node_exporter.h>
 #include <unsqueeze_layer.h>
 namespace nntrainer {
 
 static constexpr size_t SINGLE_INOUT_IDX = 0;
 
 void UnsqueezeLayer::finalize(InitLayerContext &context) {
   NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
     << "Unsqueeze only supports 1 input";
  
   const TensorDim &in_dim = context.getInputDimensions()[0]; //Input tensor
 
   auto &axis = std::get<props::Axis>(unsqueeze_props);
   auto &original_ndim = std::get<props::Original_ndim>(unsqueeze_props);

   int input_axis = axis.get();
   int input_ndim = original_ndim.get();
   

  NNTR_THROW_IF(input_ndim >= 4, std::invalid_argument)
  << "Unsqueeze layer currently supported for atmost 3 dimensions";

  NNTR_THROW_IF((input_axis > (input_ndim) || input_axis < (-1-input_ndim)), std::invalid_argument)
    << "Unsqueeze layer must have axis in range ["<<(-1-input_ndim)<<","<<(input_ndim)<<"] for input_ndims "<<input_ndim<<" ."<<" Your axis is "<<input_axis;

   if(axis < 0){
        axis.set(input_axis + input_ndim + 1); //negative axis handling
   }

   axis.set((in_dim.getNumDim() + input_axis - input_ndim - 1)); // setting axis relative to 4D tensor. 

  TensorDim out_dim = context.getInputDimensions()[0];

  for(int idx = 0; idx < input_axis ; idx++){
       out_dim[idx] = out_dim[idx+1];
  }
  out_dim[input_axis]=1;  
 
  out_dim.setDataType(context.getActivationDataType());
  context.setOutputDimensions({out_dim});

 }
 
 void UnsqueezeLayer::forwarding(RunLayerContext &context, bool training) {

   if (!context.getInPlace()) {
     context.getOutput(SINGLE_INOUT_IDX)
       .copyData(context.getInput(SINGLE_INOUT_IDX));
   }

 }
 
 void UnsqueezeLayer::calcDerivative(RunLayerContext &context) { 
  
   if (!context.getInPlace()) {
     context.getOutgoingDerivative(SINGLE_INOUT_IDX)
       .copyData(context.getIncomingDerivative(SINGLE_INOUT_IDX));
   }
 }
 
 void UnsqueezeLayer::setProperty(const std::vector<std::string> &values) {
   auto remain_props = loadProperties(values, unsqueeze_props);
   if (!remain_props.empty()) {
     std::string msg = "[UnsqueezeLayer] Unknown Layer Properties count " +
                       std::to_string(remain_props.size());
     throw exception::not_supported(msg);
   }
 }
 
 void UnsqueezeLayer::exportTo(Exporter &exporter,
                             const ml::train::ExportMethods &method) const {
   exporter.saveResult(unsqueeze_props, method, this);
 }
 
 } /* namespace nntrainer */
 