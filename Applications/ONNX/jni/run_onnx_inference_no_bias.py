import onnx
import numpy as np
from onnx import numpy_helper

def run_onnx_inference():
    # Load the ONNX model
    model_path = "simple_model_no_bias.onnx"
    model = onnx.load(model_path)
    
    # Print model info
    print("Model inputs:")
    for input_info in model.graph.input:
        print(f"  {input_info.name}: {[d.dim_value for d in input_info.type.tensor_type.shape.dim]}")
    
    print("\nModel outputs:")
    for output_info in model.graph.output:
        print(f"  {output_info.name}: {[d.dim_value for d in output_info.type.tensor_type.shape.dim]}")
    
    # Extract weights from the model
    weights = {}
    for initializer in model.graph.initializer:
        arr = numpy_helper.to_array(initializer)
        weights[initializer.name] = arr
        print(f"\nFound weight tensor: {initializer.name} with shape {arr.shape}")
    
    # Run manual inference with input [1, 2, 3, 4]
    input_data = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)  # Shape [1, 4]
    print(f"\nInput: {input_data}")
    
    # First layer: MatMul only (no bias)
    # input @ fc1.weight
    fc1_weight = weights['fc1.weight']  # Shape [4, 8]
    
    print(f"\nFC1 weight shape: {fc1_weight.shape}")
    
    fc1_output = np.dot(input_data[0], fc1_weight)
    print(f"FC1 output: {fc1_output}")
    
    # Second layer: MatMul only (no bias)
    # fc1_output @ fc2.weight
    fc2_weight = weights['fc2.weight']  # Shape [8, 2]
    
    print(f"\nFC2 weight shape: {fc2_weight.shape}")
    
    final_output = np.dot(fc1_output, fc2_weight)
    print(f"\nONNX manual calculation output: {final_output}")
    
    return final_output

if __name__ == "__main__":
    run_onnx_inference()
