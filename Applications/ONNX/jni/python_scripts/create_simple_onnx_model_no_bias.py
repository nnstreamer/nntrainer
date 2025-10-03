import onnx
import onnx.helper as helper
import numpy as np

def create_simple_onnx_model():
    """
    Create a simple ONNX model with two fully connected layers without bias.
    
    Input: [1, 4] tensor
    FC1: 4 -> 8 weights, no bias
    FC2: 8 -> 2 weights, no bias
    Output: [1, 2] tensor
    """
    
    # Create input tensor info
    input_tensor = helper.make_tensor_value_info('input', onnx.TensorProto.FLOAT, [1, 4])
    
    # Create output tensor info
    output_tensor = helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT, [1, 2])
    
    # Create weight tensor for first FC layer (4 -> 8)
    # Weight shape should be [4, 8] for the linear transformation
    fc1_weight_data = np.array([
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
    ], dtype=np.float32)
    fc1_weight_tensor = helper.make_tensor('fc1.weight', onnx.TensorProto.FLOAT, [4, 8], fc1_weight_data.flatten())
    
    # Create weight tensor for second FC layer (8 -> 2)
    # Weight shape should be [8, 2] for the linear transformation
    fc2_weight_data = np.array([
        [0.1, 0.2],
        [0.2, 0.3],
        [0.3, 0.4],
        [0.4, 0.5],
        [0.5, 0.6],
        [0.6, 0.7],
        [0.7, 0.8],
        [0.8, 0.9]
    ], dtype=np.float32)
    fc2_weight_tensor = helper.make_tensor('fc2.weight', onnx.TensorProto.FLOAT, [8, 2], fc2_weight_data.flatten())
    
    # Create FC1 layer node (MatMul operation)
    fc1_node = helper.make_node(
        'MatMul',                    # operation type
        ['input', 'fc1.weight'],     # inputs
        ['fc1_output'],              # outputs
        name='fc1'
    )
    
    # Create FC2 layer node (MatMul operation)
    fc2_node = helper.make_node(
        'MatMul',                    # operation type
        ['fc1_output', 'fc2.weight'],# inputs
        ['output'],                  # outputs
        name='fc2'
    )
    
    # Create the graph
    graph = helper.make_graph(
        [fc1_node, fc2_node],        # nodes
        'simple_fc_model_no_bias',   # name
        [input_tensor],              # inputs
        [output_tensor],             # outputs
        [fc1_weight_tensor, fc2_weight_tensor]  # initializers (weights)
    )
    
    # Create the model
    model = helper.make_model(graph)
    model.opset_import[0].version = 13  # Use ONNX opset version 13
    model.ir_version = 10  # Use compatible IR version
    
    return model

def save_model(model, filename):
    """Save the ONNX model to a file"""
    onnx.save(model, filename)
    print(f"Model saved to {filename}")

def verify_model(model):
    """Verify the model is valid"""
    try:
        onnx.checker.check_model(model)
        print("Model verification passed!")
        return True
    except onnx.checker.ValidationError as e:
        print(f"Model verification failed: {e}")
        return False

def print_model_info(model):
    """Print information about the model"""
    print("\nModel Information:")
    print(f"Model IR version: {model.ir_version}")
    print(f"Producer name: {model.producer_name}")
    print(f"Opset version: {model.opset_import[0].version}")
    
    print("\nGraph Information:")
    print(f"Graph name: {model.graph.name}")
    print(f"Number of inputs: {len(model.graph.input)}")
    print(f"Number of outputs: {len(model.graph.output)}")
    print(f"Number of nodes: {len(model.graph.node)}")
    print(f"Number of initializers: {len(model.graph.initializer)}")
    
    # Print input info
    for input_tensor in model.graph.input:
        print(f"\nInput: {input_tensor.name}")
        print(f"  Type: {input_tensor.type.tensor_type.elem_type}")
        print(f"  Shape: {[d.dim_value for d in input_tensor.type.tensor_type.shape.dim]}")
    
    # Print output info
    for output_tensor in model.graph.output:
        print(f"\nOutput: {output_tensor.name}")
        print(f"  Type: {output_tensor.type.tensor_type.elem_type}")
        print(f"  Shape: {[d.dim_value for d in output_tensor.type.tensor_type.shape.dim]}")
    
    # Print node info
    for node in model.graph.node:
        print(f"\nNode: {node.name}")
        print(f"  Op type: {node.op_type}")
        print(f"  Inputs: {node.input}")
        print(f"  Outputs: {node.output}")
        
    # Print initializer info
    for initializer in model.graph.initializer:
        print(f"\nInitializer: {initializer.name}")
        print(f"  Data type: {initializer.data_type}")
        print(f"  Shape: {initializer.dims}")

if __name__ == "__main__":
    # Create the model
    print("Creating ONNX model with two FC layers (no bias)...")
    model = create_simple_onnx_model()
    
    # Verify the model
    if verify_model(model):
        # Print model information
        print_model_info(model)
        
        # Save the model
        save_model(model, '../simple_model_no_bias.onnx')
        
        print("\nModel created successfully!")
        print("You can use this model with ONNX Runtime or other ONNX-compatible frameworks.")
    else:
        print("Failed to create valid model.")
