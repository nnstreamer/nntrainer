import onnx
import onnx.helper as helper
import numpy as np

def create_fc_onnx_model():
    """
    Create an ONNX model with:
    - Input: 2x2 tensor
    - Single FC layer without bias
    - Output: 2x5 tensor
    """
    
    # Create input tensor info
    input_tensor = helper.make_tensor_value_info('input', onnx.TensorProto.FLOAT, [2, 2])
    
    # Create output tensor info
    output_tensor = helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT, [2, 5])
    
    # Create weight tensor for FC layer
    # Input shape: [2, 2], Output shape: [2, 5]
    # Weight shape should be [2, 5] for the linear transformation
    weight_data = np.random.randn(2, 5).astype(np.float32)
    weight_tensor = helper.make_tensor('fc_weight', onnx.TensorProto.FLOAT, [2, 5], weight_data.flatten())
    
    # Create FC layer node (MatMul operation since no bias)
    fc_node = helper.make_node(
        'MatMul',                    # operation type
        ['input', 'fc_weight'],      # inputs
        ['output'],                  # outputs
        name='fc_layer'
    )
    
    # Create the graph
    graph = helper.make_graph(
        [fc_node],                   # nodes
        'simple_fc_model',           # name
        [input_tensor],              # inputs
        [output_tensor],             # outputs
        [weight_tensor]              # initializers (weights)
    )
    
    # Create the model
    model = helper.make_model(graph)
    model.opset_import[0].version = 13  # Use ONNX opset version 13
    
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

if __name__ == "__main__":
    # Create the model
    print("Creating ONNX model with 2x2 input and FC layer (2x5 output)...")
    model = create_fc_onnx_model()
    
    # Verify the model
    if verify_model(model):
        # Print model information
        print_model_info(model)
        
        # Save the model
        save_model(model, '../fc_model.onnx')
        
        print("\nModel created successfully!")
        print("You can use this model with ONNX Runtime or other ONNX-compatible frameworks.")
    else:
        print("Failed to create valid model.")
