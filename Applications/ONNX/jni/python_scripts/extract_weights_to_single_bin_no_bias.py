import onnx
import numpy as np
import json
from onnx import numpy_helper, TensorProto

def cleanName(name):
    """Clean layer names to be compatible with NNTrainer"""
    # if name.startswith('/'):
    #     name = name[1:]
    
    # name = name.replace('/', '_')
    # name = name.replace('.', '_')
    # name = name.replace(':', '_')
    # name = name.lower()
    
    # Special handling for our FC layers to match NNTrainer expectations
    if name == "fc1_weight":
        return "fc1"
    elif name == "fc2_weight":
        return "fc2"
    
    return name

def extract_weights_to_single_bin(model_path, output_bin_path, metadata_path):
    """
    Extract weights from an ONNX model and save them to a single binary file
    with metadata JSON file.
    
    Args:
        model_path: Path to the ONNX model file
        output_bin_path: Path where the single .bin file will be saved
        metadata_path: Path where the metadata JSON file will be saved
    """
    
    # Load the ONNX model
    print(f"Loading ONNX model from {model_path}")
    model = onnx.load(model_path, load_external_data=True)
    
    # Initialize variables for the single bin file
    all_weights = []
    metadata = {}
    offset = 0
    
    # Collect all weights and their metadata
    print("Extracting weights from model...")
    for tensor in model.graph.initializer:
        # Convert tensor to numpy array
        arr = numpy_helper.to_array(tensor)
        
        # Store the array for later concatenation
        all_weights.append(arr.flatten())
        
        # Save metadata (name, dtype, shape, offset, size)
        metadata[cleanName(tensor.name)] = {
            "tensor_name": tensor.name,
            "clean_name": cleanName(tensor.name),
            "dtype": str(arr.dtype),
            "onnx_dtype": TensorProto.DataType.Name(tensor.data_type),
            "shape": list(arr.shape),
            "offset": offset,
            "size": arr.size * arr.itemsize  # size in bytes
        }
        
        # Update offset for next tensor
        offset += arr.size * arr.itemsize
        
        print(f"  Extracted {tensor.name} -> shape: {arr.shape}, dtype: {arr.dtype}, offset: {offset - arr.size * arr.itemsize}")
    
    # Concatenate all weights into a single array
    if all_weights:
        print("Concatenating all weights...")
        combined_weights = np.concatenate(all_weights)
        
        # Save the combined weights to a single .bin file
        print(f"Saving weights to {output_bin_path}")
        combined_weights.tofile(output_bin_path)
        print(f"Saved all weights to {output_bin_path}")
        
        # Save metadata to a JSON file
        print(f"Saving metadata to {metadata_path}")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to {metadata_path}")
        
        print(f"Total weights: {len(all_weights)} tensors")
        print(f"Total size: {len(combined_weights)} elements, {combined_weights.nbytes} bytes")
        
        # Print metadata summary
        print("\nMetadata summary:")
        for name, info in metadata.items():
            print(f"  {name}: offset={info['offset']}, size={info['size']} bytes, shape={info['shape']}")
    else:
        print("No weights found in the model")
        return False
    
    return True

def verify_single_bin_file(bin_path, metadata_path):
    """Verify that the single bin file and metadata are consistent"""
    print(f"\nVerifying {bin_path} with {metadata_path}")
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Get file size
    import os
    file_size = os.path.getsize(bin_path)
    print(f"Binary file size: {file_size} bytes")
    
    # Verify offsets and sizes
    total_expected_size = 0
    for name, info in metadata.items():
        total_expected_size += info['size']
    
    print(f"Expected size from metadata: {total_expected_size} bytes")
    
    if file_size == total_expected_size:
        print("✓ File size matches metadata")
        return True
    else:
        print("✗ File size does not match metadata")
        return False

if __name__ == "__main__":
    # Extract weights from the simple ONNX model
    success = extract_weights_to_single_bin(
        model_path="../simple_model_no_bias.onnx",
        output_bin_path="../simple_model_no_bias_weights.bin",
        metadata_path="../simple_model_no_bias_weights.bin_metadata.json"
    )
    
    if success:
        # Verify the created files
        verify_single_bin_file("../simple_model_no_bias_weights.bin", "../simple_model_no_bias_weights.bin_metadata.json")
        print("\nWeight extraction completed successfully!")
    else:
        print("\nWeight extraction failed!")
