import onnx
import numpy as np
import json
from onnx import numpy_helper, TensorProto

def cleanName(name):
    if name.startswith('/'):
        name = name[1:]
    
    name = name.replace('/', '_')
    name = name.replace('.', '_')
    name = name.replace(':', '_')
    name = name.lower()
    
    return name
    
    
model = onnx.load("../fc_model.onnx", load_external_data=True)

metadata = {}


for tensor in model.graph.initializer:
    arr = numpy_helper.to_array(tensor)  
    
    filename = f"../{cleanName(tensor.name)}.bin"
    arr.tofile(filename)
    
    # Save metadata (name, dtype, shape, file)
    metadata[tensor.name] = {
        "file": filename,
        "tensor name": tensor.name,
        "dtype": TensorProto.DataType.Name(tensor.data_type),
        "shape": list(arr.shape)
    }
    
    print(f"Saved {tensor.name} -> {filename}, dtype={arr.dtype}, shape={arr.shape}")