import onnx
import numpy as np
import json
import os
import shutil
from onnx import numpy_helper, TensorProto

def cleanName(name):
    if name.startswith('/'):
        name = name[1:]
    
    name = name.replace('/', '_')
    name = name.replace('.', '_')
    name = name.replace(':', '_')
    name = name.lower()
    
    return name
    
    
model = onnx.load("./qwen3_model.onnx", load_external_data=True)

metadata = {}

script_dir = os.path.dirname(os.path.abspath(__file__))
folder = os.path.join(script_dir, "bins")
if os.path.exists(folder):
    shutil.rmtree(folder) 
os.makedirs(folder)

for tensor in model.graph.initializer:
    arr = numpy_helper.to_array(tensor).astype(np.float32)
    
    filename = f"./bins/{cleanName(tensor.name)}.bin"
    arr.tofile(filename)
    
    # Save metadata (name, dtype, shape, file)
    metadata[tensor.name] = {
        "file": filename,
        "tensor name": tensor.name,
        "dtype": TensorProto.DataType.Name(tensor.data_type),
        "shape": list(arr.shape)
    }
    
    print(f"Saved {tensor.name} -> {filename}, dtype={arr.dtype}, shape={arr.shape}")

with open("./weights_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)    