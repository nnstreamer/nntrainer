import re
import numpy as np

def parse_nntrainer(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Split by the separator if it exists, we'll focus on the first part for now
    # or maybe the user wants to compare the second part?
    # The user didn't specify, but usually the first part is the prompt phase.
    # Let's check the values.
    # nntrainer first block: [-0.00980008 0.0225656 ...]
    # torch first block: tensor([-0.0098,  0.0226, ...])
    # They look similar.
    
    parts = content.split('==========================')
    # We will try to match with the first part.
    content_to_parse = parts[0]
    
    # Find all arrays
    # Pattern: [number number ...]
    # It seems they can span multiple lines if they were longer, but here they seem to be on one line or wrapped.
    # In the file view:
    # 5: [-0.00980008 0.0225656 0.04084150.0160054 ...
    # It seems to be a single line with space separated values.
    
    arrays = []
    lines = content_to_parse.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith('[') and line.endswith(']'):
            # Extract numbers
            inner = line[1:-1]
            # Handle potential lack of spaces between numbers if they are negative?
            # e.g. 0.04084150.0160054 -> this looks like a missing space issue in the file view or the file itself.
            # Let's look at line 5 of nntrainer_output.txt:
            # [-0.00980008 0.0225656 0.04084150.0160054 0.0485524 0.016504-0.02666930.0164781-0.0111875-0.03469040.108574]
            # It seems some numbers are concatenated. "0.04084150.0160054" -> 0.0408415 and 0.0160054?
            # "0.016504-0.0266693" -> 0.016504 and -0.0266693
            
            # We need a regex to split numbers.
            # Pattern: optional minus, digit, dot, digits, optional exponent
            # Updated to handle cases like 0.04084150.0160054 where it might be parsed as 0.04084150 and .0160054
            nums = re.findall(r'-?(?:\d+\.\d+|\.\d+)(?:e[+-]?\d+)?', inner)
            arrays.append(np.array([float(x) for x in nums]))
            
    return arrays

def parse_torch(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
        
    # Structure:
    # [name] ...
    # tensor([values], ...)
    
    layers = []
    arrays = []
    
    lines = content.split('\n')
    current_name = None
    
    for line in lines:
        line = line.strip()
        if line.startswith('[model.layers'):
            # Extract name
            match = re.search(r'\[(.*?)\]', line)
            if match:
                current_name = match.group(1)
        elif line.startswith('tensor('):
            # Extract values
            # It might span multiple lines in torch output too, but here it looks like:
            # tensor([-0.0098,  0.0226, ...], dtype=torch.bfloat16)
            # Wait, in the file view:
            # 2: tensor([-0.0098,  0.0226,  0.0408,  0.0160,  0.0483,  0.0165, -0.0266,  0.0165,
            # 3:         -0.0112, -0.0347], dtype=torch.bfloat16)
            # It spans multiple lines.
            
            # We need to capture the full tensor string.
            # A simple way is to join all lines and then regex.
            pass
            
    # Let's do a full content regex for torch
    # Find blocks of [name] ... tensor(...)
    
    # We can iterate and maintain state
    full_text = content.replace('\n', ' ')
    
    # Regex to find names and tensors
    # Pattern: \[([^\]]+)\] output shape.*?tensor\(\[(.*?)\]
    # Be careful with nested brackets or dtypes.
    
    # Let's try a simpler approach: split by "output shape" to separate blocks
    blocks = full_text.split('output shape')
    # Skip the first split if it's empty or header
    
    # Actually, let's just find all occurrences of names and tensors.
    
    # Find all names
    names = re.findall(r'\[(model\.layers\.[^\]]+)\]', content)
    
    # Find all tensors
    # We can extract the content inside tensor([...])
    # Since it can be multiline, we use dotall
    # But there might be nested parens? No, usually just tensor([...], ...)
    
    # Let's process the file line by line to be safer
    
    current_tensor_str = ""
    in_tensor = False
    
    collected_tensors = []
    
    for line in lines:
        if 'tensor(' in line:
            in_tensor = True
            current_tensor_str = line
        elif in_tensor:
            current_tensor_str += line
        
        if in_tensor and '])' in line: # End of tensor list
            in_tensor = False
            # Extract numbers
            nums = re.findall(r'-?\d+\.\d+(?:e[+-]?\d+)?', current_tensor_str)
            collected_tensors.append(np.array([float(x) for x in nums]))
            current_tensor_str = ""
            
    return names, collected_tensors

def compare(nntrainer_arrays, torch_names, torch_arrays):
    print(f"Found {len(nntrainer_arrays)} nntrainer arrays")
    print(f"Found {len(torch_arrays)} torch arrays")
    
    limit = min(len(nntrainer_arrays), len(torch_arrays))
    
    for i in range(limit):
        nnt_arr = nntrainer_arrays[i]
        torch_arr = torch_arrays[i]
        name = torch_names[i] if i < len(torch_names) else f"Layer {i}"
        
        # Check size
        if len(nnt_arr) != len(torch_arr):
            print(f"Size mismatch at {i} ({name}): {len(nnt_arr)} vs {len(torch_arr)}")
            # Try to compare common elements?
            min_len = min(len(nnt_arr), len(torch_arr))
            nnt_arr = nnt_arr[:min_len]
            torch_arr = torch_arr[:min_len]
        
        # Compare
        # Using a tolerance. bfloat16 has low precision.
        # 1e-2 or 1e-3 might be appropriate.
        diff = np.abs(nnt_arr - torch_arr)
        max_diff = np.max(diff)
        
        print(f"Index {i}: {name} - Max Diff: {max_diff}")
        
        if max_diff > 0.1: # Threshold for "divergence"
            print(f"DIVERGENCE FOUND at Index {i}: {name}")
            print(f"NNTrainer: {nnt_arr[:5]}...")
            print(f"Torch:     {torch_arr[:5]}...")
            return

if __name__ == "__main__":
    nnt_arrays = parse_nntrainer('/home/donghak/workspace/nntrainer/nntrainer_output.txt')
    torch_names, torch_arrays = parse_torch('/home/donghak/workspace/nntrainer/torch_output.txt')
    
    compare(nnt_arrays, torch_names, torch_arrays)
