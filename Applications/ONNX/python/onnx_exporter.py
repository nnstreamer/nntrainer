"""
SPDX-License-Identifier: Apache-2.0
Copyright (C) 2025 Seungbaek Hong <sb92.hong@samsung.com>

@file onnx_exporter.py
@date 17 July 2025
@breif It modifies HuggingFace's Llama model files to be compatible with NNTrainer and
       saves them as ONNX model files.
@note This script has been tested with transformers version 4.51.3 and PyTorch version 2.3.1

@author Seungbaek Hong <sb92.hong@samsung.com>
"""

import torch
from transformers import LlamaConfig, AutoTokenizer, LlamaForCausalLM
from modeling_llama import LlamaRotaryEmbedding, NNTrainerLlamaForCausalLM

model_path = "model_directory/"

# Load & Create HuggingFace LLaMa model using transformers library
tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False)
official_model = LlamaForCausalLM.from_pretrained(model_path)

# Test loaded model
generation_config = {
    "max_length": 50,
    "num_beams": 5,
    "temperature": 0.7,
    "do_sample": True
}

prompt = "The capital of Korea is"
input_ids = tokenizer.encode(prompt, return_tensors="pt")
with torch.no_grad():
    output = official_model.generate(input_ids, **generation_config)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated text:", generated_text)

# Create Our LLaMa Model and load weights from HuggingFace model
config = LlamaConfig.from_pretrained(model_path)
custom_model = NNTrainerLlamaForCausalLM(config)
custom_model.load_state_dict(official_model.state_dict())

# Comparison and validation of logits between HuggingFace and our model
rotary_emb = LlamaRotaryEmbedding(config)
x = torch.tensor([[1,],]).view(-1, 1)
position_ids = torch.arange(1).reshape(1, -1)
cos, sin = rotary_emb(x, position_ids)
cos, sin = torch.tensor(cos.numpy()), torch.tensor(sin.numpy())
variance_epsilon = torch.tensor([[1e-6,]])

logits_of_custom_model = custom_model(x, cos, sin, variance_epsilon)
logits_of_official_model = official_model(x).logits

if ((logits_of_custom_model == logits_of_official_model).all()):
    print("<All logits matched successfully>")
else:
    print("<Some logits do not match>")

# Export ONNX odel
torch.onnx.export(
    custom_model, (x, cos, sin, variance_epsilon),
    'llama_model.onnx',
    export_params=True,
    opset_version=14,
    input_names=['input', 'cos', 'sin', 'variance_epsilon'],
    output_names=['output'],
    keep_initializers_as_inputs=False,
 )

print("<Model exported successfully>")
