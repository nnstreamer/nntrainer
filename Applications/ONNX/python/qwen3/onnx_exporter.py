from transformers import AutoModelForCausalLM, AutoTokenizer
from custom_qwen3 import  NNTrainerQwen3ForCausalLM, Qwen3RotaryEmbedding
import torch
import numpy as np

model_name = "Qwen/Qwen3-1.7B"

official_model = AutoModelForCausalLM.from_pretrained(model_name)

qwenConfig = official_model.config
custom_model =  NNTrainerQwen3ForCausalLM(qwenConfig)
custom_model.load_state_dict(official_model.state_dict(),strict=False)

rotary_emb = Qwen3RotaryEmbedding(qwenConfig)

x = torch.tensor([[3,],]).view(-1, 1)
position_ids = torch.arange(1).reshape(1, -1)
cos, sin = rotary_emb(x, position_ids)
cos, sin = torch.tensor(cos.numpy()), torch.tensor(sin.numpy())
variance_epsilon = torch.tensor([[1e-6,]])

logits_of_custom_model = custom_model(x,cos,sin,variance_epsilon)
logits_of_official_model = official_model(x).logits

if (torch.allclose(logits_of_custom_model, logits_of_official_model,atol=1e-5)):
    print("<All logits matched successfully>")
else:
    print("<Some logits do not match>")

logits_of_custom_model = logits_of_custom_model.detach().numpy()
logits_of_custom_model.tofile("./modelling_logits.bin")

torch.onnx.export(
    custom_model, (x, cos, sin, variance_epsilon),
    'qwen3_model.onnx',
    export_params=True,
    opset_version=17,
    input_names=['input', 'cos', 'sin', 'variance_epsilon'],
    output_names=['output'],
    keep_initializers_as_inputs=False,
    dynamic_axes=None,
 )

print("<Model exported successfully>") 

