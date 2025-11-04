import torch
from transformers import AutoTokenizer, Qwen3Config, AutoModelForCausalLM
from custom_qwen3 import  NNTrainerQwen3ForCausalLM, Qwen3RotaryEmbedding
import onnx
import numpy as np

model_name = "Qwen/Qwen3-1.7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)


####Offical Model####

config = Qwen3Config.from_pretrained(model_name)  

# Fixed maximum length model can handle
max_len = 256
config.max_position_embeddings = max_len
official_model = AutoModelForCausalLM.from_pretrained(model_name,config = config).eval()

#tokens to be generated
num_tokens_to_generate = 20

# Prompt
prompt = "Tell me a dad joke about a computer: "
print("\nInput prompt: ",prompt)

enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len)
cur_len = enc.input_ids.size(1)

input_ids = torch.full((1, max_len), tokenizer.pad_token_id)  # preallocate with PAD
input_ids[0, :cur_len] = enc.input_ids[0]

generated = input_ids.clone()

for step in range(num_tokens_to_generate):  # generate 20 tokens

    outputs = official_model(
        input_ids=generated
    )

    next_token_logits = outputs.logits[:, cur_len - 1, :]
    next_token_id = torch.argmax(next_token_logits, dim=-1)

    generated[0][cur_len] = next_token_id
    cur_len += 1

decoded = tokenizer.decode(generated[0, :cur_len], skip_special_tokens=True)
print("\nOfficial Model: ",decoded)

####Custom Model####

qwenConfig = official_model.config
custom_model =  NNTrainerQwen3ForCausalLM(qwenConfig)
custom_model.load_state_dict(official_model.state_dict(),strict=False)

rotary_emb = Qwen3RotaryEmbedding(qwenConfig)

generated = input_ids.clone() # input to official Qwen model
cur_len =  enc.input_ids.size(1)

position_ids = torch.arange(generated.shape[1]).reshape(1, -1).repeat(generated.shape[0], 1)
cos, sin = rotary_emb(generated.to(torch.float32), position_ids)
cos, sin = torch.tensor(cos.numpy()), torch.tensor(sin.numpy())
variance_epsilon = torch.tensor([[1e-6,]])

cos = cos.reshape(max_len * 128)
sin = sin.reshape(max_len * 128)

generated.detach().numpy().astype(np.float32).tofile('./input_tokens.bin')
cos.detach().numpy().tofile('./rotary_embeddings_cosine.bin')
sin.detach().numpy().tofile('./rotary_embeddings_sine.bin')

for step in range(num_tokens_to_generate):  # generate 20 tokens

    outputs = custom_model(
        generated.reshape(max_len,1), 
        cos,
        sin,
        variance_epsilon
    )

    next_token_logits = outputs[0][:, cur_len - 1, :]
    next_token_id = torch.argmax(next_token_logits,dim=-1)
    generated[0][cur_len] = next_token_id
    cur_len += 1


decoded = tokenizer.decode(generated[0, :cur_len], skip_special_tokens=True)
print("\nCustom Model: ",decoded)

torch.onnx.export(
    custom_model, (generated.reshape(max_len,1), cos, sin, variance_epsilon),
    './qwen3_model.onnx',
    export_params=True,
    opset_version=17,
    input_names=['input', 'cos', 'sin', 'variance_epsilon'],
    output_names=['output'],
    keep_initializers_as_inputs=False,
    dynamic_axes=None,
 )

print("<Model exported successfully>")