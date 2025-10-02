import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from modelling_qwen3 import  NNTrainerQwen3ForCausalLM, Qwen3RotaryEmbedding

model_name = "Qwen/Qwen3-1.7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)


####Offical Model####


official_model = AutoModelForCausalLM.from_pretrained(model_name).eval()

# Fixed maximum length
max_len = 1024

# Prompt
prompt = "Tell me a dad joke about a computer: "
print("\nInput prompt: ",prompt)

enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len)
cur_len = enc.input_ids.size(1)

input_ids = torch.full((1, max_len), tokenizer.pad_token_id)  # preallocate with PAD
input_ids[0, :cur_len] = enc.input_ids[0]

generated = input_ids.clone()

for step in range(20):  # generate 20 tokens
    
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

# Fixed causal mask (constant size)
causal_mask = torch.full((1, 1, max_len,max_len), float("-inf"))
causal_mask = torch.triu(causal_mask, diagonal=1) 

qwenConfig = official_model.config
custom_model =  NNTrainerQwen3ForCausalLM(qwenConfig)
custom_model.load_state_dict(official_model.state_dict())

rotary_emb = Qwen3RotaryEmbedding(qwenConfig)

generated = input_ids.clone() # input to official Qwen model
cur_len =  enc.input_ids.size(1)

position_ids = torch.arange(generated.shape[1]).view(1, -1).repeat(generated.shape[0], 1)
cos, sin = rotary_emb(generated.to(torch.float32), position_ids)
cos, sin = torch.tensor(cos.numpy()), torch.tensor(sin.numpy())
variance_epsilon = torch.tensor([[1e-6,]])

for step in range(20):  # generate 20 tokens
    
    outputs = custom_model(
        generated, 
        cos,
        sin,
        variance_epsilon,
        causal_mask
    )
    
    next_token_logits = outputs[0][:, cur_len - 1, :]
    next_token_id = torch.argmax(next_token_logits, dim=-1)

    generated[0][cur_len] = next_token_id
    cur_len += 1


decoded = tokenizer.decode(generated[0, :cur_len], skip_special_tokens=True)
print("\nCustom Model: ",decoded)
   

