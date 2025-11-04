import numpy as np
from transformers import AutoTokenizer

model_name = "Qwen/Qwen3-1.7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

generated = np.fromfile("./output_tokens.bin",dtype="float32").astype('int32')

decoded = tokenizer.decode(generated, skip_special_tokens=True)
print("NNTrainer model: ",decoded)