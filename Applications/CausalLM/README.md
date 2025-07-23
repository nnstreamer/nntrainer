# ☄️ CausalLM Inference with NNTrainer

- This application provides examples to run causal llm models using nntrainer.
- This example only provides *inference* mode, not *training* mode yet.
- **Parameter-efficient Training** mode will be supported soon.

## Supported models

- Llama
- Qwen3 (1.7b/4b/7b/14b)
- Qwen3MoE (30b-A3b)
- You can try your own model with custom layers! 
- Feel free to contribute! 😊

## How to run

- download and copy the model files from hugingface to `res/{model}` directory.
- The folder should contain
    - config.json
    - generation_config.json
    - tokenizer.json
    - tokenizer_config.json
    - vocab.json
    - nntr_config.json
    - nntrainer weight binfile (matches with the name in nntr_config.json)
    - which are usuallyl included in HF model deployment.
- compile the Application
- run the model with the following command

```
$ cd build/Applications/CausalLM
$ ./nntr_causallm {your model config folder}
```

e.g.,

```
$ ./nntr_causallm /tmp/nntrainer/Applications/CausalLM/res/qwen3-4b/
```
