# LLaMA

This application is an implementation of LLaMA.

It has been verified to run in Ubuntu and Android environments through meson build.

Reference:
- [Paper: The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)

## How to run LLaMA

Please refer to the following documents for instructions on setting up and building NNTrainer.

- [Getting Started](../../docs/getting-started.md)
- [how-to-run-example-android](../../docs/how-to-run-example-android.md)

To run your own LLaMA model, you need to prepare three additional files inside the application build directory.
- For tokenizer usage, you will require a `vocab.json` file and a `merges.txt` file tailored to your specific model.
- To load pre-trained weights, ensure that you have a `.bin` file compatible with NNTrainer. You can convert your PyTorch model's weight file to NNTrainer model's weights file using the `Applications/LLaMA/PyTorch/weights_converter.py` script provided within our official repository.
