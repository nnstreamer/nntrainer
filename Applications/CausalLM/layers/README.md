# CausalLM Custom Layers

This directory contains custom layer implementations for the CausalLM application.

## Layer Files

The following layer files should be implemented for full CausalLM functionality:

- `tie_word_embedding.cpp` - Tie word embedding layer implementation
- `swiglu.cpp` - SwiGLU activation layer implementation  
- `mha_core.cpp` - Multi-head attention core layer implementation
- `embedding_layer.cpp` - Embedding layer implementation
- `reshaped_rms_norm.cpp` - Reshaped RMS normalization layer implementation
- `qwen_moe_layer.cpp` - Qwen Mixture of Experts layer implementation

## Android Build Support

These layers are designed to work with both native and Android builds. The meson build system will automatically compile and link these layers when building CausalLM.

## Dependencies

The layers depend on the main nntrainer library and may require additional dependencies for tokenization and mathematical operations.