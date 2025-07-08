# Qwen3 MoE Application

This application is an implementation of Qwen3 MoE (Mixture of Experts) model using nntrainer.

## Overview

Qwen3 MoE is a large language model that uses the Mixture-of-Experts architecture to achieve efficient scaling. This implementation includes:

- **Custom Layers for Qwen3 MoE**:
  - SiLU (Sigmoid Linear Unit) activation layer
  - RMS Normalization layer  
  - Qwen3 MoE MLP layer with gate/up/down projections
  - (Simplified) Sparse MoE block for demo

- **Model Configuration**:
  - Vocabulary size: 151,936
  - Hidden size: 2,048
  - 48 layers (2 layers in demo mode)
  - 32 attention heads (Q), 4 key-value heads (GQA)
  - 128 experts (4 experts in demo mode)
  - 8 experts activated per token
  - Context length: 32,768 tokens

## Requirements

Please refer to the following documents for instructions on setting up and building nntrainer:

- [Getting Started](../../docs/getting-started.md)
- [How to Run Example Android](../../docs/how-to-run-example-android.md)

### Dependencies

- nntrainer (latest version)
- C++17 compatible compiler
- Meson build system

## Building

```bash
# From the nntrainer root directory
meson build
cd build
ninja Applications/Qwen3MoE/jni/nntrainer_qwen3_moe
```

## Usage

```bash
./nntrainer_qwen3_moe <weight_path> [demo_mode] [temperature]
```

### Parameters

- `weight_path`: Path to model weights file (.bin format)
- `demo_mode`: 0 or 1 (default: 1 for simplified demo with fewer layers)
- `temperature`: Sampling temperature for text generation (default: 1.0)

### Examples

```bash
# Run in demo mode with simplified model
./nntrainer_qwen3_moe weights.bin 1 0.8

# Run full model (requires substantial memory)
./nntrainer_qwen3_moe weights.bin 0 1.0

# Show help
./nntrainer_qwen3_moe
```

## Model Architecture

### Implemented Components

✅ **Working Components**:
- Token Embedding
- RMS Normalization  
- SiLU Activation
- Qwen3 MoE MLP (Gate + Up + Down projections)
- Basic model structure

❌ **Not Yet Implemented** (marked as TODO):
- Grouped Query Attention (GQA)
- Complete Sparse MoE Block with routing
- Rotary Position Embedding (can reuse from LLaMA)
- Load balancing loss

### Demo Mode vs Full Model

**Demo Mode** (`demo_mode=1`):
- 2 decoder layers instead of 48
- 4 experts instead of 128  
- Regular MLP instead of Sparse MoE
- Suitable for testing and development

**Full Model** (`demo_mode=0`):
- Complete 48-layer architecture
- 128 experts with sparse routing
- Requires significant computational resources

## File Structure

```
Applications/Qwen3MoE/
├── README.md                    # This file
├── jni/
│   ├── main.cpp                 # Main application
│   ├── silu_layer.h/.cpp        # SiLU activation
│   ├── rms_norm_layer.h/.cpp    # RMS normalization
│   ├── qwen3_moe_mlp_layer.h/.cpp # MoE MLP layer
│   └── meson.build              # Build configuration
└── qwen3_moe_analysis.md        # Detailed analysis
```

## Implementation Status

### Phase 1: Basic Components ✅
- [x] SiLU Activation Layer
- [x] RMS Normalization Layer  
- [x] Qwen3 MoE MLP Layer
- [x] Basic model structure

### Phase 2: Advanced Components ⏳
- [ ] Grouped Query Attention (GQA)
- [ ] Sparse MoE Block with routing
- [ ] Expert selection and load balancing
- [ ] Rotary Position Embedding integration

### Phase 3: Optimization 📋
- [ ] Expert parallelization
- [ ] Memory optimization
- [ ] Performance tuning
- [ ] Full model validation

## Weight Conversion

To use pre-trained Qwen3 MoE weights, you'll need to convert them from PyTorch/HuggingFace format to nntrainer format. A conversion script will be provided in future updates.

## Limitations

- Demo mode only implements a simplified version
- Full MoE routing not yet implemented
- No pre-trained weights conversion tool yet
- Limited to inference only (no training support)

## References

- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- [Qwen3 MoE HuggingFace](https://huggingface.co/Qwen/Qwen3-30B-A3B-Base)
- [nntrainer Documentation](https://github.com/nnstreamer/nntrainer)

## Contributing

This is a work-in-progress implementation. Contributions are welcome, especially for:

- Grouped Query Attention implementation
- Complete Sparse MoE Block
- Weight conversion utilities  
- Performance optimizations

## License

Apache-2.0 License - see the [LICENSE](../../../LICENSE) file for details.