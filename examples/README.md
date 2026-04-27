# KORE Examples

This directory contains example code demonstrating various KORE features.

## Examples

### Basic Tensor Operations
- `basic_tensor_ops.rs` - Basic tensor creation and manipulation
- `matmul.rs` - Matrix multiplication examples
- `quantized_ops.rs` - Using ternary and quaternary quantization

### Neural Networks
- `mlp.rs` - Multi-layer perceptron training
- `cnn.rs` - Convolutional neural network
- `transformer.rs` - Transformer model inference

### Vulkan GPU Backend
- `vulkan_matmul.rs` - GPU-accelerated matrix multiplication
- `vulkan_quantized.rs` - Quantized inference on Vulkan

### Advanced Features
- `autograd.rs` - Automatic differentiation
- `checkpoint.rs` - Gradient checkpointing
- `bf16_training.rs` - Brain-float 16 training

## Running Examples

```bash
# Run a specific example
cargo run --example basic_tensor_ops

# Run with specific features
cargo run --example vulkan_matmul --features ash
```

## Prerequisites

- Rust 1.75+
- For Vulkan examples: GPU with Vulkan 1.2+ support
- For CUDA examples: NVIDIA GPU with CUDA toolkit
