//! Quantized tensor operations example
//!
//! Demonstrates KORE's unique ultra-low-bit quantization formats.

use kore_core::{Tensor, DType};

fn main() {
    println!("=== KORE Quantized Operations Example ===\n");

    // KORE supports two unique quantization formats:
    // 1. Ternary (1.58-bit): 5 values per byte, values in {-1, 0, +1}
    // 2. Quaternary (2-bit): 4 values per byte, values in {-1, -1/3, +1/3, +1}

    println!("Quantization Formats:");
    println!("  Ternary (1.58-bit): 5 trits per byte");
    println!("  Quaternary (2-bit): 4 values per byte");
    println!();

    // Create regular FP32 tensor
    let fp32_tensor = Tensor::randn(&[1024, 1024]);
    println!("FP32 tensor: {:?}", fp32_tensor.shape().dims());
    println!("  Storage: {:?}", fp32_tensor.dtype());

    // In production, you would quantize using:
    // let ternary_weights = fp32_tensor.quantize(DType::Ternary);
    // let quaternary_weights = fp32_tensor.quantize(DType::Quaternary);

    // For this example, we show the storage format
    println!("\nStorage Format Comparison:");
    
    let numel = 1024 * 1024;
    let fp32_bytes = DType::F32.storage_bytes(numel);
    let ternary_bytes = DType::Ternary.storage_bytes(numel);
    let quaternary_bytes = DType::Quaternary.storage_bytes(numel);

    println!("  FP32 (32-bit):        {} bytes", fp32_bytes);
    println!("  Ternary (1.58-bit):   {} bytes ({:.1}% of FP32)", 
        ternary_bytes, 100.0 * ternary_bytes as f64 / fp32_bytes as f64);
    println!("  Quaternary (2-bit):   {} bytes ({:.1}% of FP32)", 
        quaternary_bytes, 100.0 * quaternary_bytes as f64 / fp32_bytes as f64);

    println!("\nMemory Savings:");
    println!("  Ternary:  {:.1}x smaller", fp32_bytes as f64 / ternary_bytes as f64);
    println!("  Quaternary: {:.1}x smaller", fp32_bytes as f64 / quaternary_bytes as f64);

    #[cfg(feature = "kore-vulkan")]
    {
        use kore_vulkan::VulkanBackend;
        
        println!("\n=== Vulkan Quantized MatMul ===");
        
        if let Ok(backend) = VulkanBackend::new() {
            let activations = Tensor::randn(&[256, 512]);
            let scales = Tensor::ones(&[512]);
            
            // Note: In production, weights would be pre-quantized
            // let ternary_weights = ...;
            // let result = backend.quantized_matmul(&activations, &ternary_weights, &scales)?;
            
            println!("✅ Vulkan backend available for quantized operations");
        } else {
            println!("⚠️  Vulkan backend not available");
        }
    }

    println!("\n=== Example Complete ===");
}
