//! Basic tensor operations example
//!
//! Demonstrates tensor creation, manipulation, and basic operations.

use kore_core::Tensor;

fn main() {
    println!("=== KORE Basic Tensor Operations ===\n");

    // Create tensors from data
    let a = Tensor::from_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::from_f32(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);

    println!("Tensor A (2x2):");
    println!("  Shape: {:?}", a.shape().dims());
    println!("  Num elements: {}", a.numel());

    println!("\nTensor B (2x2):");
    println!("  Shape: {:?}", b.shape().dims());
    println!("  Num elements: {}", b.numel());

    // Element-wise operations
    let c = a.add(&b).expect("Failed to add tensors");
    println!("\nA + B:");
    println!("  Result shape: {:?}", c.shape().dims());

    let d = a.mul(&b).expect("Failed to multiply tensors");
    println!("\nA * B (element-wise):");
    println!("  Result shape: {:?}", d.shape().dims());

    // Matrix multiplication
    let e = a.matmul(&b).expect("Failed to matmul tensors");
    println!("\nA @ B (matrix multiplication):");
    println!("  Result shape: {:?}", e.shape().dims());

    // Reshape (zero-copy view)
    let flat = a.reshape(&[4]).expect("Failed to reshape");
    println!("\nA reshaped to [4]:");
    println!("  Shape: {:?}", flat.shape().dims());
    println!("  Is contiguous: {}", flat.is_contiguous());

    // Create zeros and ones
    let zeros = Tensor::zeros(&[2, 3], kore_core::DType::F32);
    let ones = Tensor::ones(&[2, 3]);
    println!("\nZeros tensor (2x3):");
    println!("  Shape: {:?}", zeros.shape().dims());
    println!("\nOnes tensor (2x3):");
    println!("  Shape: {:?}", ones.shape().dims());

    // Random tensors
    let rand = Tensor::randn(&[3, 3]);
    println!("\nRandom normal tensor (3x3):");
    println!("  Shape: {:?}", rand.shape().dims());

    println!("\n=== Example Complete ===");
}
