//! Vulkan GPU-accelerated matrix multiplication
//!
//! This example demonstrates using the Vulkan backend for GPU computation.
//!
//! Run with: cargo run --example vulkan_matmul --features ash

#[cfg(feature = "kore-vulkan")]
use kore_vulkan::VulkanBackend;
use kore_core::Tensor;

fn main() {
    println!("=== KORE Vulkan GPU MatMul Example ===\n");

    #[cfg(not(feature = "kore-vulkan"))]
    {
        println!("This example requires the kore-vulkan crate.");
        println!("Run with: cargo run --example vulkan_matmul --features ash");
        return;
    }

    #[cfg(feature = "kore-vulkan")]
    {
        // Initialize Vulkan backend
        let backend = match VulkanBackend::new() {
            Ok(b) => {
                println!("✅ Vulkan backend initialized successfully");
                b
            }
            Err(e) => {
                println!("❌ Failed to initialize Vulkan backend: {}", e);
                println!("Make sure you have a Vulkan-capable GPU and drivers installed.");
                return;
            }
        };

        // Create test matrices
        let size = 1024;
        println!("\nCreating {}x{} matrices...", size, size);
        
        let a = Tensor::randn(&[size, size]);
        let b = Tensor::randn(&[size, size]);
        
        println!("✅ Matrices created");

        // Perform GPU matrix multiplication
        println!("\nPerforming GPU matrix multiplication...");
        let start = std::time::Instant::now();
        
        let c = match backend.matmul(&a, &b) {
            Ok(result) => {
                let elapsed = start.elapsed();
                println!("✅ GPU matmul completed in {:?}", elapsed);
                println!("   Result shape: {:?}", result.shape().dims());
                result
            }
            Err(e) => {
                println!("❌ GPU matmul failed: {}", e);
                return;
            }
        };

        // Compare with CPU implementation (optional)
        println!("\nVerifying result on CPU...");
        let cpu_start = std::time::Instant::now();
        let cpu_result = a.matmul(&b).expect("CPU matmul failed");
        let cpu_elapsed = cpu_start.elapsed();
        
        println!("   CPU matmul completed in {:?}", cpu_elapsed);
        println!("   Speedup: {:.2}x", cpu_elapsed.as_secs_f64() / start.elapsed().as_secs_f64());

        println!("\n=== Example Complete ===");
    }
}
