//! Benchmark suite for kore-vulkan operations
//!
//! Compares Vulkan compute performance against CPU baseline.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use kore_core::Tensor;
use kore_vulkan::VulkanBackend;

fn bench_matmul(c: &mut Criterion) {
    let backend = VulkanBackend::new().expect("Failed to create Vulkan backend");
    
    let mut group = c.benchmark_group("matmul");
    
    for size in [256, 512, 1024, 2048].iter() {
        let size = *size;
        
        // Create random tensors
        let a = Tensor::randn(&[size, size]);
        let b = Tensor::randn(&[size, size]);
        
        // CPU baseline (through KORE's default implementation)
        group.bench_with_input(
            BenchmarkId::new("cpu", size),
            &size,
            |bencher, _| {
                bencher.iter(|| {
                    // Note: Using a simple CPU matmul for comparison
                    // In full KORE, this would dispatch to kore-kernels
                    let _result = black_box(&a).matmul(black_box(&b));
                });
            },
        );
        
        // Vulkan implementation
        group.bench_with_input(
            BenchmarkId::new("vulkan", size),
            &size,
            |bencher, _| {
                bencher.iter(|| {
                    let _result = black_box(backend.matmul(black_box(&a), black_box(&b)));
                });
            },
        );
    }
    
    group.finish();
}

fn bench_elementwise(c: &mut Criterion) {
    let backend = VulkanBackend::new().expect("Failed to create Vulkan backend");
    
    let mut group = c.benchmark_group("elementwise_add");
    
    for size in [1024, 4096, 16384, 65536, 262144].iter() {
        let size = *size;
        
        let a = Tensor::randn(&[size]);
        let b = Tensor::randn(&[size]);
        
        // CPU baseline
        group.bench_with_input(
            BenchmarkId::new("cpu", size),
            &size,
            |bencher, _| {
                bencher.iter(|| {
                    let _result = black_box(&a).add(black_box(&b));
                });
            },
        );
        
        // Vulkan implementation
        group.bench_with_input(
            BenchmarkId::new("vulkan", size),
            &size,
            |bencher, _| {
                bencher.iter(|| {
                    let _result = black_box(backend.add(black_box(&a), black_box(&b)));
                });
            },
        );
    }
    
    group.finish();
}

fn bench_quantized_matmul(c: &mut Criterion) {
    let backend = VulkanBackend::new().expect("Failed to create Vulkan backend");
    
    let mut group = c.benchmark_group("quantized_matmul");
    
    for (m, n, k) in [(512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048)].iter() {
        // Create FP32 activations
        let a = Tensor::randn(&[*m, *k]);
        
        // Create dummy quantized weights and scales
        // In real benchmark, these would be actual quantized tensors
        let b = Tensor::randn(&[*k, *n]);  // Placeholder
        let scales = Tensor::ones(&[*n]);  // Placeholder scales
        
        group.bench_with_input(
            BenchmarkId::new("vulkan_quantized", format!("{}x{}x{}", m, n, k)),
            &(*m, *n, *k),
            |bencher, _| {
                bencher.iter(|| {
                    let _result = black_box(
                        backend.quantized_matmul(black_box(&a), black_box(&b), black_box(&scales))
                    );
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(benches, bench_matmul, bench_elementwise, bench_quantized_matmul);
criterion_main!(benches);
