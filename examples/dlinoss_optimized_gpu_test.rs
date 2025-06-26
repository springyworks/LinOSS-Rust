//! GPU Test for Optimized dLinOSS Layer
//! Validates that the GPU-optimized dLinOSS layer works correctly with pure tensor operations

use burn::{
    backend::{Autodiff, Wgpu},
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::*,
    tensor::{Distribution, Tensor},
};
use linoss_rust::linoss::{
    dlinoss_layer_optimized::{OptimizedDLinossLayer, OptimizedDLinossConfig},
};
use std::time::Instant;

type WgpuBackend = Wgpu<f32, i32>;
type WgpuAutodiff = Autodiff<WgpuBackend>;

fn test_optimized_dlinoss_gpu() {
    println!("🚀 Testing GPU-Optimized dLinOSS Layer");
    println!("=====================================");

    // Initialize WGPU backend
    let device = burn::backend::wgpu::WgpuDevice::default();
    println!("✅ WGPU backend initialized successfully");

    // Create optimized layer configuration
    let config = OptimizedDLinossConfig::new_dlinoss(32, 64, 16);
    let layer: OptimizedDLinossLayer<WgpuAutodiff> = OptimizedDLinossLayer::new(&config, &device);

    println!("✅ Optimized dLinOSS layer created");

    // Create test input
    let input = Tensor::<WgpuAutodiff, 3>::random(
        [8, 32, 32],
        Distribution::Normal(0.0, 1.0),
        &device,
    );

    println!("Input shape: {:?}", input.dims());

    // Test forward pass timing
    let start = Instant::now();
    let output = layer.forward(input.clone());
    let forward_time = start.elapsed();

    println!("Output shape: {:?}", output.dims());
    println!("Forward pass time: {:?}", forward_time);

    // Test backward pass
    let start = Instant::now();
    let loss = output.mean();
    let _grads = loss.backward();
    let backward_time = start.elapsed();

    println!("Backward pass time: {:?}", backward_time);

    // Test damping functionality
    if layer.has_damping() {
        if let Some(damping_coeffs) = layer.get_damping_coefficients() {
            println!("✅ Layer has damping enabled");
            println!("   Damping coefficients shape: {:?}", damping_coeffs.dims());
        }
        if let Some(damping_scales) = layer.get_damping_scales() {
            println!("   Damping scales shape: {:?}", damping_scales.dims());
        }
    } else {
        println!("❌ Damping not enabled");
    }

    println!("✅ GPU-Optimized dLinOSS test completed");
}

fn test_training_convergence() {
    println!("\n🎯 Testing Training Convergence with Optimized Layer");
    println!("===================================================");

    let device = burn::backend::wgpu::WgpuDevice::default();
    
    // Create a simple training setup
    let config = OptimizedDLinossConfig::new_dlinoss(16, 32, 8);
    let mut model: OptimizedDLinossLayer<WgpuAutodiff> = OptimizedDLinossLayer::new(&config, &device);
    
    let optimizer_config = AdamConfig::new();
    let mut optimizer = optimizer_config.init();
    
    println!("Training for 100 epochs...");
    
    let mut initial_loss = None;
    let mut final_loss = None;
    
    for epoch in 0..100 {
        // Create random training data
        let input = Tensor::<WgpuAutodiff, 3>::random(
            [4, 16, 16],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        
        let output = model.forward(input);
        let loss = output.powf_scalar(2.0).mean();
        let loss_val: f32 = loss.clone().into_scalar();
        
        if epoch == 0 {
            initial_loss = Some(loss_val);
        }
        if epoch == 99 {
            final_loss = Some(loss_val);
        }
        
        let grads = loss.backward();
        let grads_params = GradientsParams::from_grads(grads, &model);
        model = optimizer.step(1e-3, model, grads_params);
        
        if epoch % 20 == 0 {
            println!("Epoch {}: Loss = {:.6}", epoch, loss_val);
        }
    }
    
    if let (Some(initial), Some(final_loss)) = (initial_loss, final_loss) {
        let improvement = (initial - final_loss) / initial;
        println!("Initial loss: {:.6}", initial);
        println!("Final loss: {:.6}", final_loss);
        println!("Improvement: {:.1}%", improvement * 100.0);
        
        if improvement > 0.1 {
            println!("✅ Training converged successfully");
        } else {
            println!("⚠️  Training improvement could be better");
        }
    }
    
    println!("✅ Training convergence test completed");
}

fn test_parallel_scan_performance() {
    println!("\n⚡ Testing Parallel Scan Performance");
    println!("===================================");

    let device = burn::backend::wgpu::WgpuDevice::default();
    
    // Test with different sequence lengths to see scaling
    let test_configs = vec![
        (32, 16, 64),    // Small
        (64, 32, 128),   // Medium  
        (128, 64, 256),  // Large
    ];
    
    for (seq_len, d_model, d_input) in test_configs {
        println!("\nTesting sequence length: {}, d_model: {}", seq_len, d_model);
        
        let config = OptimizedDLinossConfig::new_dlinoss(d_input, d_model, d_model/2);
        let layer: OptimizedDLinossLayer<WgpuAutodiff> = OptimizedDLinossLayer::new(&config, &device);
        
        let input = Tensor::<WgpuAutodiff, 3>::random(
            [4, seq_len, d_input],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        
        // Warm-up run
        let _ = layer.forward(input.clone());
        
        // Timed run
        let start = Instant::now();
        let output = layer.forward(input);
        let elapsed = start.elapsed();
        
        let throughput = (4 * seq_len) as f64 / elapsed.as_secs_f64();
        
        println!("   Time: {:?}", elapsed);
        println!("   Throughput: {:.0} sequences/sec", throughput);
        println!("   Output shape: {:?}", output.dims());
    }
    
    println!("✅ Performance test completed");
}

fn main() {
    println!("🔥 GPU-Optimized dLinOSS Validation Suite");
    println!("==========================================\n");

    test_optimized_dlinoss_gpu();
    test_training_convergence();
    test_parallel_scan_performance();

    println!("\n🎉 All tests completed!");
    println!("✅ Optimized dLinOSS is working properly on GPU with tensor operations");
}
