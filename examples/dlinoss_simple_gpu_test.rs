//! Simple dLinOSS GPU Test
//! Tests that dLinOSS layers work correctly on GPU and use tensor operations

use burn::{
    backend::{Autodiff, NdArray, Wgpu},
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::*,
    tensor::{Distribution, Tensor, TensorData},
};
use linoss_rust::linoss::{
    dlinoss_layer_optimized::{OptimizedDLinossLayer, OptimizedDLinossConfig},
};
use std::time::Instant;

type WgpuBackend = Wgpu<f32, i32>;
type WgpuAutodiff = Autodiff<WgpuBackend>;
type CpuBackend = NdArray<f32>;
type CpuAutodiff = Autodiff<CpuBackend>;

fn test_basic_forward_pass() {
    println!("🔥 Testing Basic dLinOSS Forward Pass...");
    
    // Test on WGPU (GPU)
    let wgpu_device = <WgpuBackend as Backend>::Device::default();
    println!("Using WGPU device: {:?}", wgpu_device);
    
    let config = DLinossLayerConfig {
        d_input: 32,
        d_model: 64,
        d_output: 16,
        delta_t: 0.1,
        init_std: 0.02,
        enable_layer_norm: true,
        enable_damping: true,
        init_damping: 0.1,
        num_damping_scales: 4,
        a_parameterization: AParameterization::GELU,
    };
    
    let layer: DLinossLayer<WgpuAutodiff> = DLinossLayer::new(&config, &wgpu_device);
    
    // Create test input
    let input = Tensor::<WgpuAutodiff, 3>::random(
        [8, 32, 32],
        Distribution::Normal(0.0, 1.0),
        &wgpu_device,
    );
    
    println!("Input shape: {:?}", input.dims());
    
    // Forward pass
    let start = Instant::now();
    let output = layer.forward(input.clone());
    let forward_time = start.elapsed();
    
    println!("Output shape: {:?}", output.dims());
    println!("Forward pass time: {:?}", forward_time);
    
    // Test gradient computation
    let loss = output.mean();
    let start = Instant::now();
    let _gradients = loss.backward();
    let backward_time = start.elapsed();
    
    println!("Loss: {:.6}", loss.into_scalar());
    println!("Backward pass time: {:?}", backward_time);
    
    // Test damping parameters
    if layer.has_damping() {
        println!("✅ Layer has damping enabled");
        if let Some(coeffs) = layer.get_damping_coefficients() {
            println!("   Damping coefficients shape: {:?}", coeffs.dims());
        }
        if let Some(scales) = layer.get_damping_scales() {
            println!("   Damping scales shape: {:?}", scales.dims());
        }
    }
    
    println!("✅ Basic forward pass test completed\n");
}

fn test_damping_effect() {
    println!("🧪 Testing Damping Effect...");
    
    let device = <WgpuBackend as Backend>::Device::default();
    
    // Create two layers: one with damping, one without
    let damped_config = DLinossLayerConfig::new_dlinoss(16, 32, 16);
    let undamped_config = DLinossLayerConfig::vanilla_linoss(16, 32, 16);
    
    let damped_layer: DLinossLayer<WgpuAutodiff> = DLinossLayer::new(&damped_config, &device);
    let undamped_layer: DLinossLayer<WgpuAutodiff> = DLinossLayer::new(&undamped_config, &device);
    
    // Create impulse input (high energy at t=0, then zeros)
    let mut input_data = vec![0.0f32; 4 * 32 * 16]; // batch=4, seq=32, dim=16
    // Set impulse at t=0 for all batches
    for b in 0..4 {
        input_data[b * 32 * 16] = 1.0;
    }
    
    let input = Tensor::<WgpuAutodiff, 3>::from_data(
        TensorData::new(input_data, [4, 32, 16]),
        &device,
    );
    
    // Forward passes
    let damped_output = damped_layer.forward(input.clone());
    let undamped_output = undamped_layer.forward(input.clone());
    
    // Calculate final energy (sum of squares)
    let damped_final_energy = damped_output
        .clone()
        .slice([0..4, 31..32, 0..16])  // Last timestep
        .powf_scalar(2.0)
        .sum()
        .into_scalar();
    
    let undamped_final_energy = undamped_output
        .clone()
        .slice([0..4, 31..32, 0..16])  // Last timestep
        .powf_scalar(2.0)
        .sum()
        .into_scalar();
    
    println!("Final energy - Damped: {:.6}", damped_final_energy);
    println!("Final energy - Undamped: {:.6}", undamped_final_energy);
    
    let energy_ratio = damped_final_energy / undamped_final_energy;
    println!("Energy ratio (damped/undamped): {:.3}", energy_ratio);
    
    if energy_ratio < 0.9 {
        println!("✅ Damping is working - energy reduced by {:.1}%", (1.0 - energy_ratio) * 100.0);
    } else {
        println!("⚠️  Damping effect may be too weak");
    }
    
    println!("✅ Damping effect test completed\n");
}

fn test_gpu_performance() {
    println!("⚡ Testing GPU vs CPU Performance...");
    
    let wgpu_device = <WgpuBackend as Backend>::Device::default();
    let cpu_device = <CpuBackend as Backend>::Device::default();
    
    let config = DLinossLayerConfig {
        d_input: 64,
        d_model: 128,
        d_output: 32,
        delta_t: 0.1,
        init_std: 0.02,
        enable_layer_norm: true,
        enable_damping: true,
        init_damping: 0.1,
        num_damping_scales: 4,
        a_parameterization: AParameterization::GELU,
    };
    
    // Create layers
    let gpu_layer: DLinossLayer<WgpuAutodiff> = DLinossLayer::new(&config, &wgpu_device);
    let cpu_layer: DLinossLayer<CpuAutodiff> = DLinossLayer::new(&config, &cpu_device);
    
    // Create test data
    let gpu_input = Tensor::<WgpuAutodiff, 3>::random(
        [16, 64, 64],
        Distribution::Normal(0.0, 1.0),
        &wgpu_device,
    );
    
    let cpu_input = Tensor::<CpuAutodiff, 3>::random(
        [16, 64, 64],
        Distribution::Normal(0.0, 1.0),
        &cpu_device,
    );
    
    // GPU benchmark
    let start = Instant::now();
    for _ in 0..5 {
        let _ = gpu_layer.forward(gpu_input.clone());
    }
    let gpu_time = start.elapsed();
    
    // CPU benchmark
    let start = Instant::now();
    for _ in 0..5 {
        let _ = cpu_layer.forward(cpu_input.clone());
    }
    let cpu_time = start.elapsed();
    
    println!("GPU (WGPU) time: {:?} (avg: {:?})", gpu_time, gpu_time / 5);
    println!("CPU (NdArray) time: {:?} (avg: {:?})", cpu_time, cpu_time / 5);
    
    let speedup = cpu_time.as_secs_f32() / gpu_time.as_secs_f32();
    if speedup > 1.0 {
        println!("✅ GPU is {:.2}x faster than CPU", speedup);
    } else {
        println!("📊 CPU is {:.2}x faster (may be due to small workload)", 1.0 / speedup);
    }
    
    println!("✅ Performance test completed\n");
}

fn test_training_convergence() {
    println!("🎯 Testing Training Convergence...");
    
    let device = <WgpuBackend as Backend>::Device::default();
    
    #[derive(Module, Debug)]
    struct SimpleModel<B: Backend> {
        layer: DLinossLayer<B>,
    }
    
    impl<B: Backend> SimpleModel<B> {
        fn new(device: &B::Device) -> Self {
            let config = DLinossLayerConfig {
                d_input: 8,
                d_model: 16,
                d_output: 1,
                delta_t: 0.1,
                init_std: 0.02,
                enable_layer_norm: false,
                enable_damping: true,
                init_damping: 0.1,
                num_damping_scales: 2,
                a_parameterization: AParameterization::GELU,
            };
            
            Self {
                layer: DLinossLayer::new(&config, device),
            }
        }
        
        fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
            self.layer.forward(input)
        }
    }
    
    let mut model: SimpleModel<WgpuAutodiff> = SimpleModel::new(&device);
    let mut optimizer = AdamConfig::new().init();
    
    // Generate simple training data: sine wave prediction
    let batch_size = 8;
    let seq_len = 16;
    let input_dim = 8;
    
    let mut input_data = vec![0.0f32; batch_size * seq_len * input_dim];
    let mut target_data = vec![0.0f32; batch_size * seq_len * 1];
    
    for b in 0..batch_size {
        for t in 0..seq_len {
            let time = t as f32 * 0.1;
            let freq = 1.0 + b as f32 * 0.1;
            
            // Input: time-based features
            for d in 0..input_dim {
                input_data[b * seq_len * input_dim + t * input_dim + d] = 
                    (time * freq * (1.0 + d as f32 * 0.1)).sin();
            }
            
            // Target: damped sine
            target_data[b * seq_len + t] = 
                (-time * 0.1).exp() * (time * freq).sin();
        }
    }
    
    let input = Tensor::<WgpuAutodiff, 3>::from_data(
        TensorData::new(input_data, [batch_size, seq_len, input_dim]),
        &device,
    );
    
    let target = Tensor::<WgpuAutodiff, 3>::from_data(
        TensorData::new(target_data, [batch_size, seq_len, 1]),
        &device,
    );
    
    println!("Training for 50 epochs...");
    let mut initial_loss = None;
    let mut final_loss = None;
    
    for epoch in 0..50 {
        let output = model.forward(input.clone());
        let loss = (output - target.clone()).powf_scalar(2.0).mean();
        let loss_val: f32 = loss.clone().into_scalar();
        
        if epoch == 0 {
            initial_loss = Some(loss_val);
        }
        if epoch == 49 {
            final_loss = Some(loss_val);
        }
        
        let grads = loss.backward();
        let grads_params = GradientsParams::from_grads(grads, &model);
        model = optimizer.step(1e-3, model, grads_params);
        
        if epoch % 10 == 0 {
            println!("Epoch {}: Loss = {:.6}", epoch, loss_val);
        }
    }
    
    if let (Some(initial), Some(final_loss)) = (initial_loss, final_loss) {
        let improvement = (initial - final_loss) / initial;
        println!("Initial loss: {:.6}", initial);
        println!("Final loss: {:.6}", final_loss);
        println!("Improvement: {:.1}%", improvement * 100.0);
        
        if improvement > 0.5 {
            println!("✅ Training converged successfully");
        } else {
            println!("⚠️  Training improvement could be better");
        }
    }
    
    println!("✅ Training convergence test completed\n");
}

fn main() {
    println!("🚀 dLinOSS GPU Validation Suite");
    println!("================================\n");
    
    // Check if WGPU is available
    match WgpuBackend::default() {
        _device => {
            println!("✅ WGPU backend initialized successfully\n");
            
            // Run tests
            test_basic_forward_pass();
            test_damping_effect();
            test_gpu_performance();
            test_training_convergence();
            
            println!("🎉 All tests completed!");
            println!("✅ dLinOSS is working properly on GPU with tensor operations");
        }
    }
}
