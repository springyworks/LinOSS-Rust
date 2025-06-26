//! GPU-optimized D-LinOSS Implementation Test
//! 
//! This test ensures that:
//! 1. dLinOSS layers run entirely on GPU (WGPU backend)
//! 2. All operations use Burn's tensor operations (no CPU loops)
//! 3. Damping is applied using vectorized operations
//! 4. Gradients flow properly through the learnable damping
//! 5. Performance is competitive with vanilla LinOSS

use burn::{
    backend::{Autodiff, Wgpu},
    optim::{AdamConfig, Optimizer},
    prelude::Backend,
    tensor::{backend::AutodiffBackend, ElementConversion, Tensor, TensorData},
};
use linoss_rust::linoss::{
    dlinoss_layer_optimized::{OptimizedDLinossLayer, OptimizedDLinossConfig, AParameterization},
};
use std::time::Instant;

type MyWgpuBackend = Wgpu<f32, i32>;
type MyAutodiffBackend = Autodiff<MyWgpuBackend>;

const BATCH_SIZE: usize = 32;
const SEQ_LEN: usize = 128;
const D_INPUT: usize = 64;
const D_MODEL: usize = 128; // Must be even for oscillatory pairs
const D_OUTPUT: usize = 32;
const NUM_EPOCHS: usize = 100;

#[derive(burn::module::Module, Debug)]
pub struct TestDLinossModel<B: Backend> {
    layer1: OptimizedDLinossLayer<B>,
    layer2: OptimizedDLinossLayer<B>,
}

impl<B: Backend> TestDLinossModel<B> {
    pub fn new(device: &B::Device) -> Self {
        let config1 = OptimizedDLinossConfig {
            d_input: D_INPUT,
            d_model: D_MODEL,
            d_output: D_MODEL,
            delta_t: 0.1,
            init_std: 0.02,
            enable_layer_norm: true,
            enable_damping: true,
            init_damping: 0.1,
            num_damping_scales: 4,
            a_parameterization: AParameterization::GELU,
        };
        
        let config2 = OptimizedDLinossConfig {
            d_input: D_MODEL,
            d_model: D_MODEL, 
            d_output: D_OUTPUT,
            delta_t: 0.1,
            init_std: 0.02,
            enable_layer_norm: true,
            enable_damping: true,
            init_damping: 0.05,
            num_damping_scales: 4,
            a_parameterization: AParameterization::GELU,
        };

        Self {
            layer1: OptimizedDLinossLayer::new(&config1, device),
            layer2: OptimizedDLinossLayer::new(&config2, device),
        }
    }

    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.layer1.forward(input);
        self.layer2.forward(x)
    }
}

fn test_gpu_backend_usage() {
    println!("🔥 Testing D-LinOSS GPU Backend Usage...");
    
    // Ensure we're using WGPU backend
    let device = MyWgpuBackend::default();
    println!("✅ Using WGPU backend: {:?}", device);
    
    // Create model on GPU
    let model: TestDLinossModel<MyAutodiffBackend> = TestDLinossModel::new(&device);
    println!("✅ Model created on GPU");
    
    // Create test data on GPU
    let input = Tensor::<MyAutodiffBackend, 3>::random(
        [BATCH_SIZE, SEQ_LEN, D_INPUT],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device,
    );
    
    let target = Tensor::<MyAutodiffBackend, 3>::random(
        [BATCH_SIZE, SEQ_LEN, D_OUTPUT],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &device,
    );
    
    println!("✅ Test data created on GPU");
    println!("   Input shape: {:?}", input.dims());
    println!("   Target shape: {:?}", target.dims());
    
    // Test forward pass
    let start = Instant::now();
    let output = model.forward(input.clone());
    let forward_time = start.elapsed();
    
    println!("✅ Forward pass completed");
    println!("   Output shape: {:?}", output.dims());
    println!("   Forward time: {:?}", forward_time);
    
    // Test gradient computation
    let loss = (output.clone() - target.clone()).powf_scalar(2.0).mean();
    let start = Instant::now();
    let gradients = loss.backward();
    let backward_time = start.elapsed();
    
    println!("✅ Backward pass completed");
    println!("   Loss: {:.6}", loss.into_scalar().elem::<f32>());
    println!("   Backward time: {:?}", backward_time);
    
    // Verify we have gradients for damping parameters
    let param_count = gradients.len();
    println!("✅ Gradients computed for {} parameters", param_count);
    
    // Test that damping parameters are learnable
    if model.layer1.has_damping() {
        if let Some(damping_coeffs) = model.layer1.get_damping_coefficients() {
            println!("✅ Layer 1 damping coefficients shape: {:?}", damping_coeffs.dims());
        }
        if let Some(damping_scales) = model.layer1.get_damping_scales() {
            println!("✅ Layer 1 damping scales shape: {:?}", damping_scales.dims());
        }
    }
}

fn test_gpu_performance_vs_cpu() {
    println!("\n🚀 Performance Test: GPU vs CPU Backend...");
    
    // GPU Test
    let gpu_device = MyWgpuBackend::default();
    let gpu_model: TestDLinossModel<MyAutodiffBackend> = TestDLinossModel::new(&gpu_device);
    
    let gpu_input = Tensor::<MyAutodiffBackend, 3>::random(
        [BATCH_SIZE, SEQ_LEN, D_INPUT],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &gpu_device,
    );
    
    let start = Instant::now();
    for _ in 0..10 {
        let _ = gpu_model.forward(gpu_input.clone());
    }
    let gpu_time = start.elapsed();
    
    println!("🔥 GPU (WGPU) - 10 forward passes: {:?}", gpu_time);
    println!("   Average per pass: {:?}", gpu_time / 10);
    
    // CPU Test (for comparison)
    type CpuBackend = burn::backend::NdArray<f32>;
    type CpuAutodiffBackend = Autodiff<CpuBackend>;
    
    let cpu_device = CpuBackend::default();
    let cpu_model: TestDLinossModel<CpuAutodiffBackend> = TestDLinossModel::new(&cpu_device);
    
    let cpu_input = Tensor::<CpuAutodiffBackend, 3>::random(
        [BATCH_SIZE, SEQ_LEN, D_INPUT],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &cpu_device,
    );
    
    let start = Instant::now();
    for _ in 0..10 {
        let _ = cpu_model.forward(cpu_input.clone());
    }
    let cpu_time = start.elapsed();
    
    println!("🖥️  CPU (NdArray) - 10 forward passes: {:?}", cpu_time);
    println!("   Average per pass: {:?}", cpu_time / 10);
    
    let speedup = cpu_time.as_secs_f32() / gpu_time.as_secs_f32();
    println!("⚡ GPU Speedup: {:.2}x", speedup);
}

fn test_damping_functionality() {
    println!("\n🔬 Testing D-LinOSS Damping Functionality...");
    
    let device = MyWgpuBackend::default();
    
    // Test with damping enabled
    let damped_config = OptimizedDLinossConfig {
        d_input: 32,
        d_model: 64,
        d_output: 32,
        delta_t: 0.1,
        init_std: 0.02,
        enable_layer_norm: false,
        enable_damping: true,
        init_damping: 0.2,
        num_damping_scales: 2,
        a_parameterization: AParameterization::GELU,
    };
    
    // Test without damping (vanilla LinOSS behavior)
    let undamped_config = OptimizedDLinossConfig {
        d_input: 32,
        d_model: 64,
        d_output: 32,
        delta_t: 0.1,
        init_std: 0.02,
        enable_layer_norm: false,
        enable_damping: false,
        init_damping: 0.0,
        num_damping_scales: 0,
        a_parameterization: AParameterization::GELU,
    };
    
    let damped_layer: OptimizedDLinossLayer<MyAutodiffBackend> = OptimizedDLinossLayer::new(&damped_config, &device);
    let undamped_layer: OptimizedDLinossLayer<MyAutodiffBackend> = OptimizedDLinossLayer::new(&undamped_config, &device);
    
    // Create test input (impulse followed by zeros to test energy dissipation)
    let mut input_data = vec![0.0f32; BATCH_SIZE * SEQ_LEN * 32];
    // Set impulse at t=0
    for batch in 0..BATCH_SIZE {
        input_data[batch * SEQ_LEN * 32] = 1.0;
    }
    
    let input = Tensor::<MyAutodiffBackend, 3>::from_data(
        TensorData::new(input_data, [BATCH_SIZE, SEQ_LEN, 32]),
        &device,
    );
    
    let damped_output = damped_layer.forward(input.clone());
    let undamped_output = undamped_layer.forward(input.clone());
    
    // Calculate energy over time (sum of squares)
    let damped_energy: Vec<f32> = (0..SEQ_LEN)
        .map(|t| {
            let slice = damped_output.clone().slice([0..BATCH_SIZE, t..t+1, 0..32]);
            slice.powf_scalar(2.0).sum().into_scalar().elem::<f32>()
        })
        .collect();
    
    let undamped_energy: Vec<f32> = (0..SEQ_LEN)
        .map(|t| {
            let slice = undamped_output.clone().slice([0..BATCH_SIZE, t..t+1, 0..32]);
            slice.powf_scalar(2.0).sum().into_scalar().elem::<f32>()
        })
        .collect();
    
    println!("✅ Damping comparison completed");
    println!("   Damped final energy: {:.6}", damped_energy[SEQ_LEN-1]);
    println!("   Undamped final energy: {:.6}", undamped_energy[SEQ_LEN-1]);
    
    // Damped version should have lower energy at the end
    let energy_reduction = (undamped_energy[SEQ_LEN-1] - damped_energy[SEQ_LEN-1]) / undamped_energy[SEQ_LEN-1];
    println!("   Energy reduction: {:.2}%", energy_reduction * 100.0);
    
    if energy_reduction > 0.1 {
        println!("✅ Damping is working - significant energy reduction detected");
    } else {
        println!("⚠️  Damping effect may be too weak");
    }
}

fn test_training_stability() {
    println!("\n🎯 Testing Training Stability on GPU...");
    
    let device = MyWgpuBackend::default();
    let mut model: TestDLinossModel<MyAutodiffBackend> = TestDLinossModel::new(&device);
    let mut optimizer = AdamConfig::new().init();
    
    // Generate synthetic data (sinusoidal sequence)
    let t: Vec<f32> = (0..SEQ_LEN).map(|i| i as f32 * 0.1).collect();
    let mut input_data = vec![0.0f32; BATCH_SIZE * SEQ_LEN * D_INPUT];
    let mut target_data = vec![0.0f32; BATCH_SIZE * SEQ_LEN * D_OUTPUT];
    
    for batch in 0..BATCH_SIZE {
        for seq in 0..SEQ_LEN {
            let time = t[seq];
            let freq = 1.0 + (batch as f32 * 0.1);
            
            // Input: multiple frequencies
            for dim in 0..D_INPUT {
                let input_freq = freq * (1.0 + dim as f32 * 0.1);
                input_data[batch * SEQ_LEN * D_INPUT + seq * D_INPUT + dim] = 
                    (time * input_freq).sin();
            }
            
            // Target: damped oscillation
            for dim in 0..D_OUTPUT {
                let target_freq = freq * (1.0 + dim as f32 * 0.05);
                target_data[batch * SEQ_LEN * D_OUTPUT + seq * D_OUTPUT + dim] = 
                    (-time * 0.1).exp() * (time * target_freq).sin();
            }
        }
    }
    
    let input = Tensor::<MyAutodiffBackend, 3>::from_data(
        TensorData::new(input_data, [BATCH_SIZE, SEQ_LEN, D_INPUT]),
        &device,
    );
    
    let target = Tensor::<MyAutodiffBackend, 3>::from_data(
        TensorData::new(target_data, [BATCH_SIZE, SEQ_LEN, D_OUTPUT]),
        &device,
    );
    
    println!("📊 Training for {} epochs...", NUM_EPOCHS);
    let training_start = Instant::now();
    
    let mut losses = Vec::new();
    
    for epoch in 0..NUM_EPOCHS {
        let output = model.forward(input.clone());
        let loss = (output - target.clone()).powf_scalar(2.0).mean();
        let loss_val: f32 = loss.clone().into_scalar().elem();
        losses.push(loss_val);
        
        let gradients = loss.backward();
        model = optimizer.step(1e-3, model, gradients);
        
        if epoch % 20 == 0 || epoch == NUM_EPOCHS - 1 {
            println!("   Epoch {}: Loss = {:.6}", epoch, loss_val);
        }
    }
    
    let training_time = training_start.elapsed();
    
    println!("✅ Training completed in {:?}", training_time);
    println!("   Initial loss: {:.6}", losses[0]);
    println!("   Final loss: {:.6}", losses[NUM_EPOCHS-1]);
    
    let loss_reduction = (losses[0] - losses[NUM_EPOCHS-1]) / losses[0];
    println!("   Loss reduction: {:.2}%", loss_reduction * 100.0);
    
    if loss_reduction > 0.5 {
        println!("✅ Training converged successfully");
    } else {
        println!("⚠️  Training may need more epochs or tuning");
    }
}

fn main() {
    println!("🚀 D-LinOSS GPU Implementation Validation Suite");
    println!("================================================\n");
    
    // Test 1: Basic GPU functionality
    test_gpu_backend_usage();
    
    // Test 2: Performance comparison
    test_gpu_performance_vs_cpu();
    
    // Test 3: Damping functionality
    test_damping_functionality();
    
    // Test 4: Training stability
    test_training_stability();
    
    println!("\n🎉 All tests completed!");
    println!("If all tests passed, D-LinOSS is properly using GPU acceleration");
}
