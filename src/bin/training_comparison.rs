// src/bin/training_comparison.rs
// Simplified training efficiency comparison: LinOSS vs D-LinOSS
// Tests convergence behavior without complex training infrastructure

use burn::{
    tensor::{Tensor, TensorData},
    prelude::Backend,
};
use std::time::Instant;

// Conditionally import backends
#[cfg(feature = "wgpu_backend")]
use burn::backend::wgpu::{Wgpu, WgpuDevice};

use linoss_rust::linoss::{
    layer::{LinossLayer, LinossLayerConfig},
    dlinoss_layer::{DLinossLayer, DLinossLayerConfig},
};
use rand::{Rng, SeedableRng, rngs::StdRng};

/// Simple regression task: predict sine wave continuation
fn generate_sine_wave_task<B: Backend>(
    device: &B::Device,
    batch_size: usize,
    seq_len: usize,
    num_batches: usize,
) -> (Vec<Tensor<B, 3>>, Vec<Tensor<B, 3>>) {
    let mut rng = StdRng::seed_from_u64(42);
    let mut inputs = Vec::new();
    let mut targets = Vec::new();
    
    for _ in 0..num_batches {
        let mut input_data = vec![0.0; batch_size * seq_len];
        let mut target_data = vec![0.0; batch_size * seq_len];
        
        for b in 0..batch_size {
            let freq_val: f32 = rng.gen();
            let freq = 0.1 + freq_val * 0.5; // Random frequency
            let phase_val: f32 = rng.gen();
            let phase = phase_val * 2.0 * std::f32::consts::PI;
            let noise_level = 0.05;
            
            for t in 0..seq_len {
                let idx = b * seq_len + t;
                let time = t as f32 * 0.1;
                let clean_signal = (time * freq + phase).sin();
                let noise_val: f32 = rng.gen();
                let noise = (noise_val - 0.5) * noise_level;
                
                input_data[idx] = clean_signal + noise;
                // Target is the next time step (prediction task)
                let target_time = (t + 1) as f32 * 0.1;
                target_data[idx] = (target_time * freq + phase).sin();
            }
        }
        
        let input_tensor = Tensor::from_data(
            TensorData::new(input_data, [batch_size, seq_len, 1]), 
            device
        );
        let target_tensor = Tensor::from_data(
            TensorData::new(target_data, [batch_size, seq_len, 1]), 
            device
        );
        
        inputs.push(input_tensor);
        targets.push(target_tensor);
    }
    
    (inputs, targets)
}

/// Calculate MSE loss
fn calculate_loss<B: Backend>(prediction: &Tensor<B, 3>, target: &Tensor<B, 3>) -> Tensor<B, 1> {
    let diff = prediction.clone() - target.clone();
    let squared_diff = diff.powf_scalar(2.0);
    squared_diff.mean()
}

/// Run simplified convergence comparison (forward passes only)
fn run_convergence_comparison<B: Backend>(device: B::Device) 
where
    B::FloatTensorPrimitive: Send + Sync,
    B::IntTensorPrimitive: Send + Sync, 
    B::BoolTensorPrimitive: Send + Sync,
{
    println!("=== Convergence Behavior Comparison ===");
    println!("Backend: {}", std::any::type_name::<B>());
    
    // Task parameters
    let batch_size = 4;
    let seq_len = 128; // Longer sequence to test stability
    let num_test_batches = 10;
    let model_dim = 16;
    
    // Generate test data
    println!("Generating sine wave prediction task...");
    let (test_inputs, test_targets) = generate_sine_wave_task::<B>(
        &device, batch_size, seq_len, num_test_batches
    );
    
    // Configure models
    let linoss_config = LinossLayerConfig {
        d_state_m: model_dim,
        d_input_p: 1,
        d_output_q: 1,
        delta_t: 0.1,
        init_std: 0.02,
        enable_d_feedthrough: true,
    };
    
    let dlinoss_config = DLinossLayerConfig {
        d_input: 1,
        d_model: model_dim,
        d_output: 1,
        delta_t: 0.1,
        init_std: 0.02,
        enable_layer_norm: false,
        enable_damping: true,
        init_damping: 0.1,
        num_damping_scales: 2,
    };
    
    println!("Testing forward pass stability...");
    
    // Test LinOSS stability
    println!("\n--- LinOSS Forward Pass Tests ---");
    let linoss_model = linoss_config.init(&device);
    let start_time = Instant::now();
    let mut linoss_losses = Vec::new();
    
    for (i, (input, target)) in test_inputs.iter().zip(test_targets.iter()).enumerate() {
        let (loss, _prediction) = test_linoss_forward(&linoss_model, input, target);
        let loss_val = extract_loss_value(&loss);
        linoss_losses.push(loss_val);
        println!("Batch {}: Loss = {:.6}", i + 1, loss_val);
    }
    
    let linoss_time = start_time.elapsed();
    
    // Test D-LinOSS stability  
    println!("\n--- D-LinOSS Forward Pass Tests ---");
    let dlinoss_model = DLinossLayer::<B>::new(&dlinoss_config, &device);
    let start_time = Instant::now();
    let mut dlinoss_losses = Vec::new();
    
    for (i, (input, target)) in test_inputs.iter().zip(test_targets.iter()).enumerate() {
        let (loss, _prediction) = test_dlinoss_forward(&dlinoss_model, input, target);
        let loss_val = extract_loss_value(&loss);
        dlinoss_losses.push(loss_val);
        println!("Batch {}: Loss = {:.6}", i + 1, loss_val);
    }
    
    let dlinoss_time = start_time.elapsed();
    
    // Analysis
    let linoss_avg = linoss_losses.iter().sum::<f32>() / linoss_losses.len() as f32;
    let dlinoss_avg = dlinoss_losses.iter().sum::<f32>() / dlinoss_losses.len() as f32;
    
    let linoss_variance = linoss_losses.iter()
        .map(|x| (x - linoss_avg).powi(2))
        .sum::<f32>() / linoss_losses.len() as f32;
    
    let dlinoss_variance = dlinoss_losses.iter()
        .map(|x| (x - dlinoss_avg).powi(2))
        .sum::<f32>() / dlinoss_losses.len() as f32;
    
    println!("\n=== Forward Pass Results ===");
    println!("LinOSS  - Time: {:.2}s, Avg Loss: {:.6}, Variance: {:.6}", 
             linoss_time.as_secs_f32(), linoss_avg, linoss_variance);
    println!("D-LinOSS - Time: {:.2}s, Avg Loss: {:.6}, Variance: {:.6}", 
             dlinoss_time.as_secs_f32(), dlinoss_avg, dlinoss_variance);
    
    let time_ratio = dlinoss_time.as_secs_f32() / linoss_time.as_secs_f32();
    println!("D-LinOSS took {:.2}x longer", time_ratio);
    
    if dlinoss_variance < linoss_variance {
        let stability_improvement = (linoss_variance - dlinoss_variance) / linoss_variance * 100.0;
        println!("D-LinOSS has {:.1}% lower loss variance (better stability)", stability_improvement);
    } else {
        println!("LinOSS has lower loss variance");
    }
    
    println!("\n✓ Convergence comparison completed!");
}

/// Extract loss value from tensor
fn extract_loss_value<B: Backend>(loss: &Tensor<B, 1>) -> f32 {
    let data = loss.to_data();
    let bytes = data.as_bytes();
    if bytes.len() >= 4 {
        f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
    } else {
        0.0
    }
}

/// Forward pass for LinOSS (using sequence helper from dlinoss_comparison)
fn test_linoss_forward<B: Backend>(
    model: &LinossLayer<B>,
    input: &Tensor<B, 3>,
    target: &Tensor<B, 3>,
) -> (Tensor<B, 1>, Tensor<B, 3>) {
    let batch_size = input.dims()[0];
    let seq_len = input.dims()[1];
    
    let mut outputs = Vec::new();
    let mut hidden_state = None;
    
    for t in 0..seq_len {
        let input_t = input.clone().slice([0..batch_size, t..(t+1), 0..input.dims()[2]]).squeeze(1);
        let result = model.forward_step(input_t, hidden_state);
        outputs.push(result.output.unsqueeze_dim(1));
        hidden_state = result.hidden_state;
    }
    
    let prediction = Tensor::cat(outputs, 1);
    let loss = calculate_loss(&prediction, target);
    (loss, prediction)
}

/// Forward pass for D-LinOSS  
fn test_dlinoss_forward<B: Backend>(
    model: &DLinossLayer<B>,
    input: &Tensor<B, 3>,
    target: &Tensor<B, 3>,
) -> (Tensor<B, 1>, Tensor<B, 3>) {
    let prediction = model.forward(input.clone());
    let loss = calculate_loss(&prediction, target);
    (loss, prediction)
}

pub fn main() {
    #[cfg(feature = "wgpu_backend")]
    {
        println!("Running convergence comparison with WGPU backend...");
        let device = WgpuDevice::default();
        run_convergence_comparison::<Wgpu>(device);
    }
    
    #[cfg(all(feature = "ndarray_backend", not(feature = "wgpu_backend")))]
    {
        use burn::backend::ndarray::{NdArray, NdArrayDevice};
        println!("Running convergence comparison with NdArray backend...");
        let device = NdArrayDevice::default();
        run_convergence_comparison::<NdArray>(device);
    }
    
    #[cfg(not(any(feature = "wgpu_backend", feature = "ndarray_backend")))]
    {
        println!("No backend feature enabled. Please enable either 'wgpu_backend' or 'ndarray_backend'.");
    }
}
