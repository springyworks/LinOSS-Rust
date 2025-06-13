#![recursion_limit = "256"]

// Simple sine wave prediction example for LinOSS
// This demonstrates basic LinOSS model initialization and inference

use std::f32::consts::PI;

// Burn imports
use burn::{
    module::AutodiffModule,
    tensor::{TensorData, Tensor},
};

// LinOSS imports
use linoss_rust::linoss::{
    model::FullLinossModelConfig, 
    block::LinossBlockConfig
};

// Type aliases for convenience based on selected backend
#[cfg(all(feature = "ndarray_backend", not(feature = "wgpu_backend")))]
type MyBackend = burn::backend::ndarray::NdArray<f32>;
#[cfg(all(feature = "ndarray_backend", not(feature = "wgpu_backend")))]
type MyAutodiffBackend = burn::backend::Autodiff<burn::backend::ndarray::NdArray<f32>>;

#[cfg(all(feature = "wgpu_backend", not(feature = "ndarray_backend")))]
type MyBackend = burn::backend::wgpu::Wgpu;
#[cfg(all(feature = "wgpu_backend", not(feature = "ndarray_backend")))]
type MyAutodiffBackend = burn::backend::Autodiff<burn::backend::wgpu::Wgpu>;

#[cfg(all(feature = "wgpu_backend", feature = "ndarray_backend"))]
type MyBackend = burn::backend::wgpu::Wgpu; // Default to WGPU if both are enabled
#[cfg(all(feature = "wgpu_backend", feature = "ndarray_backend"))]
type MyAutodiffBackend = burn::backend::Autodiff<burn::backend::wgpu::Wgpu>;

#[cfg(not(any(feature = "ndarray_backend", feature = "wgpu_backend")))]
type MyBackend = burn::backend::wgpu::Wgpu; // Default to WGPU if no backend feature is explicitly selected
#[cfg(not(any(feature = "ndarray_backend", feature = "wgpu_backend")))]
type MyAutodiffBackend = burn::backend::Autodiff<burn::backend::wgpu::Wgpu>;

// Constants
const D_INPUT: usize = 1;
const D_MODEL: usize = 64;
const D_OUTPUT: usize = 1;
const N_LAYERS: usize = 2;
const BATCH_SIZE: usize = 1;
const SEQ_LEN: usize = 50;

// Simple sine wave data generation
fn generate_sine_wave_sequence(length: usize, time_step: f32, phase: f32) -> Vec<f32> {
    (0..length)
        .map(|i| (2.0 * PI * (i as f32 * time_step + phase)).sin())
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    println!("LinOSS Sine Wave Prediction Test");

    // Initialize device
    let device = Default::default();

    // Create model configuration
    let block_config = LinossBlockConfig {
        d_state_m: D_MODEL / 2,
        d_ff: D_MODEL * 2,
        delta_t: 0.1,
        init_std: 0.02,
        enable_d_feedthrough: true,
    };

    let model_config = FullLinossModelConfig {
        d_input: D_INPUT,
        d_model: D_MODEL,
        d_output: D_OUTPUT,
        n_layers: N_LAYERS,
        linoss_block_config: block_config,
    };

    // Initialize model for training (autodiff backend)
    let model = model_config.init::<MyAutodiffBackend>(&device);
    
    println!("Model initialized successfully!");
    println!("Starting basic sine wave prediction test...");

    // Generate test data
    let test_sine_data = generate_sine_wave_sequence(SEQ_LEN + 5, 0.1, 0.0);
    let test_input: Vec<f32> = test_sine_data[0..SEQ_LEN].to_vec();
    let test_targets: Vec<f32> = test_sine_data[1..SEQ_LEN + 1].to_vec();

    // Convert to inference backend for testing
    let inference_model = model.valid();

    let test_input_tensor = Tensor::<MyBackend, 3>::from_data(
        TensorData::new(test_input.clone(), [BATCH_SIZE, SEQ_LEN, D_INPUT]), &device
    );

    let test_prediction = inference_model.forward(test_input_tensor);

    // Extract prediction values
    let pred_data = test_prediction.into_data().into_vec().unwrap();

    println!("\nPrediction Results (first 10 steps):");
    println!("Step | Input    | Target   | Prediction | Error");
    println!("-----|----------|----------|------------|----------");
    
    for i in 0..10.min(test_input.len()) {
        let input_val: f32 = test_input[i];
        let target_val: f32 = test_targets[i];
        let pred_val: f32 = pred_data[i];
        let error: f32 = (target_val - pred_val).abs();
        
        println!("{:4} | {:8.4} | {:8.4} | {:10.4} | {:8.4}", 
                 i, input_val, target_val, pred_val, error);
    }

    let mse: f32 = test_targets.iter().zip(pred_data.iter())
        .map(|(t, p)| {
            let diff = *t - *p;
            diff.powi(2)
        })
        .sum::<f32>() / test_targets.len() as f32;
    
    println!("\nTest MSE (untrained model): {:.6}", mse);
    println!("LinOSS model inference test completed successfully!");
    
    Ok(())
}
