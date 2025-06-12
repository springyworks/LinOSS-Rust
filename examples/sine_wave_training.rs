#![recursion_limit = "256"]

// Simple sine wave training example for LinOSS
// This demonstrates basic training loop to learn sine wave prediction

use std::f32::consts::PI;

// Burn imports
use burn::{
    module::AutodiffModule,
    tensor::{TensorData, Tensor},
    optim::{AdamConfig, Optimizer, GradientsParams},
};

// LinOSS imports
use linoss_rust::linoss::{
    model::{FullLinossModel, FullLinossModelConfig}, 
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
type MyBackend = burn::backend::wgpu::Wgpu;
#[cfg(all(feature = "wgpu_backend", feature = "ndarray_backend"))]
type MyAutodiffBackend = burn::backend::Autodiff<burn::backend::wgpu::Wgpu>;

#[cfg(not(any(feature = "ndarray_backend", feature = "wgpu_backend")))]
type MyBackend = burn::backend::wgpu::Wgpu;
#[cfg(not(any(feature = "ndarray_backend", feature = "wgpu_backend")))]
type MyAutodiffBackend = burn::backend::Autodiff<burn::backend::wgpu::Wgpu>;

// Constants
const D_INPUT: usize = 1;
const D_MODEL: usize = 32;  // Smaller model for faster training
const D_OUTPUT: usize = 1;
const N_LAYERS: usize = 2;
const BATCH_SIZE: usize = 4;
const SEQ_LEN: usize = 20;   // Shorter sequences for training
const NUM_EPOCHS: usize = 50;
const LEARNING_RATE: f64 = 0.001;

// Data generation functions
fn generate_sine_wave_sequence(length: usize, time_step: f32, phase: f32, frequency: f32) -> Vec<f32> {
    (0..length)
        .map(|i| (frequency * 2.0 * PI * (i as f32 * time_step + phase)).sin())
        .collect()
}

fn generate_training_batch(
    batch_size: usize, 
    seq_len: usize, 
    device: &<MyAutodiffBackend as burn::tensor::backend::Backend>::Device
) -> (Tensor<MyAutodiffBackend, 3>, Tensor<MyAutodiffBackend, 3>) {
    let mut inputs = Vec::new();
    let mut targets = Vec::new();
    
    for _ in 0..batch_size {
        // Random phase and frequency for variety
        let phase = (rand::random::<f32>() - 0.5) * 2.0 * PI;
        let frequency = 0.5 + rand::random::<f32>() * 1.5; // 0.5 to 2.0 Hz
        
        let full_sequence = generate_sine_wave_sequence(seq_len + 1, 0.1, phase, frequency);
        let input_seq: Vec<f32> = full_sequence[0..seq_len].to_vec();
        let target_seq: Vec<f32> = full_sequence[1..seq_len + 1].to_vec();
        
        inputs.extend(input_seq);
        targets.extend(target_seq);
    }
    
    let input_tensor = Tensor::<MyAutodiffBackend, 3>::from_data(
        TensorData::new(inputs, [batch_size, seq_len, D_INPUT]), device
    );
    
    let target_tensor = Tensor::<MyAutodiffBackend, 3>::from_data(
        TensorData::new(targets, [batch_size, seq_len, D_OUTPUT]), device
    );
    
    (input_tensor, target_tensor)
}

fn test_model_predictions(model: &FullLinossModel<MyBackend>, device: &<MyBackend as burn::tensor::backend::Backend>::Device) {
    println!("\n--- Testing Model Predictions ---");
    
    // Generate a test sine wave
    let test_sine_data = generate_sine_wave_sequence(SEQ_LEN + 5, 0.1, 0.0, 1.0);
    let test_input: Vec<f32> = test_sine_data[0..SEQ_LEN].to_vec();
    let test_targets: Vec<f32> = test_sine_data[1..SEQ_LEN + 1].to_vec();

    let test_input_tensor = Tensor::<MyBackend, 3>::from_data(
        TensorData::new(test_input.clone(), [1, SEQ_LEN, D_INPUT]), device
    );

    let test_prediction = model.forward(test_input_tensor);
    let pred_data = test_prediction.into_data().into_vec().unwrap();

    println!("Step | Input    | Target   | Prediction | Error");
    println!("-----|----------|----------|------------|----------");
    
    let mut total_error = 0.0f32;
    for i in 0..8.min(test_input.len()) {
        let input_val: f32 = test_input[i];
        let target_val: f32 = test_targets[i];
        let pred_val: f32 = pred_data[i];
        let error: f32 = (target_val - pred_val).abs();
        total_error += error;
        
        println!("{:4} | {:8.4} | {:8.4} | {:10.4} | {:8.4}", 
                 i, input_val, target_val, pred_val, error);
    }

    let mse: f32 = test_targets.iter().zip(pred_data.iter())
        .map(|(t, p)| {
            let diff = *t as f32 - *p as f32;
            diff.powi(2)
        })
        .sum::<f32>() / test_targets.len() as f32;
    
    println!("Average Error: {:.6}", total_error / 8.0);
    println!("Test MSE: {:.6}", mse);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    println!("LinOSS Sine Wave Training Example");

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

    // Initialize model for training
    let mut model = model_config.init::<MyAutodiffBackend>(&device);
    
    // Create optimizer
    let optimizer_config = AdamConfig::new();
    let mut optimizer = optimizer_config.init();
    
    println!("Model initialized. Starting training...");
    println!("Epochs: {}, Batch Size: {}, Sequence Length: {}", NUM_EPOCHS, BATCH_SIZE, SEQ_LEN);

    // Test before training
    println!("\n=== Before Training ===");
    let initial_test_model = model.clone().valid();
    test_model_predictions(&initial_test_model, &device);

    // Training loop
    for epoch in 0..NUM_EPOCHS {
        let mut total_loss = 0.0f32;
        let batches_per_epoch = 5; // Number of batches per epoch
        
        for _batch_idx in 0..batches_per_epoch {
            // Generate training batch
            let (input_batch, target_batch) = generate_training_batch(BATCH_SIZE, SEQ_LEN, &device);
            
            // Forward pass
            let prediction = model.forward(input_batch);
            let loss = (prediction - target_batch).powf_scalar(2.0).mean();
            let loss_value = loss.clone().into_scalar();
            total_loss += loss_value;
            
            // Backward pass
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optimizer.step(LEARNING_RATE, model, grads);
        }
        
        let avg_loss = total_loss / batches_per_epoch as f32;
        
        if epoch % 10 == 0 || epoch == NUM_EPOCHS - 1 {
            println!("Epoch {:3}/{} | Avg Loss: {:.6}", epoch + 1, NUM_EPOCHS, avg_loss);
        }
        
        // Test every 20 epochs
        if epoch % 20 == 19 || epoch == NUM_EPOCHS - 1 {
            let test_model = model.clone().valid();
            test_model_predictions(&test_model, &device);
        }
    }

    println!("\n=== Training Completed! ===");
    println!("The model should now predict sine waves much better than before.");
    
    Ok(())
}
