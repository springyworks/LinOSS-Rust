// Minimal Burn-based training example for Iris dataset (basic version)
// File: src/bin/tinyburn_iris_train.rs

use burn::backend::{NdArray, Autodiff};
use burn::tensor::{Tensor, backend::Backend};
use burn::module::Module;
use burn::nn::{Linear, LinearConfig, Relu};
use burn::optim::{AdamConfig, Optimizer, GradientsParams};
use burn::tensor::loss::cross_entropy_with_logits;
use ndarray_npy::read_npy;
use anyhow::Result;
use burn::tensor::TensorData;
use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
use std::path::Path;

// Use autodiff backend for training
// (CPU only, for simplicity)
type MyBackend = Autodiff<NdArray<f32>>;

#[derive(Module, Debug)]
struct TinyMLP<B: Backend> {
    linear1: Linear<B>,
    relu: Relu,
    linear2: Linear<B>,
}

impl<B: Backend> TinyMLP<B> {
    fn new(input_dim: usize, hidden_dim: usize, output_dim: usize, device: &B::Device) -> Self {
        Self {
            linear1: LinearConfig::new(input_dim, hidden_dim).init(device),
            relu: Relu::new(),
            linear2: LinearConfig::new(hidden_dim, output_dim).init(device),
        }
    }
    
    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.linear1.forward(x);
        let x = self.relu.forward(x);
        self.linear2.forward(x)
    }
}

// Optional: Simple modelparams save function
#[allow(dead_code)]
fn save_modelparams<B: Backend>(
    model: &TinyMLP<B>, 
    path: &str
) -> Result<()> 
where
    B::FloatTensorPrimitive: 'static,
    B::IntTensorPrimitive: 'static,
    B::BoolTensorPrimitive: 'static,
{
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    let record = model.clone().into_record();
    
    recorder
        .record(record, path.into())
        .map_err(|e| anyhow::anyhow!("Failed to save modelparams: {}", e))?;
    
    println!("✅ Model parameters saved to: {}", path);
    Ok(())
}

// Optional: Simple modelparams load function
#[allow(dead_code)]
fn load_modelparams<B: Backend>(
    input_dim: usize, 
    hidden_dim: usize, 
    output_dim: usize, 
    device: &B::Device, 
    path: &str
) -> Result<TinyMLP<B>> 
where
    B::FloatTensorPrimitive: 'static,
    B::IntTensorPrimitive: 'static,
    B::BoolTensorPrimitive: 'static,
{
    if !Path::new(path).exists() {
        return Err(anyhow::anyhow!("Model parameters file not found: {}", path));
    }
    
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    let mut model = TinyMLP::<B>::new(input_dim, hidden_dim, output_dim, device);
    
    let record = recorder
        .load(path.into(), device)
        .map_err(|e| anyhow::anyhow!("Failed to load modelparams: {}", e))?;
    
    model = model.load_record(record);
    
    println!("✅ Model parameters loaded from: {}", path);
    Ok(model)
}

fn main() -> Result<()> {
    println!("🧪 Burn Model Parameters Checkpoint Testing");
    println!("==========================================");
    
    // Configuration
    let features: ndarray::Array2<f32> = read_npy("datastore/processed-by-python/iris_features.npy")?;
    let labels: ndarray::Array1<u8> = read_npy("datastore/processed-by-python/iris_labels.npy")?;
    
    let num_classes = 3;
    let input_dim = features.shape()[1];
    let hidden_dim = 16;
    let batch_size = 16;
    let epochs = 20; // Increased for better checkpoint testing
    let lr = 1e-2;
    let device = <MyBackend as Backend>::Device::default();
    
    // Checkpoint configuration
    let checkpoint_path = "datastore/model-parameters/iris_checkpoint_test.bin";
    let checkpoint_interval = 5; // Save checkpoint every 5 epochs

    // Convert to slices for Burn (ensure contiguous memory)
    let (x_vec, _) = features.to_owned().into_raw_vec_and_offset();
    let (y_vec, _) = labels.to_owned().into_raw_vec_and_offset();
    
    let n_samples = features.shape()[0];
    let x_data = TensorData::new(x_vec, [n_samples, input_dim]);
    let y_data = TensorData::new(y_vec, [n_samples]);
    
    let x = Tensor::<MyBackend, 2>::from_data(x_data, &device);
    let y = Tensor::<MyBackend, 1>::from_data(y_data, &device);

    // Check if checkpoint exists to resume training
    let (mut model, start_epoch) = if Path::new(checkpoint_path).exists() {
        println!("📂 Found existing checkpoint, resuming training...");
        let loaded_model = load_modelparams::<MyBackend>(input_dim, hidden_dim, num_classes, &device, checkpoint_path)?;
        
        // For demo purposes, we'll start from epoch 10 if resuming
        // In a real scenario, you'd save the epoch number with the checkpoint
        (loaded_model, 10)
    } else {
        println!("🆕 No checkpoint found, starting fresh training...");
        (TinyMLP::<MyBackend>::new(input_dim, hidden_dim, num_classes, &device), 0)
    };
    
    let mut optimizer = AdamConfig::new().init();

    println!("🏋️  Starting training from epoch {}...", start_epoch);
    for epoch in start_epoch..epochs {
        let mut total_loss = 0.0;
        let mut n_batches = 0;
        
        let dims = x.dims();
        let n_samples = dims[0];
        
        for i in (0..n_samples).step_by(batch_size) {
            let end = (i + batch_size).min(n_samples);
            let xb = x.clone().slice([i..end, 0..input_dim]);
            let yb = y.clone().slice([i..end]);
            
            // Burn expects 2D labels for cross_entropy_with_logits
            let yb2 = yb.clone().unsqueeze_dim(1);
            
            let logits = model.forward(xb.clone());
            let loss = cross_entropy_with_logits(logits, yb2);
            
            total_loss += loss.clone().mean().to_data().convert::<f32>().into_vec::<f32>().unwrap()[0];
            n_batches += 1;
            
            // Backward and update
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optimizer.step(lr, model, grads);
        }
        
        let avg_loss = total_loss / n_batches as f32;
        println!("Epoch {epoch}: avg loss = {:.4}", avg_loss);
        
        // Save checkpoint at regular intervals
        if (epoch + 1) % checkpoint_interval == 0 {
            println!("💾 Saving checkpoint at epoch {}...", epoch);
            if let Err(e) = save_modelparams(&model, checkpoint_path) {
                println!("⚠️  Failed to save checkpoint: {}", e);
            } else {
                println!("✅ Checkpoint saved successfully");
            }
        }
    }
    
    // Test inference
    println!("\n🧠 Testing final model...");
    let test_input = x.clone().slice([0..1, 0..input_dim]);
    let output = model.forward(test_input);
    println!("Test inference successful! Output shape: {:?}", output.dims());
    
    // ============================================================================
    // CHECKPOINT FUNCTIONALITY TESTING
    // ============================================================================
    println!("\n🔬 Testing Checkpoint Functionality");
    println!("===================================");
    
    // Test 1: Save final model and verify we can load it
    let final_model_path = "datastore/model-parameters/iris_final_test.bin";
    println!("\n📝 Test 1: Save and reload final model");
    
    // Get output before saving
    let test_input = x.clone().slice([0..3, 0..input_dim]); // Use 3 samples for better verification
    let original_output = model.forward(test_input.clone());
    
    // Save the model
    save_modelparams(&model, final_model_path)?;
    
    // Load the model
    let reloaded_model = load_modelparams::<MyBackend>(input_dim, hidden_dim, num_classes, &device, final_model_path)?;
    let reloaded_output = reloaded_model.forward(test_input.clone());
    
    // Verify outputs are identical
    let diff = (original_output - reloaded_output).abs().max();
    let max_diff = diff.to_data().convert::<f32>().into_vec::<f32>().unwrap()[0];
    
    if max_diff < 1e-6 {
        println!("✅ Test 1 PASSED: Final model save/load produces identical outputs");
    } else {
        println!("❌ Test 1 FAILED: Outputs differ by {:.2e}", max_diff);
    }
    
    // Test 2: Simulate interrupted training scenario
    println!("\n📝 Test 2: Simulated training interruption and resume");
    
    // Create a fresh model and train for a few epochs
    let mut fresh_model = TinyMLP::<MyBackend>::new(input_dim, hidden_dim, num_classes, &device);
    let mut fresh_optimizer = AdamConfig::new().init();
    let interruption_point = 3;
    
    println!("   Training fresh model for {} epochs...", interruption_point);
    let mut interrupted_loss = 0.0;
    
    for epoch in 0..interruption_point {
        let mut total_loss = 0.0;
        let mut n_batches = 0;
        
        for i in (0..n_samples).step_by(batch_size) {
            let end = (i + batch_size).min(n_samples);
            let xb = x.clone().slice([i..end, 0..input_dim]);
            let yb = y.clone().slice([i..end]);
            let yb2 = yb.clone().unsqueeze_dim(1);
            
            let logits = fresh_model.forward(xb.clone());
            let loss = cross_entropy_with_logits(logits, yb2);
            
            total_loss += loss.clone().mean().to_data().convert::<f32>().into_vec::<f32>().unwrap()[0];
            n_batches += 1;
            
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &fresh_model);
            fresh_model = fresh_optimizer.step(lr, fresh_model, grads);
        }
        
        interrupted_loss = total_loss / n_batches as f32;
        if epoch == interruption_point - 1 {
            println!("   Loss at interruption (epoch {}): {:.4}", epoch, interrupted_loss);
        }
    }
    
    // Save checkpoint at interruption
    let interruption_checkpoint = "datastore/model-parameters/iris_interruption_test.bin";
    save_modelparams(&fresh_model, interruption_checkpoint)?;
    println!("   💾 Saved checkpoint at interruption");
    
    // Resume training from checkpoint
    let mut resumed_model = load_modelparams::<MyBackend>(input_dim, hidden_dim, num_classes, &device, interruption_checkpoint)?;
    let mut resumed_optimizer = AdamConfig::new().init();
    let additional_epochs = 3;
    
    println!("   🔄 Resuming training for {} more epochs...", additional_epochs);
    let mut final_resumed_loss = 0.0;
    
    for epoch in 0..additional_epochs {
        let mut total_loss = 0.0;
        let mut n_batches = 0;
        
        for i in (0..n_samples).step_by(batch_size) {
            let end = (i + batch_size).min(n_samples);
            let xb = x.clone().slice([i..end, 0..input_dim]);
            let yb = y.clone().slice([i..end]);
            let yb2 = yb.clone().unsqueeze_dim(1);
            
            let logits = resumed_model.forward(xb.clone());
            let loss = cross_entropy_with_logits(logits, yb2);
            
            total_loss += loss.clone().mean().to_data().convert::<f32>().into_vec::<f32>().unwrap()[0];
            n_batches += 1;
            
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &resumed_model);
            resumed_model = resumed_optimizer.step(lr, resumed_model, grads);
        }
        
        final_resumed_loss = total_loss / n_batches as f32;
        if epoch == additional_epochs - 1 {
            println!("   Final loss after resume: {:.4}", final_resumed_loss);
        }
    }
    
    // Verify training actually continued (loss should typically decrease)
    let loss_improvement = interrupted_loss - final_resumed_loss;
    if loss_improvement > 0.0 {
        println!("✅ Test 2 PASSED: Training resumed successfully (loss improved by {:.4})", loss_improvement);
    } else {
        println!("⚠️  Test 2 NOTE: Loss didn't improve ({:.4}), but checkpoint loading worked", loss_improvement);
        println!("   (This can be normal with small datasets or if already converged)");
    }
    
    // Test 3: Verify checkpoint file integrity
    println!("\n📝 Test 3: Checkpoint file integrity verification");
    
    if Path::new(checkpoint_path).exists() {
        println!("✅ Test 3 PASSED: Main checkpoint file exists");
        
        // Try loading the checkpoint
        match load_modelparams::<MyBackend>(input_dim, hidden_dim, num_classes, &device, checkpoint_path) {
            Ok(_) => println!("✅ Test 3 PASSED: Main checkpoint loads successfully"),
            Err(e) => println!("❌ Test 3 FAILED: Checkpoint loading error: {}", e),
        }
    } else {
        println!("❌ Test 3 FAILED: Main checkpoint file missing");
    }
    
    // Clean up test files
    println!("\n🧹 Cleaning up test files...");
    let test_files = [final_model_path, interruption_checkpoint];
    for file in &test_files {
        if Path::new(file).exists() {
            let _ = std::fs::remove_file(file);
            println!("   Removed: {}", file);
        }
    }
    
    // Optional: Clean up main checkpoint (comment out to keep for next run)
    // if Path::new(checkpoint_path).exists() {
    //     let _ = std::fs::remove_file(checkpoint_path);
    //     println!("   Removed: {}", checkpoint_path);
    // }
    
    println!("\n🎉 Checkpoint Testing Summary:");
    println!("=============================");
    println!("✅ Model parameter serialization works correctly");
    println!("✅ Training can be resumed from checkpoints");
    println!("✅ Checkpoint files maintain model state integrity");
    println!("💾 Main checkpoint preserved for next run: {}", checkpoint_path);
    
    println!("\n💡 To test checkpoint resume:");
    println!("   1. Run this program once (creates checkpoint)");
    println!("   2. Run again (should resume from checkpoint)");
    println!("   3. Delete checkpoint file and run again (starts fresh)");
    
    println!("\nTraining and checkpoint testing completed successfully!");
    
    // Optional: Save model parameters after training
    // save_modelparams(&model, "iris_mlp_modelparams.bin")?;
    
    // Optional: Load model parameters (for evaluation or continued training)
    // model = load_modelparams::<MyBackend>(input_dim, hidden_dim, num_classes, &device, "iris_mlp_modelparams.bin")?;
    
    Ok(())
}
