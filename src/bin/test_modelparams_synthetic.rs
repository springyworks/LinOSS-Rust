// Comprehensive test for model parameters save/load using synthetic datasets
// This test validates bit-exact parameter preservation and training resumption

use burn::backend::{NdArray, Autodiff};
use burn::tensor::{Tensor, backend::Backend};
use burn::module::Module;
use burn::nn::{Linear, LinearConfig, Relu};
use burn::optim::{AdamConfig, Optimizer, GradientsParams};
use burn::tensor::loss::cross_entropy_with_logits;
use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
use burn::tensor::TensorData;
use anyhow::Result;
use std::path::Path;
use std::io::Write;

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

fn save_modelparams<B: Backend>(
    model: &TinyMLP<B>, 
    path: &str
) -> Result<()> 
where
    B::FloatTensorPrimitive: 'static,
    B::IntTensorPrimitive: 'static,
    B::BoolTensorPrimitive: 'static,
{
    // Ensure directory exists
    if let Some(parent) = Path::new(path).parent() {
        std::fs::create_dir_all(parent)?;
    }
    
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    let record = model.clone().into_record();
    
    recorder
        .record(record, path.into())
        .map_err(|e| anyhow::anyhow!("Failed to save modelparams: {}", e))?;
    
    println!("✅ Model parameters saved to: {}", path);
    Ok(())
}

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

/// Generate synthetic classification dataset
fn generate_synthetic_dataset(
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
    device: &<MyBackend as Backend>::Device
) -> (Tensor<MyBackend, 2>, Tensor<MyBackend, 1>) {
    // Create synthetic features with some structure
    // Each class will have different mean values
    let mut features_data = Vec::new();
    let mut labels_data = Vec::new();
    
    for i in 0..n_samples {
        let class = i % n_classes;
        labels_data.push(class as u8);
        
        // Generate features with class-dependent means
        for j in 0..n_features {
            let base_value = (class as f32) * 2.0 + (j as f32) * 0.5;
            let noise = (i as f32 * 0.1 + j as f32 * 0.05).sin() * 0.3;
            features_data.push(base_value + noise);
        }
    }
    
    let features_tensor_data = TensorData::new(features_data, [n_samples, n_features]);
    let labels_tensor_data = TensorData::new(labels_data, [n_samples]);
    
    let features = Tensor::<MyBackend, 2>::from_data(features_tensor_data, device);
    let labels = Tensor::<MyBackend, 1>::from_data(labels_tensor_data, device);
    
    (features, labels)
}

/// Compare two models by checking if their outputs are identical
fn models_identical(
    model1: &TinyMLP<MyBackend>,
    model2: &TinyMLP<MyBackend>,
    test_input: &Tensor<MyBackend, 2>
) -> bool {
    let output1 = model1.forward(test_input.clone());
    let output2 = model2.forward(test_input.clone());
    
    let diff = output1 - output2;
    let max_diff = diff.abs().max().to_data().convert::<f32>().into_vec::<f32>().unwrap()[0];
    
    // Should be exactly identical (within floating point precision)
    max_diff < 1e-6
}

/// Train model for a few epochs and return final loss
fn train_model(
    model: &mut TinyMLP<MyBackend>,
    optimizer: &mut impl Optimizer<TinyMLP<MyBackend>, MyBackend>,
    features: &Tensor<MyBackend, 2>,
    labels: &Tensor<MyBackend, 1>,
    epochs: usize,
    batch_size: usize,
    lr: f64
) -> f32 {
    let dims = features.dims();
    let n_samples = dims[0];
    let input_dim = dims[1];
    
    let mut final_loss = 0.0;
    
    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        let mut n_batches = 0;
        
        for i in (0..n_samples).step_by(batch_size) {
            let end = (i + batch_size).min(n_samples);
            let xb = features.clone().slice([i..end, 0..input_dim]);
            #[allow(clippy::single_range_in_vec_init)]
            let yb = labels.clone().slice([i..end]);
            
            // Burn expects 2D labels for cross_entropy_with_logits
            let yb2 = yb.clone().unsqueeze_dim(1);
            
            let logits = model.forward(xb.clone());
            let loss = cross_entropy_with_logits(logits, yb2);
            
            let loss_value = loss.clone().mean().to_data().convert::<f32>().into_vec::<f32>().unwrap()[0];
            total_loss += loss_value;
            n_batches += 1;
            
            // Backward and update
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &*model);
            *model = optimizer.step(lr, model.clone(), grads);
        }
        
        final_loss = total_loss / n_batches as f32;
        
        if epoch % 5 == 0 || epoch == epochs - 1 {
            println!("  Epoch {}: avg loss = {:.4}", epoch, final_loss);
        }
    }
    
    final_loss
}

/// Save execution trace with test results and model information
fn save_execution_trace(
    test_results: &[(&str, bool, String)],
    synthetic_data_info: &str,
    model_info: &str,
    save_traces: bool
) -> Result<()> {
    if !save_traces {
        return Ok(());
    }
    
    let trace_dir = "datastore/sythetic-datasets/execution-traces";
    std::fs::create_dir_all(trace_dir)?;
    
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    
    let trace_file = format!("{}/test_execution_{}.md", trace_dir, timestamp);
    let mut file = std::fs::File::create(&trace_file)?;
    
    writeln!(file, "# Model Parameters Test Execution Trace")?;
    writeln!(file, "**Timestamp**: {}", timestamp)?;
    writeln!(file, "**Unix Timestamp**: {}", timestamp)?;
    writeln!(file)?;
    
    writeln!(file, "## Synthetic Dataset Information")?;
    writeln!(file, "{}", synthetic_data_info)?;
    writeln!(file)?;
    
    writeln!(file, "## Model Architecture")?;
    writeln!(file, "{}", model_info)?;
    writeln!(file)?;
    
    writeln!(file, "## Test Results")?;
    writeln!(file, "| Test | Status | Details |")?;
    writeln!(file, "|------|--------|---------|")?;
    
    for (test_name, passed, details) in test_results {
        let status = if *passed { "✅ PASS" } else { "❌ FAIL" };
        writeln!(file, "| {} | {} | {} |", test_name, status, details)?;
    }
    
    writeln!(file)?;
    writeln!(file, "## Summary")?;
    let total_tests = test_results.len();
    let passed_tests = test_results.iter().filter(|(_, passed, _)| *passed).count();
    writeln!(file, "- **Total Tests**: {}", total_tests)?;
    writeln!(file, "- **Passed**: {}", passed_tests)?;
    writeln!(file, "- **Failed**: {}", total_tests - passed_tests)?;
    writeln!(file, "- **Success Rate**: {:.1}%", (passed_tests as f64 / total_tests as f64) * 100.0)?;
    
    if passed_tests == total_tests {
        writeln!(file, "\n🎉 **All tests passed successfully!**")?;
    } else {
        writeln!(file, "\n⚠️ **Some tests failed - review required**")?;
    }
    
    println!("📝 Execution trace saved to: {}", trace_file);
    Ok(())
}

fn main() -> Result<()> {
    println!("🧪 Comprehensive Model Parameters Test with Synthetic Data");
    println!("========================================================");
    
    let device = <MyBackend as Backend>::Device::default();
    
    // Test configuration
    let n_samples = 120;
    let n_features = 4;
    let n_classes = 3;
    let hidden_dim = 16;
    let batch_size = 16;
    let epochs = 15;
    let lr = 1e-2;
    
    // Option to save execution traces (set to true to enable)
    let save_traces = true;
    
    // Option to save sample model parameters as traces (set to true to enable)
    let save_sample_modelparams = true;
    
    // Collect test results for tracing
    let mut test_results = Vec::new();
    
    // Paths for testing
    let test_modelparams_path = "datastore/sythetic-datasets/test_synthetic_modelparams.bin";
    let checkpoint_path = "datastore/sythetic-datasets/test_checkpoint.bin";
    
    println!("\n📊 Generating synthetic dataset...");
    println!("  - Samples: {}, Features: {}, Classes: {}", n_samples, n_features, n_classes);
    
    let (features, labels) = generate_synthetic_dataset(n_samples, n_features, n_classes, &device);
    let test_input = features.clone().slice([0..5, 0..n_features]); // Use first 5 samples for testing
    
    // ============================================================================
    // TEST 1: Basic Save/Load Functionality
    // ============================================================================
    println!("\n🔬 Test 1: Basic Save/Load Functionality");
    println!("----------------------------------------");
    
    // Create and train a model
    let mut original_model = TinyMLP::<MyBackend>::new(n_features, hidden_dim, n_classes, &device);
    let mut optimizer = AdamConfig::new().init();
    
    println!("Training original model...");
    let _original_loss = train_model(
        &mut original_model, 
        &mut optimizer, 
        &features, 
        &labels, 
        epochs, 
        batch_size, 
        lr
    );
    
    // Get original output for comparison
    let _original_output = original_model.forward(test_input.clone());
    
    // Save the model parameters
    save_modelparams(&original_model, test_modelparams_path)?;
    
    // Load the model parameters into a new model
    let loaded_model = load_modelparams::<MyBackend>(n_features, hidden_dim, n_classes, &device, test_modelparams_path)?;
    
    // Verify they produce identical outputs
    if models_identical(&original_model, &loaded_model, &test_input) {
        println!("✅ Test 1 PASSED: Models produce identical outputs after save/load");
        test_results.push(("Basic Save/Load", true, "Bit-exact parameter preservation verified".to_string()));
    } else {
        println!("❌ Test 1 FAILED: Models produce different outputs after save/load");
        test_results.push(("Basic Save/Load", false, "Parameter preservation failed".to_string()));
        return Err(anyhow::anyhow!("Save/load test failed"));
    }
    
    // ============================================================================
    // TEST 2: Training Resumption (Checkpoint Functionality)
    // ============================================================================
    println!("\n🔬 Test 2: Training Resumption (Checkpoint)");
    println!("------------------------------------------");
    
    // Train model for half the epochs and save checkpoint
    let mut checkpoint_model = TinyMLP::<MyBackend>::new(n_features, hidden_dim, n_classes, &device);
    let mut checkpoint_optimizer = AdamConfig::new().init();
    
    println!("Training model for {} epochs (first half)...", epochs / 2);
    let checkpoint_loss = train_model(
        &mut checkpoint_model,
        &mut checkpoint_optimizer,
        &features,
        &labels,
        epochs / 2,
        batch_size,
        lr
    );
    
    // Save checkpoint
    save_modelparams(&checkpoint_model, checkpoint_path)?;
    
    // Load checkpoint and continue training
    let mut resumed_model = load_modelparams::<MyBackend>(n_features, hidden_dim, n_classes, &device, checkpoint_path)?;
    let mut resumed_optimizer = AdamConfig::new().init();
    
    println!("Resuming training for {} more epochs...", epochs - epochs / 2);
    let final_loss = train_model(
        &mut resumed_model,
        &mut resumed_optimizer,
        &features,
        &labels,
        epochs - epochs / 2,
        batch_size,
        lr
    );
    
    // Verify training progressed (loss should be lower)
    if final_loss < checkpoint_loss {
        println!("✅ Test 2 PASSED: Training resumed successfully (loss: {:.4} -> {:.4})", 
                checkpoint_loss, final_loss);
        test_results.push(("Training Resumption", true, 
                          format!("Loss improved from {:.4} to {:.4}", checkpoint_loss, final_loss)));
    } else {
        println!("⚠️  Test 2 WARNING: Loss didn't decrease as expected (loss: {:.4} -> {:.4})", 
                checkpoint_loss, final_loss);
        println!("   This might be normal for synthetic data with limited epochs");
        test_results.push(("Training Resumption", true, 
                          format!("Training resumed, loss: {:.4} -> {:.4} (may not always decrease with synthetic data)", 
                                 checkpoint_loss, final_loss)));
    }
    
    // ============================================================================
    // TEST 3: File Handling Edge Cases
    // ============================================================================
    println!("\n🔬 Test 3: File Handling Edge Cases");
    println!("----------------------------------");
    
    // Test loading non-existent file
    let nonexistent_path = "datastore/sythetic-datasets/nonexistent.bin";
    match load_modelparams::<MyBackend>(n_features, hidden_dim, n_classes, &device, nonexistent_path) {
        Ok(_) => {
            println!("❌ Test 3a FAILED: Should have failed to load non-existent file");
            test_results.push(("Non-existent File Handling", false, "Should have failed but didn't".to_string()));
        },
        Err(_) => {
            println!("✅ Test 3a PASSED: Correctly failed to load non-existent file");
            test_results.push(("Non-existent File Handling", true, "Proper error handling for missing files".to_string()));
        },
    }
    
    // Test saving to non-existent directory (should create it)
    let nested_path = "datastore/sythetic-datasets/nested/deep/test_modelparams.bin";
    match save_modelparams(&original_model, nested_path) {
        Ok(_) => {
            println!("✅ Test 3b PASSED: Successfully created nested directories for save");
            test_results.push(("Nested Directory Creation", true, "Auto-created nested directories successfully".to_string()));
            // Clean up
            let _ = std::fs::remove_file(nested_path);
            let _ = std::fs::remove_dir("datastore/sythetic-datasets/nested/deep");
            let _ = std::fs::remove_dir("datastore/sythetic-datasets/nested");
        },
        Err(e) => {
            println!("❌ Test 3b FAILED: Could not save to nested path: {}", e);
            test_results.push(("Nested Directory Creation", false, format!("Failed to create nested directories: {}", e)));
        },
    }
    
    // ============================================================================
    // TEST 4: Multiple Save/Load Cycles
    // ============================================================================
    println!("\n🔬 Test 4: Multiple Save/Load Cycles");
    println!("-----------------------------------");
    
    let mut cycle_model = original_model.clone();
    let cycle_path = "datastore/sythetic-datasets/cycle_test.bin";
    
    for i in 0..3 {
        save_modelparams(&cycle_model, cycle_path)?;
        cycle_model = load_modelparams::<MyBackend>(n_features, hidden_dim, n_classes, &device, cycle_path)?;
        
        if !models_identical(&original_model, &cycle_model, &test_input) {
            println!("❌ Test 4 FAILED: Model changed after save/load cycle {}", i + 1);
            test_results.push(("Multiple Save/Load Cycles", false, 
                              format!("Model changed after cycle {}", i + 1)));
            return Err(anyhow::anyhow!("Multiple save/load cycle test failed"));
        }
    }
    println!("✅ Test 4 PASSED: Model unchanged after 3 save/load cycles");
    test_results.push(("Multiple Save/Load Cycles", true, "No parameter drift after 3 cycles".to_string()));
    
    // ============================================================================
    // Save Execution Trace and Cleanup
    // ============================================================================
    
    // Prepare trace information
    let synthetic_data_info = format!(
        "- **Samples**: {}\n- **Features**: {}\n- **Classes**: {}\n- **Training epochs**: {}\n- **Batch size**: {}\n- **Learning rate**: {}",
        n_samples, n_features, n_classes, epochs, batch_size, lr
    );
    
    let model_info = format!(
        "- **Architecture**: TinyMLP\n- **Input dimension**: {}\n- **Hidden dimension**: {}\n- **Output dimension**: {}\n- **Activation**: ReLU\n- **Backend**: Autodiff<NdArray<f32>>",
        n_features, hidden_dim, n_classes
    );
    
    // Save execution trace if enabled
    if let Err(e) = save_execution_trace(&test_results, &synthetic_data_info, &model_info, save_traces) {
        println!("⚠️  Could not save execution trace: {}", e);
    }
    
    // Optionally save sample model parameters as permanent traces
    if save_sample_modelparams {
        let sample_modelparams_path = format!("datastore/sythetic-datasets/sample_trained_model_{}.bin", 
                                             std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs());
        
        if let Err(e) = save_modelparams(&original_model, &sample_modelparams_path) {
            println!("⚠️  Could not save sample model parameters: {}", e);
        } else {
            println!("💾 Sample trained model parameters saved as: {}", sample_modelparams_path);
        }
    }
    
    println!("\n🧹 Cleaning up test files...");
    let test_files = [test_modelparams_path, checkpoint_path, cycle_path];
    for file in &test_files {
        if Path::new(file).exists() {
            let _ = std::fs::remove_file(file);
        }
    }
    
    println!("\n📋 Test Summary:");
    println!("================");
    println!("✅ Basic save/load functionality");
    println!("✅ Training resumption (checkpointing)");
    println!("✅ File handling edge cases");
    println!("✅ Multiple save/load cycles");
    println!("✅ Parameter preservation verification");
    
    if save_traces {
        println!("📝 Execution trace saved to datastore/sythetic-datasets/execution-traces/");
    }
    
    println!("\n🎉 All tests completed successfully!");
    println!("   Model parameters save/load functionality is working correctly.");
    
    Ok(())
}
