// Test script to demonstrate Burn model save/load functionality
// File: src/bin/test_model_save_load.rs

use burn::backend::{NdArray, Autodiff};
use burn::tensor::{Tensor, backend::Backend};
use burn::module::Module;
use burn::nn::{Linear, LinearConfig, Relu};
use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
use burn::tensor::TensorData;
use anyhow::Result;
use std::path::Path;

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

fn main() -> Result<()> {
    println!("🧪 Testing Burn Model Parameters Save/Load Functionality");
    println!("======================================================");
    
    let device = <MyBackend as Backend>::Device::default();
    let input_dim = 4;
    let hidden_dim = 8;
    let output_dim = 3;
    let modelparams_path = "datastore/model-parameters/test_modelparams.bin";
    
    // Create a model
    println!("\n1. Creating a new model...");
    let original_model = TinyMLP::<MyBackend>::new(input_dim, hidden_dim, output_dim, &device);
    
    // Create some test data
    println!("2. Creating test data...");
    let test_data = vec![1.0, 2.0, 3.0, 4.0];
    let x_data = TensorData::new(test_data, [1, input_dim]);
    let x = Tensor::<MyBackend, 2>::from_data(x_data, &device);
    
    // Get original predictions
    println!("3. Getting original model predictions...");
    let original_output = original_model.forward(x.clone());
    let original_values = original_output.to_data().convert::<f32>().into_vec::<f32>().unwrap();
    println!("   Original output: {:?}", original_values);
    
    // Save the model parameters
    println!("4. Saving the model parameters...");
    save_modelparams(&original_model, modelparams_path)?;
    
    // Load the model parameters
    println!("5. Loading the model parameters...");
    let loaded_model = load_modelparams(input_dim, hidden_dim, output_dim, &device, modelparams_path)?;
    
    // Get loaded model predictions
    println!("6. Getting loaded model predictions...");
    let loaded_output = loaded_model.forward(x.clone());
    let loaded_values = loaded_output.to_data().convert::<f32>().into_vec::<f32>().unwrap();
    println!("   Loaded output:   {:?}", loaded_values);
    
    // Compare outputs
    println!("7. Comparing outputs...");
    let mut match_count = 0;
    let tolerance = 1e-6;
    
    for (i, (orig, loaded)) in original_values.iter().zip(loaded_values.iter()).enumerate() {
        let diff = (orig - loaded).abs();
        let matches = diff < tolerance;
        println!("   Index {}: orig={:.6}, loaded={:.6}, diff={:.2e}, match={}", 
                 i, orig, loaded, diff, matches);
        if matches {
            match_count += 1;
        }
    }
    
    if match_count == original_values.len() {
        println!("\n🎉 SUCCESS: All outputs match! Model parameters save/load works correctly.");
    } else {
        println!("\n❌ FAILURE: Some outputs don't match. Check save/load implementation.");
    }
    
    // Clean up
    if Path::new(modelparams_path).exists() {
        std::fs::remove_file(modelparams_path)?;
        println!("🧹 Cleaned up test model parameters file.");
    }
    
    println!("\n✅ Test completed!");
    Ok(())
}
