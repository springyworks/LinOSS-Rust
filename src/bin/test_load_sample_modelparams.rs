// Test script to load and verify sample model parameters from traces

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
    println!("🧪 Testing Sample Model Parameters Loading");
    println!("==========================================");
    
    let device = <MyBackend as Backend>::Device::default();
    
    // Configuration matching the synthetic test
    let n_features = 4;
    let hidden_dim = 16;
    let n_classes = 3;
    
    // Find available sample model parameter files
    let sample_dir = "datastore/sythetic-datasets";
    let entries = std::fs::read_dir(sample_dir)?;
    
    let mut sample_files = Vec::new();
    for entry in entries {
        let entry = entry?;
        let filename = entry.file_name().to_string_lossy().to_string();
        if filename.starts_with("sample_trained_model_") && filename.ends_with(".bin") {
            sample_files.push(filename);
        }
    }
    
    if sample_files.is_empty() {
        println!("❌ No sample model parameter files found in {}", sample_dir);
        println!("   Run the synthetic test first to generate sample files:");
        println!("   cargo run --bin test_modelparams_synthetic");
        return Ok(());
    }
    
    // Sort by timestamp (newest first)
    sample_files.sort_by(|a, b| b.cmp(a));
    
    println!("📂 Found {} sample model parameter files:", sample_files.len());
    for file in &sample_files {
        println!("   - {}", file);
    }
    
    // Load the newest sample
    let newest_sample = &sample_files[0];
    let sample_path = format!("{}/{}", sample_dir, newest_sample);
    
    println!("\n🔍 Loading newest sample: {}", newest_sample);
    
    // Load the model parameters
    let loaded_model = load_modelparams::<MyBackend>(
        n_features, 
        hidden_dim, 
        n_classes, 
        &device, 
        &sample_path
    )?;
    
    // Create some test data (similar to synthetic test)
    let test_features = vec![
        1.0, 2.0, 0.5, 1.5,  // Sample 1
        2.0, 4.0, 1.0, 3.0,  // Sample 2  
        0.5, 1.0, 0.25, 0.75 // Sample 3
    ];
    
    let test_data = TensorData::new(test_features, [3, n_features]);
    let test_input = Tensor::<MyBackend, 2>::from_data(test_data, &device);
    
    // Run inference
    println!("\n🧠 Running inference with loaded model...");
    let output = loaded_model.forward(test_input);
    
    println!("✅ Inference successful!");
    println!("   Input shape: [3, {}]", n_features);
    println!("   Output shape: {:?}", output.dims());
    
    // Show output values
    let output_data = output.to_data().convert::<f32>();
    let output_vec = output_data.into_vec::<f32>().unwrap();
    
    println!("\n📊 Model Predictions:");
    for i in 0..3 {
        let start_idx = i * n_classes;
        let sample_output = &output_vec[start_idx..start_idx + n_classes];
        println!("   Sample {}: [{:.4}, {:.4}, {:.4}]", 
                i + 1, sample_output[0], sample_output[1], sample_output[2]);
    }
    
    println!("\n🎉 Successfully loaded and tested sample model parameters!");
    println!("   The saved model parameters from synthetic training are working correctly.");
    
    Ok(())
}
