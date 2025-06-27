// Minimal Burn-based training example for Iris dataset with model saving/loading
// File: src/bin/tinyburn_iris_train_with_checkpoint.rs

#[cfg(feature = "npy_files")]
mod npy_training {

use burn::backend::{NdArray, Autodiff};
use burn::tensor::{Tensor, backend::Backend};
use burn::module::Module;
use burn::nn::{Linear, LinearConfig, Relu};
use burn::optim::{AdamConfig, Optimizer, GradientsParams};
use burn::tensor::loss::cross_entropy_with_logits;
use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
use ndarray_npy::read_npy;
use anyhow::Result;
use burn::tensor::TensorData;
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

// Function to save model parameters
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

// Function to load model parameters
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
    
    // Create a new model with same architecture
    let mut model = TinyMLP::<B>::new(input_dim, hidden_dim, output_dim, device);
    
    // Load the record
    let record = recorder
        .load(path.into(), device)
        .map_err(|e| anyhow::anyhow!("Failed to load modelparams: {}", e))?;
    
    // Load the record into the model
    model = model.load_record(record);
    
    println!("✅ Model parameters loaded from: {}", path);
    Ok(model)
}

fn training_main() -> Result<()> {
    // Configuration
    let features: ndarray::Array2<f32> = read_npy("datastore/processed-by-python/iris_features.npy")?;
    let labels: ndarray::Array1<u8> = read_npy("datastore/processed-by-python/iris_labels.npy")?;
    let num_classes = 3;
    let input_dim = features.shape()[1];
    let hidden_dim = 16;
    let batch_size = 16;
    let epochs = 10;
    let lr = 1e-2;
    let device = <MyBackend as Backend>::Device::default();
    
    // Model parameter file paths
    let modelparams_checkpoint_path = "datastore/model-parameters/tiny_mlp_iris_checkpoint.bin";
    let modelparams_final_path = "datastore/model-parameters/tiny_mlp_iris_final.bin";

    // Convert data to tensors
    let (x_vec, _) = features.to_owned().into_raw_vec_and_offset();
    let (y_vec, _) = labels.to_owned().into_raw_vec_and_offset();
    let n_samples = features.shape()[0];
    let x_data = TensorData::new(x_vec, [n_samples, input_dim]);
    let y_data = TensorData::new(y_vec, [n_samples]);
    let x = Tensor::<MyBackend, 2>::from_data(x_data, &device);
    let y = Tensor::<MyBackend, 1>::from_data(y_data, &device);

    // Try to load existing model parameters, or create a new model
    let mut model = if Path::new(modelparams_checkpoint_path).exists() {
        println!("🔄 Loading existing model from checkpoint...");
        load_modelparams(input_dim, hidden_dim, num_classes, &device, modelparams_checkpoint_path)?
    } else {
        println!("🆕 Creating new model...");
        TinyMLP::<MyBackend>::new(input_dim, hidden_dim, num_classes, &device)
    };

    let mut optimizer = AdamConfig::new().init();

    println!("🚀 Starting training for {} epochs...", epochs);
    
    // Training loop
    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        let mut n_batches = 0;
        let dims = x.dims();
        let n_samples = dims[0];
        
        for i in (0..n_samples).step_by(batch_size) {
            let end = (i + batch_size).min(n_samples);
            let xb = x.clone().slice([i..end, 0..input_dim]);
            #[allow(clippy::single_range_in_vec_init)]
            let yb = y.clone().slice([i..end]);
            
            // Burn expects 2D labels for cross_entropy_with_logits
            let yb2 = yb.clone().unsqueeze_dim(1);
            let logits = model.forward(xb.clone());
            let loss = cross_entropy_with_logits(logits, yb2);
            
            let loss_value = loss.clone().mean().to_data().convert::<f32>().into_vec::<f32>().unwrap()[0];
            total_loss += loss_value;
            n_batches += 1;
            
            // Backward and update
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optimizer.step(lr, model, grads);
        }
        
        let avg_loss = total_loss / n_batches as f32;
        println!("Epoch {}: avg loss = {:.4}", epoch, avg_loss);
        
        // Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0 {
            save_modelparams(&model, modelparams_checkpoint_path)?;
            println!("💾 Saved checkpoint at epoch {}", epoch + 1);
        }
    }

    // Save final model parameters
    save_modelparams(&model, modelparams_final_path)?;
    println!("🎉 Training completed! Final model parameters saved.");

    // Demonstrate loading and inference
    println!("\n🔍 Testing model parameters loading and inference...");
    let loaded_model = load_modelparams(input_dim, hidden_dim, num_classes, &device, modelparams_final_path)?;
    
    // Test with first few samples
    let test_x = x.clone().slice([0..3, 0..input_dim]);
    #[allow(clippy::single_range_in_vec_init)]
    let test_y = y.clone().slice([0..3]);
    
    let predictions = loaded_model.forward(test_x);
    let pred_classes = predictions.clone().argmax(1);
    
    println!("Test predictions:");
    let pred_data = pred_classes.to_data().convert::<u8>().into_vec::<u8>().unwrap();
    let true_data = test_y.to_data().convert::<u8>().into_vec::<u8>().unwrap();
    
    for i in 0..3 {
        println!("  Sample {}: predicted={}, actual={}", i, pred_data[i], true_data[i]);
    }

    println!("\n📁 Model parameter files saved to:");
    println!("  - Checkpoint: {}", modelparams_checkpoint_path);
    println!("  - Final parameters: {}", modelparams_final_path);
    
    Ok(())
}

}

#[cfg(not(feature = "npy_files"))]
fn main() {
    println!("NPY file I/O requires 'npy_files' feature. Enable it with:");
    println!("cargo run --features npy_files --bin tinyburn_iris_train_with_checkpoint");
}

#[cfg(feature = "npy_files")]
fn main() -> anyhow::Result<()> {
    npy_training::training_main()
}
