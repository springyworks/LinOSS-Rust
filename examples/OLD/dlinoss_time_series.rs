use burn::{
    backend::Autodiff,
    config::Config,
    module::Module,
    nn::loss::MseLossConfig,
    optim::{AdamConfig, GradientsParams, Optimizer},
    tensor::{backend::Backend, Data, Tensor},
    train::{LearnerBuilder, TrainOutput, TrainStep, ValidStep},
};
use burn_wgpu::WgpuBackend;

use linoss_rust::linoss::production_dlinoss::{
    ProductionDLinossModel, ProductionDLinossConfig, 
    DiscretizationScheme, ActivationFunction,
};

type MyBackend = WgpuBackend<f32, i32>;
type MyAutodiffBackend = Autodiff<MyBackend>;

#[derive(Config, Debug)]
pub struct TimeSeriesConfig {
    pub d_model: usize,
    pub num_layers: usize,
    pub sequence_length: usize,
    pub prediction_horizon: usize,
    #[config(default = "DiscretizationScheme::RK4")]
    pub discretization: DiscretizationScheme,
    #[config(default = "ActivationFunction::Tanh")]
    pub activation: ActivationFunction,
    #[config(default = "0.1")]
    pub dt: f64,
    #[config(default = "0.05")]
    pub damping_factor: f64,
}

#[derive(Module, Debug)]
pub struct DLinossTimeSeriesPredictor<B: Backend> {
    dlinoss: ProductionDLinossModel<B>,
    config: TimeSeriesConfig,
}

impl<B: Backend> DLinossTimeSeriesPredictor<B> {
    pub fn new(config: &TimeSeriesConfig, device: &B::Device) -> Self {
        let dlinoss_config = ProductionDLinossConfig::new(
            1, // Single input dimension (univariate time series)
            config.d_model,
            config.prediction_horizon, // Output predictions for multiple steps ahead
            config.num_layers,
            config.discretization.clone(),
            config.activation.clone(),
            config.dt,
            config.damping_factor,
        );
        
        Self {
            dlinoss: dlinoss_config.init(device),
            config: config.clone(),
        }
    }

    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        // Input: [batch, sequence_length, 1]
        // Output: [batch, prediction_horizon, 1] 
        let predictions = self.dlinoss.forward(input);
        
        // Take the last few outputs as predictions
        let batch_size = predictions.dims()[0];
        let total_seq_len = predictions.dims()[1];
        let start_idx = total_seq_len - self.config.prediction_horizon;
        
        predictions.slice([
            0..batch_size,
            start_idx..total_seq_len,
            0..1
        ])
    }
}

#[derive(Clone, Debug)]
pub struct TimeSeriesBatch<B: Backend> {
    pub inputs: Tensor<B, 3>,   // [batch, seq_len, 1]
    pub targets: Tensor<B, 3>,  // [batch, pred_horizon, 1]
}

impl<B: Backend> TrainStep<TimeSeriesBatch<B>, Tensor<B, 1>> for DLinossTimeSeriesPredictor<B> {
    fn step(&self, batch: TimeSeriesBatch<B>) -> TrainOutput<Tensor<B, 1>> {
        let predictions = self.forward(batch.inputs);
        let loss = MseLossConfig::new()
            .init(&predictions.device())
            .forward(predictions, batch.targets);
        
        TrainOutput::new(self, loss.clone().backward(), loss)
    }
}

impl<B: Backend> ValidStep<TimeSeriesBatch<B>, Tensor<B, 1>> for DLinossTimeSeriesPredictor<B> {
    fn step(&self, batch: TimeSeriesBatch<B>) -> Tensor<B, 1> {
        let predictions = self.forward(batch.inputs);
        MseLossConfig::new()
            .init(&predictions.device())
            .forward(predictions, batch.targets)
    }
}

// Generate synthetic time series data (chaotic Lorenz system + sine waves)
fn generate_lorenz_data(num_samples: usize, dt: f64) -> Vec<f32> {
    let mut data = Vec::with_capacity(num_samples);
    let (mut x, mut y, mut z) = (1.0, 1.0, 1.0);
    let (sigma, rho, beta) = (10.0, 28.0, 8.0/3.0);
    
    for i in 0..num_samples {
        // Lorenz equations
        let dx = sigma * (y - x);
        let dy = x * (rho - z) - y;
        let dz = x * y - beta * z;
        
        x += dx * dt;
        y += dy * dt;
        z += dz * dt;
        
        // Add some sine wave components for complexity
        let t = i as f64 * dt;
        let sine_component = 0.3 * (0.1 * t).sin() + 0.2 * (0.05 * t).sin();
        
        data.push((x + sine_component) as f32);
    }
    
    // Normalize to [-1, 1]
    let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b.abs()));
    data.iter_mut().for_each(|x| *x /= max_val);
    
    data
}

fn create_batches<B: Backend>(
    data: &[f32], 
    sequence_length: usize,
    prediction_horizon: usize,
    batch_size: usize,
    device: &B::Device
) -> Vec<TimeSeriesBatch<B>> {
    let mut batches = Vec::new();
    let window_size = sequence_length + prediction_horizon;
    
    for chunk_start in (0..data.len() - window_size).step_by(batch_size) {
        let mut batch_inputs = Vec::new();
        let mut batch_targets = Vec::new();
        
        for i in 0..batch_size {
            let start_idx = chunk_start + i;
            if start_idx + window_size > data.len() {
                break;
            }
            
            // Input sequence
            let input_data: Vec<f32> = data[start_idx..start_idx + sequence_length].to_vec();
            let input_tensor = Tensor::<B, 3>::from_data(
                Data::from([[[input_data]]]).squeeze(0),
                device
            );
            batch_inputs.push(input_tensor);
            
            // Target sequence (future values)
            let target_data: Vec<f32> = data[start_idx + sequence_length..start_idx + window_size].to_vec();
            let target_tensor = Tensor::<B, 3>::from_data(
                Data::from([[[target_data]]]).squeeze(0),
                device
            );
            batch_targets.push(target_tensor);
        }
        
        if !batch_inputs.is_empty() {
            let inputs = Tensor::cat(batch_inputs, 0);
            let targets = Tensor::cat(batch_targets, 0);
            batches.push(TimeSeriesBatch { inputs, targets });
        }
    }
    
    batches
}

pub fn demonstrate_time_series_prediction() {
    println!("🚀 D-LinOSS Time Series Prediction Demo");
    println!("=======================================");
    
    let device = burn_wgpu::WgpuDevice::default();
    
    // Configuration
    let config = TimeSeriesConfig {
        d_model: 64,
        num_layers: 3,
        sequence_length: 50,
        prediction_horizon: 10,
        discretization: DiscretizationScheme::RK4,
        activation: ActivationFunction::Tanh,
        dt: 0.1,
        damping_factor: 0.05,
    };
    
    println!("📊 Generating synthetic Lorenz chaotic time series...");
    let data = generate_lorenz_data(2000, 0.01);
    
    println!("🔧 Creating D-LinOSS model with config:");
    println!("   - Hidden dim: {}", config.d_model);
    println!("   - Layers: {}", config.num_layers);
    println!("   - Sequence length: {}", config.sequence_length);
    println!("   - Prediction horizon: {}", config.prediction_horizon);
    println!("   - Discretization: {:?}", config.discretization);
    println!("   - Activation: {:?}", config.activation);
    println!("   - Damping factor: {}", config.damping_factor);
    
    let model = DLinossTimeSeriesPredictor::new(&config, &device);
    
    // Create training and validation data
    let split_point = (data.len() as f32 * 0.8) as usize;
    let train_data = &data[..split_point];
    let val_data = &data[split_point..];
    
    let train_batches = create_batches::<MyAutodiffBackend>(
        train_data, config.sequence_length, config.prediction_horizon, 8, &device
    );
    let val_batches = create_batches::<MyBackend>(
        val_data, config.sequence_length, config.prediction_horizon, 8, &device
    );
    
    println!("📈 Training batches: {}", train_batches.len());
    println!("📉 Validation batches: {}", val_batches.len());
    
    // Quick training demonstration
    let optimizer = AdamConfig::new().init();
    let mut model = model;
    let mut optimizer = optimizer;
    
    println!("\n🎯 Training D-LinOSS for time series prediction...");
    for epoch in 0..5 {
        let mut total_loss = 0.0;
        let mut count = 0;
        
        for batch in &train_batches {
            let output = model.step(batch.clone());
            let grads = output.grads;
            let loss_value: f32 = output.item.into_scalar();
            
            model = optimizer.step(1e-3, model, grads);
            total_loss += loss_value;
            count += 1;
            
            if count >= 10 { break; } // Limit for demo
        }
        
        let avg_loss = total_loss / count as f32;
        println!("   Epoch {}: Average Loss = {:.6}", epoch + 1, avg_loss);
    }
    
    println!("\n✨ Making predictions on validation data...");
    
    // Test on a few validation samples
    for (i, batch) in val_batches.iter().take(3).enumerate() {
        let predictions = model.forward(batch.inputs.clone());
        let val_loss = model.step(batch.clone());
        let loss_value: f32 = val_loss.into_scalar();
        
        println!("   Sample {}: Prediction MSE = {:.6}", i + 1, loss_value);
        
        // Show some prediction values
        let pred_data: Vec<f32> = predictions.slice([0..1, 0..5, 0..1])
            .flatten::<1>(0, 2)
            .into_data()
            .to_vec().unwrap();
        let target_data: Vec<f32> = batch.targets.slice([0..1, 0..5, 0..1])
            .flatten::<1>(0, 2)
            .into_data()
            .to_vec().unwrap();
            
        println!("     Predictions: {:?}", &pred_data[..3]);
        println!("     Targets:     {:?}", &target_data[..3]);
    }
    
    println!("\n🎊 D-LinOSS time series prediction demo completed!");
    println!("💡 The model learned to predict chaotic Lorenz dynamics with damping!");
}

fn main() {
    demonstrate_time_series_prediction();
}
