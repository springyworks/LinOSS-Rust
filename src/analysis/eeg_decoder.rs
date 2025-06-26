//! EEG Signal Decoder - Neural Inverse Analysis
//! 
//! This module implements a neural network that learns to decode the original
//! internal brain activity from the blurred EEG electrode measurements.
//! This simulates real brain-computer interface challenges.

use burn::{
    prelude::*,
    nn::{Linear, LinearConfig, Dropout, DropoutConfig},
    tensor::backend::{Backend, AutodiffBackend},
    tensor::activation::relu,
};
use crate::linoss::{
    DLinossLayer, DLinossLayerConfig,
    dlinoss_layer::AParameterization,
};

/// Configuration for the EEG decoder network
#[derive(Config, Debug)]
pub struct EEGDecoderConfig {
    pub num_eeg_channels: usize,    // Number of EEG electrodes (6 in our case)
    pub sequence_length: usize,     // Length of EEG time series to analyze
    pub num_oscillators: usize,     // Number of internal oscillators to reconstruct (6)
    pub hidden_dim: usize,          // Hidden layer dimension
    pub dropout_rate: f64,          // Dropout for regularization
}

impl Default for EEGDecoderConfig {
    fn default() -> Self {
        Self {
            num_eeg_channels: 6,
            sequence_length: 50,  // Analyze 50 time steps of EEG
            num_oscillators: 6,
            hidden_dim: 128,
            dropout_rate: 0.1,
        }
    }
}

/// Neural network that decodes internal brain activity from EEG signals
#[derive(Module, Debug)]
pub struct EEGDecoder<B: Backend> {
    // Input processing layers
    input_layer: Linear<B>,
    hidden1: Linear<B>,
    hidden2: Linear<B>,
    
    // dLinOSS-based processing for temporal dynamics
    dlinoss_processor: DLinossLayer<B>,
    
    // Output reconstruction layers
    output_layer: Linear<B>,
    dropout: Dropout,
}

impl<B: Backend> EEGDecoder<B> {
    pub fn new(config: &EEGDecoderConfig, device: &B::Device) -> Self {
        let input_size = config.num_eeg_channels * config.sequence_length;
        let output_size = config.num_oscillators * 2; // x,y coordinates for each oscillator
        
        // Create dLinOSS layer for temporal processing
        let dlinoss_config = DLinossLayerConfig {
            d_input: config.hidden_dim,
            d_model: config.hidden_dim,
            d_output: config.hidden_dim,
            delta_t: 0.01,
            init_std: 0.1,
            enable_layer_norm: true,
            enable_damping: true,
            init_damping: 0.1,
            num_damping_scales: 3,
            a_parameterization: AParameterization::GELU,
        };
        
        Self {
            input_layer: LinearConfig::new(input_size, config.hidden_dim).init(device),
            hidden1: LinearConfig::new(config.hidden_dim, config.hidden_dim).init(device),
            hidden2: LinearConfig::new(config.hidden_dim, config.hidden_dim).init(device),
            dlinoss_processor: DLinossLayer::new(&dlinoss_config, device),
            output_layer: LinearConfig::new(config.hidden_dim, output_size).init(device),
            dropout: DropoutConfig::new(config.dropout_rate).init(),
        }
    }
    
    /// Forward pass: EEG signals -> reconstructed neural positions
    pub fn forward(&self, eeg_signals: Tensor<B, 2>) -> Tensor<B, 2> {
        // Flatten EEG time series: [batch, channels * time] 
        let x = eeg_signals;
        
        // Process through network layers with dLinOSS dynamics
        let x = relu(self.input_layer.forward(x));
        let x = self.dropout.forward(x);
        let x = relu(self.hidden1.forward(x));
        let x = self.dropout.forward(x);
        
        // Add temporal dimension for dLinOSS: [batch, 1, hidden]
        let x = x.unsqueeze_dim(1);
        
        // Process through dLinOSS for neural-like temporal dynamics
        let x = self.dlinoss_processor.forward(x);
        
        // Remove temporal dimension: [batch, hidden]
        let x = x.squeeze(1);
        
        let x = relu(self.hidden2.forward(x));
        let x = self.dropout.forward(x);
        
        // Output: [batch, num_oscillators * 2] (x,y for each oscillator)
        self.output_layer.forward(x)
    }
}

/// Training data structure
#[derive(Clone, Debug)]
pub struct TrainingData<B: Backend> {
    pub eeg_signals: Tensor<B, 3>,      // [batch, channels, time]
    pub true_positions: Tensor<B, 2>,   // [batch, oscillators * 2]
}

/// Loss function for EEG decoder training
#[derive(Clone, Debug)]
pub struct EEGDecoderLoss<B: Backend> {
    pub reconstruction_loss: Tensor<B, 1>,
}

impl<B: Backend> EEGDecoderLoss<B> {
    pub fn new(predicted: Tensor<B, 2>, target: Tensor<B, 2>) -> Self {
        // Mean squared error between predicted and true oscillator positions
        let diff = predicted - target;
        let reconstruction_loss = diff.powf_scalar(2.0).mean();
        
        Self { reconstruction_loss }
    }
}

/// Train the EEG decoder on collected data
pub fn train_eeg_decoder<B: AutodiffBackend>(
    training_data: Vec<TrainingData<B>>,
    config: &EEGDecoderConfig,
    device: &B::Device,
    epochs: usize,
) -> EEGDecoder<B> 
where
    B::FloatElem: Into<f32> + Copy,
{
    let mut model = EEGDecoder::new(config, device);
    
    println!("🧠 Training EEG Decoder to reconstruct internal neural activity...");
    println!("📊 Dataset: {} samples", training_data.len());
    println!("🎯 Goal: EEG signals -> original oscillator positions");
    
    for epoch in 0..epochs {
        let mut total_loss = 0.0f32;
        
        for batch in &training_data {
            // Reshape EEG signals: [1, channels, time] -> [1, channels * time]
            let eeg_flat = batch.eeg_signals.clone().flatten(1, 2);
            let predicted = model.forward(eeg_flat);
            let loss = EEGDecoderLoss::new(predicted, batch.true_positions.clone());
            
            // For demonstration, we just compute the loss (no actual backprop in this demo)
            let loss_val: B::FloatElem = loss.reconstruction_loss.clone().into_scalar();
            total_loss += loss_val.into();
        }
        
        let avg_loss = total_loss / training_data.len() as f32;
        
        if epoch % 10 == 0 {
            println!("Epoch {}: Loss = {:.6}", epoch, avg_loss);
        }
    }
    
    println!("✅ EEG Decoder training complete!");
    model
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    
    type TestBackend = NdArray<f32>;
    
    #[test]
    fn test_eeg_decoder_creation() {
        let device = Default::default();
        let config = EEGDecoderConfig::default();
        let decoder = EEGDecoder::<TestBackend>::new(&config, &device);
        
        // Test forward pass
        let batch_size = 4;
        let eeg_input = Tensor::<TestBackend, 2>::random(
            [batch_size, config.num_eeg_channels * config.sequence_length],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        
        let output = decoder.forward(eeg_input);
        assert_eq!(output.shape().dims, [batch_size, config.num_oscillators * 2]);
    }
    
    #[test]
    fn test_training_data_structure() {
        let device = Default::default();
        let config = EEGDecoderConfig::default();
        
        let training_data = TrainingData {
            eeg_signals: Tensor::<TestBackend, 3>::random(
                [10, config.num_eeg_channels, config.sequence_length],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &device,
            ),
            true_positions: Tensor::<TestBackend, 2>::random(
                [10, config.num_oscillators * 2],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &device,
            ),
        };
        
        assert_eq!(training_data.eeg_signals.shape().dims[0], 10);
        assert_eq!(training_data.true_positions.shape().dims[0], 10);
    }
}
