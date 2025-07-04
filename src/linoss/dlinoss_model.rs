//! Complete D-LinOSS Model Implementation
//! 
//! Full model architecture following "Learning to Dissipate Energy in Oscillatory State-Space Models"
//! Includes input encoding, multiple D-LinOSS blocks, and output projection.

use burn::{
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{backend::Backend, Distribution, Tensor},
};
use crate::linoss::dlinoss_block::{DLinossBlock, DLinossBlockConfig};

/// Configuration for complete D-LinOSS model
#[derive(Config, Debug)]
pub struct DLinossModelConfig {
    /// Number of D-LinOSS blocks
    pub num_blocks: usize,
    /// Input dimension
    pub input_dim: usize,
    /// SSM state size for each block
    pub ssm_size: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Output dimension
    pub output_dim: usize,
    /// Enable learnable damping in all blocks
    pub damping: bool,
    /// Whether this is a classification task
    pub classification: bool,
    /// Whether to use linear output (no tanh)
    pub linear_output: bool,
    /// Output step (for prediction tasks)
    pub output_step: usize,
    /// Discretization parameters
    pub r_min: f64,
    pub theta_max: f64,
    /// Dropout rate
    pub dropout_rate: f64,
}

impl DLinossModelConfig {
    /// Create D-LinOSS configuration for classification
    pub fn classification(input_dim: usize, num_classes: usize) -> Self {
        Self {
            num_blocks: 4,
            input_dim,
            ssm_size: 64,
            hidden_dim: 128,
            output_dim: num_classes,
            damping: true,
            classification: true,
            linear_output: true,
            output_step: 1,
            r_min: 0.9,
            theta_max: std::f64::consts::PI,
            dropout_rate: 0.05,
        }
    }
    
    /// Create D-LinOSS configuration for regression/forecasting
    pub fn regression(input_dim: usize, output_dim: usize) -> Self {
        Self {
            num_blocks: 6,
            input_dim,
            ssm_size: 64,
            hidden_dim: 128,
            output_dim,
            damping: true,
            classification: false,
            linear_output: false,
            output_step: 1,
            r_min: 0.9,
            theta_max: std::f64::consts::PI,
            dropout_rate: 0.05,
        }
    }
    
    /// Create vanilla LinOSS configuration (no damping)
    pub fn vanilla_linoss(input_dim: usize, output_dim: usize) -> Self {
        Self {
            num_blocks: 4,
            input_dim,
            ssm_size: 64,
            hidden_dim: 128,
            output_dim,
            damping: false,  // Key difference
            classification: false,
            linear_output: false,
            output_step: 1,
            r_min: 0.9,
            theta_max: std::f64::consts::PI,
            dropout_rate: 0.05,
        }
    }
}

/// Complete D-LinOSS model with proper architecture
#[derive(Module, Debug)]
pub struct DLinossModel<B: Backend> {
    /// Input encoder (projects input to hidden dimension)
    input_encoder: Linear<B>,
    /// Stack of D-LinOSS blocks
    blocks: Vec<DLinossBlock<B>>,
    /// Output projection layer
    output_layer: Linear<B>,
    /// Model configuration
    config: DLinossModelConfig,
}

impl<B: Backend> DLinossModel<B> {
    /// Create a new D-LinOSS model
    pub fn new(config: &DLinossModelConfig, device: &B::Device) -> Self {
        // Input encoder: input_dim -> hidden_dim
        let input_encoder = LinearConfig::new(config.input_dim, config.hidden_dim)
            .init(device);
        
        // Create D-LinOSS blocks
        let mut blocks = Vec::new();
        for _ in 0..config.num_blocks {
            let block_config = DLinossBlockConfig {
                ssm_size: config.ssm_size,
                hidden_dim: config.hidden_dim,
                input_features: config.hidden_dim,
                output_features: config.hidden_dim,
                damping: config.damping,
                delta_t: 1.0,
                dropout_rate: config.dropout_rate,
                r_min: config.r_min,
                theta_max: config.theta_max,
                init_std: 0.02,
            };
            blocks.push(DLinossBlock::new(&block_config, device));
        }
        
        // Output layer: hidden_dim -> output_dim
        let output_layer = LinearConfig::new(config.hidden_dim, config.output_dim)
            .init(device);
        
        Self {
            input_encoder,
            blocks,
            output_layer,
            config: config.clone(),
        }
    }
    
    /// Forward pass through complete D-LinOSS model
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, seq_len, _] = input.dims();
        
        // Input encoding: project each timestep to hidden dimension
        // input: [batch, seq_len, input_dim] -> [batch, seq_len, hidden_dim]
        let mut x = Tensor::zeros([batch_size, seq_len, self.config.hidden_dim], &input.device());
        for t in 0..seq_len {
            let input_t = input.clone().slice([0..batch_size, t..t+1, 0..input.dims()[2]]).squeeze(1);
            let encoded_t = self.input_encoder.forward(input_t);
            x = x.slice_assign([0..batch_size, t..t+1, 0..self.config.hidden_dim], encoded_t.unsqueeze_dim(1));
        }
        
        // Pass through D-LinOSS blocks
        for block in &self.blocks {
            x = block.forward(x);
        }
        
        // Output processing
        if self.config.classification {
            // For classification: average pool over sequence, then project
            let pooled = x.mean_dim(1); // [batch, hidden_dim]
            let logits = self.output_layer.forward(pooled);
            
            // Apply softmax
            burn::tensor::activation::softmax(logits, 1).unsqueeze_dim(1) // [batch, 1, output_dim]
        } else {
            // For regression: project each timestep
            let mut outputs = Vec::new();
            for t in (self.config.output_step - 1..seq_len).step_by(self.config.output_step) {
                let x_t = x.clone().slice([0..batch_size, t..t+1, 0..self.config.hidden_dim]).squeeze(1);
                let output_t = self.output_layer.forward(x_t);
                
                let final_output = if self.config.linear_output {
                    output_t
                } else {
                    burn::tensor::activation::tanh(output_t)
                };
                
                outputs.push(final_output.unsqueeze_dim(1));
            }
            
            if outputs.is_empty() {
                // Fallback: at least return something from the last timestep
                let x_last = x.clone().slice([0..batch_size, seq_len-1..seq_len, 0..self.config.hidden_dim]).squeeze(1);
                let output_last = self.output_layer.forward(x_last);
                let final_output = if self.config.linear_output {
                    output_last
                } else {
                    burn::tensor::activation::tanh(output_last)
                };
                final_output.unsqueeze_dim(1)
            } else {
                Tensor::cat(outputs, 1)
            }
        }
    }
    
    /// Get model configuration
    pub fn config(&self) -> &DLinossModelConfig {
        &self.config
    }
    
    /// Check if damping is enabled in this model
    pub fn has_damping(&self) -> bool {
        self.config.damping
    }
    
    /// Get number of parameters (approximate)
    pub fn num_parameters(&self) -> usize {
        // Rough estimate based on architecture
        let encoder_params = self.config.input_dim * self.config.hidden_dim + self.config.hidden_dim;
        let output_params = self.config.hidden_dim * self.config.output_dim + self.config.output_dim;
        
        // Each block has roughly:
        // - SSM parameters: ssm_size * (2 + hidden_dim * 2) for A, G, B, C matrices
        // - GLU parameters: hidden_dim * hidden_dim * 2  
        // - Layer norm: hidden_dim * 2
        let block_params = self.config.ssm_size * (2 + self.config.hidden_dim * 2) + 
                          self.config.hidden_dim * self.config.hidden_dim * 2 + 
                          self.config.hidden_dim * 2;
        
        encoder_params + output_params + self.config.num_blocks * block_params
    }
}

/// Convenience function to create D-LinOSS model for time series classification
pub fn create_dlinoss_classifier<B: Backend>(
    input_dim: usize,
    num_classes: usize,
    device: &B::Device,
) -> DLinossModel<B> {
    let config = DLinossModelConfig::classification(input_dim, num_classes);
    DLinossModel::new(&config, device)
}

/// Convenience function to create D-LinOSS model for time series forecasting
pub fn create_dlinoss_forecaster<B: Backend>(
    input_dim: usize,
    output_dim: usize,
    device: &B::Device,
) -> DLinossModel<B> {
    let config = DLinossModelConfig::regression(input_dim, output_dim);
    DLinossModel::new(&config, device)
}

/// Convenience function to create vanilla LinOSS model (no damping)
pub fn create_vanilla_linoss<B: Backend>(
    input_dim: usize,
    output_dim: usize,
    device: &B::Device,
) -> DLinossModel<B> {
    let config = DLinossModelConfig::vanilla_linoss(input_dim, output_dim);
    DLinossModel::new(&config, device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_dlinoss_classifier() {
        let device = Default::default();
        let model = create_dlinoss_classifier::<TestBackend>(10, 5, &device);
        
        let input = Tensor::<TestBackend, 3>::random([4, 20, 10], Distribution::Normal(0.0, 1.0), &device);
        let output = model.forward(input);
        
        // Classification output should be [batch, 1, num_classes]
        assert_eq!(output.dims(), [4, 1, 5]);
        assert!(model.has_damping());
    }
    
    #[test]
    fn test_dlinoss_forecaster() {
        let device = Default::default();
        let model = create_dlinoss_forecaster::<TestBackend>(3, 1, &device);
        
        let input = Tensor::<TestBackend, 3>::random([2, 50, 3], Distribution::Normal(0.0, 1.0), &device);
        let output = model.forward(input);
        
        // Should have output for prediction
        assert_eq!(output.dims()[0], 2); // batch size
        assert_eq!(output.dims()[2], 1); // output dim
        assert!(model.has_damping());
    }
    
    #[test]
    fn test_vanilla_linoss() {
        let device = Default::default();
        let model = create_vanilla_linoss::<TestBackend>(5, 2, &device);
        
        let input = Tensor::<TestBackend, 3>::random([3, 15, 5], Distribution::Normal(0.0, 1.0), &device);
        let _output = model.forward(input);
        
        // Vanilla LinOSS should not have damping
        assert!(!model.has_damping());
    }
    
    #[test]
    fn test_model_parameters() {
        let device = Default::default();
        let config = DLinossModelConfig::classification(10, 5);
        let model = DLinossModel::<TestBackend>::new(&config, &device);
        
        let num_params = model.num_parameters();
        println!("Estimated parameters: {}", num_params);
        assert!(num_params > 0);
    }
}
