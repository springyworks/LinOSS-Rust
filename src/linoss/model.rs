// src/linoss/model.rs
// Implementation of the FullLinOSSModel using the Burn framework.

use burn::module::Module; // Removed AutodiffModule
// use burn::record::Record; // Removed Record
use burn::nn::{Linear, LinearConfig};
use burn::prelude::Config;
use burn::tensor::{Tensor, backend::Backend};
use log::{debug, info};
// use serde::{Serialize, Deserialize}; // Removed unused imports
use num_traits::FloatConst;

use super::block::{LinossBlock, LinossBlockConfig};

/// Output of a LinOSS layer or block, containing the main output and the hidden state.
#[derive(Debug)]
pub struct LinossOutput<B: Backend, const D: usize> {
    /// The main output tensor (e.g., x_t).
    pub output: Tensor<B, D>,
    /// The hidden state tensor (e.g., y_t or a tuple of states).
    /// For LinossLayer, this is y_t. For LinossBlock, it could be the LinossLayer's y_t.
    pub hidden_state: Option<Tensor<B, D>>,
}

/// Configuration for the `FullLinossModel`.
#[derive(Config, Debug)]
pub struct FullLinossModelConfig {
    pub d_input: usize,
    pub d_model: usize,
    pub d_output: usize,
    pub n_layers: usize,
    pub linoss_block_config: LinossBlockConfig,
}

impl FullLinossModelConfig {
    /// Initializes a `FullLinossModel` with the given device.
    pub fn init<B: Backend>(&self, device: &B::Device) -> FullLinossModel<B> 
    where B::FloatElem: FloatConst
    {
        let encoder = LinearConfig::new(self.d_input, self.d_model).init(device);
        let decoder = LinearConfig::new(self.d_model, self.d_output).init(device);

        let mut blocks = Vec::with_capacity(self.n_layers);
        for _ in 0..self.n_layers {
            blocks.push(self.linoss_block_config.init(self.d_model, device));
        }

        FullLinossModel {
            encoder,
            blocks,
            decoder,
        }
    }
}

/// The main LinOSS model, stacking multiple LinossBlocks.
///
/// Comprises an input encoder, a series of LinossBlocks, and an output decoder.
#[derive(Module, Debug)] // This derive should handle Module, AutodiffModule, and Record generation
pub struct FullLinossModel<B: Backend> {
    encoder: Linear<B>,
    blocks: Vec<LinossBlock<B>>,
    decoder: Linear<B>,
}

impl<B: Backend> FullLinossModel<B>
where
    B::FloatElem: From<f32> + std::ops::Mul<Output = B::FloatElem> + Copy + FloatConst,
{
    /// Processes an input sequence through the LinOSS model.
    ///
    /// # Arguments
    ///
    /// * `input`: A tensor of shape `[batch_size, sequence_length, d_input]`.
    ///
    /// # Returns
    ///
    /// * A tensor of shape `[batch_size, sequence_length, d_output]`.
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        debug!(
            "FullLinossModel: Before Encoder, input shape: {:?}",
            input.dims()
        );
        let mut x = self.encoder.forward(input);
        debug!(
            "FullLinossModel: After Encoder, Before Blocks, x shape: {:?}",
            x.dims()
        );

        for (i, block) in self.blocks.iter().enumerate() {
            debug!(
                "FullLinossModel: Before Block {}, x shape: {:?}",
                i,
                x.dims()
            );
            x = block.forward(x);
            debug!(
                "FullLinossModel: After Block {}, x shape: {:?}",
                i,
                x.dims()
            );
        }

        debug!(
            "FullLinossModel: After Blocks, Before Decoder, x shape: {:?}",
            x.dims()
        );
        let output = self.decoder.forward(x);
        info!(
            "FullLinossModel: After Decoder (Final Output), output shape: {:?}",
            output.dims()
        );

        output
    }
}

// Manual implementations of Module, AutodiffModule, and Record for FullLinossModelRecord
// are removed as #[derive(Module)] should handle them.

#[cfg(test)]
mod tests {
    use super::*; 
    use burn::tensor::{Tensor, Distribution, backend::Backend};
    use log::{debug, info};

    type MyTestBackend = burn::backend::ndarray::NdArray<f32>;

    fn init_test_logger() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    #[test]
    fn test_full_linoss_model_forward_pass() {
        init_test_logger();
        info!("--- Running test_full_linoss_model_forward_pass (Original Dimensions) ---");
        let device: <MyTestBackend as Backend>::Device = Default::default();

        let d_input = 10;
        let d_model = 20;
        let d_output = 5;
        let n_layers = 2;
        let batch_size = 2;
        let seq_len = 3;

        run_model_test_with_params::<MyTestBackend>(
            d_input, d_model, d_output, n_layers, batch_size, seq_len, &device,
        );
        info!(
            "--- test_full_linoss_model_forward_pass (Original Dimensions) completed successfully ---"
        );
    }

    #[test]
    fn test_full_linoss_model_forward_pass_varied_dimensions() {
        init_test_logger();
        info!("--- Running test_full_linoss_model_forward_pass (Varied Dimensions) ---");
        let device: <MyTestBackend as Backend>::Device = Default::default();

        let d_input = 8;
        let d_model = 16;
        let d_output = 4;
        let n_layers = 1;
        let batch_size = 1;
        let seq_len = 5;

        run_model_test_with_params::<MyTestBackend>(
            d_input, d_model, d_output, n_layers, batch_size, seq_len, &device,
        );
        info!(
            "--- test_full_linoss_model_forward_pass (Varied Dimensions) completed successfully ---"
        );
    }

    fn run_model_test_with_params<B: Backend<FloatElem = f32>>(
        d_input: usize,
        d_model: usize,
        d_output: usize,
        n_layers: usize,
        batch_size: usize,
        seq_len: usize,
        device: &B::Device,
    ) {
        let block_conf = LinossBlockConfig {
            d_state_m: d_model / 2, 
            d_ff: d_model * 2,      
            delta_t: 0.1,           
            init_std: 0.02,
            enable_d_feedthrough: true,
        };

        let model_config = FullLinossModelConfig {
            d_input,
            d_model,
            d_output,
            n_layers,
            linoss_block_config: block_conf,
        };
        info!(
            "FullLinossModel Test: Initializing model with config: {:?}",
            model_config
        );

        let model: FullLinossModel<B> = model_config.init(device);

        let input_tensor = Tensor::<B, 3>::random(
            [batch_size, seq_len, d_input],
            Distribution::Uniform(0.0, 1.0),
            device,
        );

        debug!(
            "FullLinossModel Test: Input tensor shape: {:?}",
            input_tensor.dims()
        );

        let output_tensor = model.forward(input_tensor);
        debug!(
            "FullLinossModel Test: Output tensor shape: {:?}",
            output_tensor.dims()
        );

        let expected_dims = [batch_size, seq_len, d_output];
        assert_eq!(
            output_tensor.dims(),
            expected_dims,
            "Output tensor shape mismatch"
        );

        info!(
            "FullLinossModel Test: Forward pass completed for d_input={}, d_model={}, d_output={}, n_layers={}",
            d_input, d_model, d_output, n_layers
        );
    }
}