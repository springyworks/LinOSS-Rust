// src/linoss/block.rs
// Placeholder for LinOSS Block components

use burn::{
    module::Module,
    nn::{Linear, LinearConfig, LayerNorm, LayerNormConfig, Initializer}, 
    tensor::{
        backend::Backend,
        activation::silu, // Corrected: import silu directly
        Tensor,
    },
    config::Config,
};
use log::debug;
use crate::linoss::LinossOutput; // Added import for LinossOutput

use super::layer::{LinossLayer, LinossLayerConfig};

#[derive(Config, Debug)]
pub struct LinossBlockConfig { // Renamed from LinOSSBlockConfig
    pub d_state_m: usize,
    pub d_ff: usize,
    pub delta_t: f32,
    #[config(default = 0.02)]
    pub init_std: f64,
    #[config(default = true)]
    pub enable_d_feedthrough: bool, // Renamed from enable_D_feedthrough
}

impl LinossBlockConfig {
    pub fn init<B: Backend>(&self, d_model: usize, device: &B::Device) -> LinossBlock<B> { // Renamed return type
        // Initialize LinossLayerConfig by directly setting its fields
        let layer_config = LinossLayerConfig {
            d_state_m: self.d_state_m,
            d_input_p: d_model, // Assuming d_model is the input dimension for the layer within the block
            d_output_q: d_model, // Assuming d_model is also the output dimension for the layer
            delta_t: self.delta_t,
            init_std: self.init_std,
            enable_d_feedthrough: self.enable_d_feedthrough,
        };

        let layer = layer_config.init(device);

        let glu_linear_a_config = LinearConfig::new(d_model, self.d_ff)
            .with_initializer(Initializer::Normal { mean: 0.0, std: self.init_std });
        let glu_linear_b_config = LinearConfig::new(d_model, self.d_ff)
            .with_initializer(Initializer::Normal { mean: 0.0, std: self.init_std });
        let output_linear_config = LinearConfig::new(self.d_ff, d_model)
            .with_initializer(Initializer::Normal { mean: 0.0, std: self.init_std });
        
        let norm_config = LayerNormConfig::new(d_model);

        LinossBlock { // Renamed struct
            d_state_m: self.d_state_m,
            layer,
            glu_linear_a: glu_linear_a_config.init(device),
            glu_linear_b: glu_linear_b_config.init(device),
            output_linear: output_linear_config.init(device),
            norm: norm_config.init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct LinossBlock<B: Backend> { // Renamed from LinOSSBlock
    d_state_m: usize,
    layer: LinossLayer<B>,
    glu_linear_a: Linear<B>,
    glu_linear_b: Linear<B>,
    output_linear: Linear<B>,
    norm: LayerNorm<B>,
}

impl<B: Backend> LinossBlock<B> { // Renamed struct
    // This method processes a sequence and returns the output sequence and final states.
    // It can be used by the new `forward` method or directly if states are needed.
    pub fn forward_recurrent(
        &self,
        input: Tensor<B, 3>, 
        y_prev_block_state: Tensor<B, 2>,
        z_prev_block_state: Tensor<B, 2>,
    ) -> (Tensor<B, 3>, (Tensor<B, 2>, Tensor<B, 2>))
    where
        B::FloatElem: From<f32> + std::ops::Mul<Output = B::FloatElem> + Copy,
    {
        let [batch_size, seq_len, d_model_dim] = input.dims();
        let mut outputs = Vec::with_capacity(seq_len);

        let mut current_y_state = y_prev_block_state;
        let mut current_z_state = z_prev_block_state;

        for i in 0..seq_len {
            let u_t = input.clone().slice([0..batch_size, i..i + 1, 0..d_model_dim]).squeeze(1); 

            // The LinossLayer now takes an Option for the hidden state.
            // The state tuple (y_state, z_state) is specific to the older LinOSS model or a different interpretation.
            // For LinossLayer (IM), the primary recurrent state is y_state.
            // We pass y_state as the hidden_state_option.
            // The z_state is not directly part of LinossLayer's recurrent loop in this simplified IM version.
            // It was used in the oscillator formulation for the second state variable.
            // If LinossBlock needs to manage a z_state for other purposes (e.g. specific block architectures),
            // it should handle it externally to LinossLayer's forward_step.

            let linoss_layer_output: LinossOutput<B, 2> = self.layer.forward_step(
                u_t.clone(), 
                Some(current_y_state.clone()) // Pass y_state as the optional hidden state
            );

            let x_t_linoss = linoss_layer_output.output;
            // The hidden_state from LinossLayer is the next y_state.
            let next_y_state = linoss_layer_output.hidden_state.unwrap_or_else(|| 
                Tensor::zeros_like(&current_y_state) // Should always be Some if layer logic is correct
            );
            // next_z_state is not returned by this LinossLayer. If needed, it must be computed differently.
            // For now, let's assume z_state is not modified or needed further in this specific block step.
            // If z_state were part of a larger recurrent structure managed by the block, it would be updated here.
            // For a simple block just wrapping a layer, next_z_state might just be the old z_state or zeros.
            // Let's pass through current_z_state for now if the API expects it, though it's not used by LinossLayer.
            let next_z_state = current_z_state; // Placeholder, as LinossLayer doesn't return it.

            // Apply GLU to the output of the Linoss layer (x_t_linoss)
            let glu_a_out = self.glu_linear_a.forward(x_t_linoss.clone()); // Input d_model, Output d_ff
            let glu_b_out = self.glu_linear_b.forward(x_t_linoss.clone()); // Input d_model, Output d_ff
            let x_after_glu = glu_a_out * silu(glu_b_out); // Shape [batch_size, d_ff]
            debug!("LinossBlock: After GLU (x_after_glu), shape: {:?}", x_after_glu.dims());

            // Projection back to d_model
            let x_projected = self.output_linear.forward(x_after_glu); // Input d_ff, Output d_model. Shape [batch_size, d_model]
            debug!("LinossBlock: After OutputLinear (x_projected), shape: {:?}", x_projected.dims());

            // Residual connection and Layer Normalization
            let x_residual = u_t.clone() + x_projected; // Element-wise sum
            let x_t_block_output = self.norm.forward(x_residual); // Apply LayerNorm. Shape [batch_size, d_model]
            debug!("LinossBlock: After Norm (x_t_block_output), shape: {:?}", x_t_block_output.dims());


            outputs.push(x_t_block_output.unsqueeze_dim(1)); // Add sequence dim back
            current_y_state = next_y_state;
            current_z_state = next_z_state;
        }

        let stacked_outputs = Tensor::cat(outputs, 1);
        (stacked_outputs, (current_y_state, current_z_state))
    }

    // New forward method for use in FullLinossModel
    // It processes the entire input sequence and returns the output sequence.
    // Hidden states are initialized and managed internally for this block's pass.
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3>
    where
        B::FloatElem: From<f32> + std::ops::Mul<Output = B::FloatElem> + Copy,
    {
        let [batch_size, _seq_len, _d_model] = input.dims();
        let device = input.device(); // Get device from input tensor

        // Initialize hidden states for this block's pass
        let (initial_y_state, initial_z_state) = self.init_hidden_states(batch_size, &device);

        // Call the recurrent logic, which handles the sequence internally
        let (output_sequence, _final_states) = 
            self.forward_recurrent(input, initial_y_state, initial_z_state);
        
        output_sequence // Return only the output sequence
    }

    /// Initializes hidden states (y and z) for a LinossBlock.
    ///
    /// # Arguments
    /// * `batch_size`: The batch size for the states.
    /// * `device`: The device on which to create the tensors.
    ///
    /// # Returns
    /// A tuple `(initial_y_state, initial_z_state)`, both tensors of shape `[batch_size, d_state_m]`.
    pub fn init_hidden_states(&self, batch_size: usize, device: &B::Device) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let initial_y_state = Tensor::zeros([batch_size, self.d_state_m], device);
        let initial_z_state = Tensor::zeros([batch_size, self.d_state_m], device);
        (initial_y_state, initial_z_state)
    }

    pub fn forward_sequential_scan(
        &self,
        input: Tensor<B, 3>, // [batch_size, seq_len, d_model]
        y_prev_block_state: Tensor<B, 2>, // [batch_size, d_state_m]
        z_prev_block_state: Tensor<B, 2>, // [batch_size, d_state_m]
    ) -> (Tensor<B, 3>, (Tensor<B, 2>, Tensor<B, 2>))
    where
        B::FloatElem: From<f32> + std::ops::Mul<Output = B::FloatElem> + Copy, // Uses B from the impl block
    {
        let [batch_size, seq_len, d_model_dim] = input.dims();
        let mut outputs = Vec::with_capacity(seq_len);

        let mut current_y_state = y_prev_block_state;
        let mut current_z_state = z_prev_block_state;

        for i in 0..seq_len {
            let u_t = input.clone().slice([0..batch_size, i..i + 1, 0..d_model_dim]).squeeze(1); 

            // The LinossLayer now takes an Option for the hidden state.
            // The state tuple (y_state, z_state) is specific to the older LinOSS model or a different interpretation.
            // For LinossLayer (IM), the primary recurrent state is y_state.
            // We pass y_state as the hidden_state_option.
            // The z_state is not directly part of LinossLayer's recurrent loop in this simplified IM version.
            // It was used in the oscillator formulation for the second state variable.
            // If LinossBlock needs to manage a z_state for other purposes (e.g. specific block architectures),
            // it should handle it externally to LinossLayer's forward_step.

            let linoss_layer_output: LinossOutput<B, 2> = self.layer.forward_step(
                u_t.clone(), 
                Some(current_y_state.clone()) // Pass y_state as the optional hidden state
            );

            let x_t_linoss = linoss_layer_output.output;
            let next_y_state = linoss_layer_output.hidden_state.unwrap_or_else(|| 
                Tensor::zeros_like(&current_y_state)
            );
            let next_z_state = current_z_state; // Placeholder

            // Apply GLU to the output of the Linoss layer (x_t_linoss)
            let glu_a_out = self.glu_linear_a.forward(x_t_linoss.clone()); // Input d_model, Output d_ff
            let glu_b_out = self.glu_linear_b.forward(x_t_linoss.clone()); // Input d_model, Output d_ff
            let x_after_glu = glu_a_out * silu(glu_b_out); // Shape [batch_size, d_ff]
            debug!("LinossBlock: After GLU (x_after_glu), shape: {:?}", x_after_glu.dims());

            // Projection back to d_model
            let x_projected = self.output_linear.forward(x_after_glu); // Input d_ff, Output d_model. Shape [batch_size, d_model]
            debug!("LinossBlock: After OutputLinear (x_projected), shape: {:?}", x_projected.dims());

            // Residual connection and Layer Normalization
            let x_residual = u_t.clone() + x_projected; // Element-wise sum
            let x_t_block_output = self.norm.forward(x_residual); // Apply LayerNorm. Shape [batch_size, d_model]
            debug!("LinossBlock: After Norm (x_t_block_output), shape: {:?}", x_t_block_output.dims());

            outputs.push(x_t_block_output.unsqueeze_dim(1)); // Add sequence dim back
            current_y_state = next_y_state;
            current_z_state = next_z_state;
        }

        let stacked_outputs = Tensor::cat(outputs, 1);
        (stacked_outputs, (current_y_state, current_z_state))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{ndarray::NdArray, Autodiff};
    use burn::tensor::Distribution;

    type TestBackend = Autodiff<NdArray>; 

    #[test]
    fn test_linoss_block_config_init() {
        let device = <TestBackend as Backend>::Device::default();

        let d_model_block = 128;
        let d_state_m_layer = 64; 
        let d_ff_block = 256;
        let delta_t_block = 0.05;

        let block_config = LinossBlockConfig { // Renamed
            d_state_m: d_state_m_layer,
            d_ff: d_ff_block,
            delta_t: delta_t_block,
            init_std: 0.02, 
            enable_d_feedthrough: true, // Use the new snake_case field
        };
        
        let block: LinossBlock<TestBackend> = block_config.init(d_model_block, &device); // Renamed

        assert_eq!(block.layer.d_state_m(), d_state_m_layer);
        assert_eq!(block.d_state_m, d_state_m_layer);
        assert_eq!(block.layer.d_input_p(), d_model_block);
        assert_eq!(block.layer.d_output_q(), d_model_block); 
        assert_eq!(block.layer.delta_t(), delta_t_block);

        let _ = &block.glu_linear_a.weight;
        let _ = &block.glu_linear_b.weight;
        let _ = &block.output_linear.weight;
        let _ = &block.norm.gamma;

        debug!("LinossBlockConfig init test completed. Layer input_p: {}, output_q: {}", block.layer.d_input_p(), block.layer.d_output_q());
    }

    #[test]
    fn test_linoss_block_forward_recurrent() { // This tests the original recurrent method
        let device = <TestBackend as Backend>::Device::default();

        let d_model = 64;
        let d_state_m = 32;
        let d_ff = 128;
        let delta_t = 0.1;

        let config = LinossBlockConfig { // Renamed
            d_state_m,
            d_ff,
            delta_t,
            init_std: 0.02,
            enable_d_feedthrough: true,
        };
        
        let block: LinossBlock<TestBackend> = config.init(d_model, &device); // Renamed

        let batch_size = 2;
        let seq_len = 5;

        let (initial_y, initial_z) = block.init_hidden_states(batch_size, &device);

        let input_tensor = Tensor::<TestBackend, 3>::random(
            [batch_size, seq_len, d_model],
            Distribution::Uniform(0.0, 1.0),
            &device,
        );

        debug!("Test: Calling LinossBlock forward_recurrent pass...");
        let (output, (final_y, final_z)) = block.forward_recurrent(input_tensor.clone(), initial_y.clone(), initial_z.clone());
        debug!("Test: LinossBlock forward_recurrent pass completed.");

        assert_eq!(output.dims(), [batch_size, seq_len, d_model]);
        assert_eq!(final_y.dims(), [batch_size, d_state_m]);
        assert_eq!(final_z.dims(), [batch_size, d_state_m]);
    }

    #[test]
    fn test_linoss_block_forward() { // New test for the new forward method
        let device = <TestBackend as Backend>::Device::default();

        let d_model = 64;
        let d_state_m = 32; // d_state_m for the layer within the block
        let d_ff = 128;
        let delta_t = 0.1;

        let config = LinossBlockConfig { // Renamed
            d_state_m,
            d_ff,
            delta_t,
            init_std: 0.02,
            enable_d_feedthrough: true,
        };
        
        let block: LinossBlock<TestBackend> = config.init(d_model, &device); // Renamed

        let batch_size = 2;
        let seq_len = 5;

        let input_tensor = Tensor::<TestBackend, 3>::random(
            [batch_size, seq_len, d_model],
            Distribution::Uniform(0.0, 1.0),
            &device,
        );

        debug!("Test: Calling LinossBlock forward pass...");
        let output = block.forward(input_tensor.clone());
        debug!("Test: LinossBlock forward pass completed.");

        assert_eq!(output.dims(), [batch_size, seq_len, d_model]);
        // We don't check internal states here as they are managed by the forward method itself.
    }
}
