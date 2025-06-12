// src/data/synthetic.rs
// Synthetic data generation for Burn pipeline prototyping

use burn::tensor::{backend::Backend, Tensor, TensorData};

/// Generate a small synthetic sine wave dataset for testing
pub fn generate_sine_wave<B: Backend>(batch: usize, seq_len: usize, input_dim: usize, device: &B::Device) -> Tensor<B, 3> {
    let mut data = Vec::with_capacity(batch * seq_len * input_dim);
    for b in 0..batch {
        for t in 0..seq_len {
            for i in 0..input_dim {
                let val = ((t as f32) * 0.2 + (i as f32) * 0.5 + (b as f32)).sin();
                data.push(val);
            }
        }
    }
    let tensor_data = TensorData::new(data, [batch, seq_len, input_dim]);
    Tensor::<B, 3>::from_data(tensor_data, device)
}
