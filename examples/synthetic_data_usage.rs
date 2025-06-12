// examples/synthetic_data_usage.rs
// Example: Using synthetic sine wave data with Burn and LinOSS

use burn::backend::ndarray::NdArray;
use linoss_rust::data::synthetic::generate_sine_wave;

fn main() {
    let batch = 2;
    let seq_len = 8;
    let input_dim = 3;
    let device = Default::default();
    let data = generate_sine_wave::<NdArray<f32>>(batch, seq_len, input_dim, &device);
    let data_vec: Vec<f32> = data.clone().into_data().into_vec().unwrap();
    println!("Synthetic sine wave data shape: {:?}", data.shape());
    println!("Total elements: {}", data_vec.len());
    println!("First batch, first sequence (all input_dim): {:?}", &data_vec[0..input_dim]);
    println!("First 10 values: {:?}", &data_vec[0..10.min(data_vec.len())]);
}
