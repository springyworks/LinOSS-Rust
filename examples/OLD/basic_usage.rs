// /examples/basic_usage.rs

use linoss_rust::linoss::{
    layer::{LinossLayer, LinossLayerConfig}, // Fix import to use layer
};
use burn::tensor::{Tensor, TensorData, Shape, backend::Backend};
use log::{info, debug};

// Define a backend (e.g., NdArray for CPU, or Wgpu for GPU if configured)
type AppBackend = burn::backend::ndarray::NdArray<f32>;
// To use a GPU backend like WGPU, you might do something like:
// type AppBackend = burn::backend::wgpu::Wgpu<burn::backend::wgpu::AutoGraphicsApi, f32, i32>;

// Helper function to initialize the logger
fn init_logger() {
    // You can customize the logger further, e.g., by setting a default log level
    // if RUST_LOG is not set.
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
}

fn main() {
    init_logger();
    info!("--- Running Linoss Model: Basic Usage Example ---");

    let device: <AppBackend as Backend>::Device = Default::default();

    // Configuration
    let d_input = 2;
    let d_output = 2;
    let d_state = 4;
    let dt = 0.1;
    let batch_size = 1;

    let config = LinossLayerConfig {
        d_state_m: d_state,
        d_input_p: d_input,
        d_output_q: d_output,
        delta_t: dt,
        init_std: 0.02,
        enable_d_feedthrough: false, // Explicitly set, or rely on default
    };
    let model: LinossLayer<AppBackend> = config.init(&device);

    // Create a dummy input tensor: [batch_size, d_input]
    let input_elem_count = batch_size * d_input;
    let input_vec: Vec<f32> = (0..input_elem_count).map(|x| (x % 10) as f32 * 0.1 + 0.05).collect();
    let input_data = TensorData::new(input_vec.clone(), Shape::new([batch_size, d_input]));
    let input_tensor: Tensor<AppBackend, 2> = Tensor::from_data(input_data.convert::<f32>(), &device);
    
    info!("Input tensor created with shape: {:?}", input_tensor.dims());
    debug!("Sample input data (first {} elements): {:?}", d_input * 2.min(input_vec.len()), &input_vec[..d_input * 2.min(input_vec.len())]);

    // Perform a forward pass
    info!("Performing forward pass...");
    let output = model.forward_step(input_tensor.clone(), None);
    info!("Forward pass completed.");

    // Process and display output
    let output_tensor = output.output;
    let output_shape = output_tensor.dims();
    info!("Output tensor shape: {:?}", output_shape);

    let output_data: TensorData = output_tensor.into_data();
    let output_vec: Vec<f32> = output_data.convert::<f32>().into_vec().unwrap();
    
    info!("Output tensor data (first {} elements):", (d_output * 2).min(output_vec.len()));
    for (i, val) in output_vec.iter().take((d_output * 2).min(output_vec.len())).enumerate() {
        info!("Output[{}]: {:.4}", i, val);
    }

    // Example: Check if output is not all zeros (simple sanity check)
    let mut all_zeros = true;
    let mut has_nan_or_inf = false;
    for &val in output_vec.iter() {
        if val != 0.0 {
            all_zeros = false;
        }
        if val.is_nan() || val.is_infinite() {
            has_nan_or_inf = true;
        }
    }

    if all_zeros {
        log::warn!("Output tensor is all zeros. This might indicate an issue or specific model state.");
    } else {
        info!("Output tensor contains non-zero values.");
    }

    if has_nan_or_inf {
        log::error!("Output tensor contains NaN or Inf values! This indicates a numerical stability issue.");
    } else {
        info!("Output tensor does not contain NaN or Inf values.");
    }

    info!("--- Linoss Model: Basic Usage Example Finished ---");
}
