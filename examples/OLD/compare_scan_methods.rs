// examples/compare_scan_methods.rs
// Compare different scan methods for LinOSS recurrence

use std::time::Instant;

use burn::tensor::backend::Backend;
use burn::tensor::{Distribution, Tensor};
// use burn::nn::LayerNormConfig; // Not directly used here, but LinossBlockConfig uses it internally
use linoss_rust::linoss::{
    LinossLayer, LinossLayerConfig, // LinossLayer and its config
    model::{FullLinossModel, FullLinossModelConfig}, // FullLinossModel and its config from model.rs
    block::{LinossBlockConfig}, // LinossBlockConfig from block.rs
};
use std::io::{stdout, Write}; // Added for flushing stdout

type CustomNdArrayBackend = burn::backend::NdArray<f32>;
type CustomNdArrayDevice = burn::backend::ndarray::NdArrayDevice;

#[cfg(feature = "wgpu_backend")]
type CustomWgpuBackend = burn::backend::Wgpu<f32, i32>;
#[cfg(feature = "wgpu_backend")]
type CustomWgpuDevice = burn::backend::wgpu::WgpuDevice;

const PROGRESS_CHARS: [char; 4] = ['-', '\\', '|', '/'];

fn main() {
    // Example entry point
    run_benchmark::<CustomNdArrayBackend>(CustomNdArrayDevice::default(), "NdArray Backend");

    #[cfg(feature = "wgpu_backend")]
    {
        let device_wgpu = CustomWgpuDevice::default();
        run_benchmark::<CustomWgpuBackend>(device_wgpu, "WGPU Backend");
    }
}

fn run_benchmark<B: Backend>(device: B::Device, backend_name: &str)
where
    B::FloatElem: burn::tensor::Element
        + rand::distr::uniform::SampleUniform // For Tensor::random
        + From<f32>
        + std::ops::Mul<Output = B::FloatElem>
        + Copy
        + num_traits::FloatConst, // Add FloatConst trait bound
    B::IntElem: burn::tensor::Element,
{
    println!("\n--- Running Benchmark for {} ---", backend_name);

    let hidden_size = 32; // Reasonable size for benchmark demo
    let input_size = 1;
    let output_size = hidden_size;
    let num_model_layers = 2;
    let delta_t_model = 0.1;

    let linoss_block_config = LinossBlockConfig {
        d_state_m: hidden_size, 
        d_ff: hidden_size * 2, // Example: feed-forward dimension, often a multiple of d_model/hidden_size
        delta_t: delta_t_model,
        init_std: 0.02, 
        enable_d_feedthrough: true, 
    };

    let model_config = FullLinossModelConfig {
        d_input: input_size,
        d_model: hidden_size,
        d_output: output_size, // d_output is used by the decoder in FullLinossModel
        n_layers: num_model_layers, 
        linoss_block_config, 
    };

    let model: FullLinossModel<B> = model_config.init(&device); // Use config.init()

    let seq_len = 50;     // Reasonable sequence length for demo
    let batch_size = 4;   // Reasonable batch size for demo

    let model_input_tensor: Tensor<B, 3> = Tensor::random(
        [batch_size, seq_len, input_size], // Use array for shape
        Distribution::Default,
        &device,
    );

    let start_time_model = Instant::now();
    let output_model = model.forward(model_input_tensor.clone());
    let duration_model = start_time_model.elapsed();
    println!(
        "FullLinossModel forward pass output (first batch, first step, first val): {:?}",
        // Use .into_vec().unwrap()[0] for single value, or .as_slice().unwrap()[0]
        output_model.slice([0..1, 0..1, 0..1]).into_data().convert::<f32>().into_vec::<f32>().unwrap()[0]
    );
    println!("FullLinossModel forward pass duration: {:?}", duration_model);


    // --- Benchmarking a single LinossLayer ---
    let layer_d_input_p = input_size;
    let layer_d_state_m = hidden_size;
    let layer_d_output_q = hidden_size;
    let delta_t_layer = 0.05;

    let layer_config = LinossLayerConfig {
        d_input_p: layer_d_input_p, // Correct field name
        d_output_q: layer_d_output_q, // Correct field name
        d_state_m: layer_d_state_m, // Correct field name
        delta_t: delta_t_layer,
        enable_d_feedthrough: false, 
        init_std: 0.02,
    };
    let layer: LinossLayer<B> = layer_config.init(&device); // Use config.init(), remove mut

    let layer_input_tensor_seq: Tensor<B, 3> = Tensor::random(
        [batch_size, seq_len, layer_d_input_p], // Use array for shape
        Distribution::Default,
        &device,
    );
    
    let mut current_hidden_state: Option<Tensor<B, 2>> = None;

    let start_time_layer_seq = Instant::now();
    
    let mut output_tensors_list = Vec::with_capacity(seq_len);

    let spinner_chars = PROGRESS_CHARS;
    let mut spinner_idx = 0;

    println!(); // Add a newline before the loop starts
    for i in 0..seq_len {
        // Update and print spinner every iteration
        print!("  LinossLayer: processing step {}/{} {} \r", i, seq_len, spinner_chars[spinner_idx]);
        stdout().flush().unwrap(); // Ensure it prints immediately
        spinner_idx = (spinner_idx + 1) % spinner_chars.len();

        let u_t = layer_input_tensor_seq.clone().slice([0..batch_size, i..(i+1), 0..layer.d_input_p()]).squeeze(1); 
        let linoss_output = layer.forward_step(u_t, current_hidden_state.clone());
        output_tensors_list.push(linoss_output.output.unsqueeze_dim::<3>(1)); 
        current_hidden_state = linoss_output.hidden_state;
    }
    let output_layer_seq: Tensor<B, 3> = Tensor::cat(output_tensors_list, 1); 

    let duration_layer_seq = start_time_layer_seq.elapsed();
    println!("\r  LinossLayer manual forward_sequence completed.                                       "); // Clear spinner line
    println!(
        "LinossLayer manual forward_sequence output (first batch, first step, first val): {:?}",
        // Use .into_vec().unwrap()[0]
        output_layer_seq.slice([0..1, 0..1, 0..1]).into_data().convert::<f32>().into_vec::<f32>().unwrap()[0]
    );
    println!(
        "LinossLayer manual forward_sequence duration: {:?}",
        duration_layer_seq
    );
}
