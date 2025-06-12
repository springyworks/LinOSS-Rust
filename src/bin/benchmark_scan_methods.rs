// src/bin/benchmark_scan_methods.rs
// Benchmark different scan methods for LinOSS recurrence

use std::time::Instant;

use burn::{
    tensor::{
        // Corrected: Backend trait is in prelude
        Tensor, TensorData, Shape,
    },
    prelude::Backend, // Import Backend trait from prelude
};

// Conditionally import backend specifics
#[cfg(feature = "ndarray_backend")]
use burn::backend::{ndarray::{NdArray, NdArrayDevice}}; // Removed unused Autodiff as AutodiffNdArray

// Corrected WGPU imports based on wgpu_minimal_test.rs
#[cfg(feature = "wgpu_backend")]
use burn::backend::{
    wgpu::{Wgpu, WgpuDevice}, // Wgpu, WgpuDevice from burn::backend::wgpu
    // Removed unused Autodiff as AutodiffWgpu
};


use linoss_rust::linoss::{
    layer::{LinossLayer, LinossLayerConfig}, // Updated to LinossLayer and LinossLayerConfig
};
use rand::{Rng, SeedableRng, rngs::StdRng}; // Added Rng

fn main() {
    // Configuration
    let config = LinossLayerConfig { // Updated to LinossLayerConfig
        d_input_p: 10, // Renamed from d_input
        d_state_m: 20,
        d_output_q: 5,
        delta_t: 0.1,
        init_std: 0.02,
        enable_d_feedthrough: true, // Corrected field name
    };

    let batch_size = 4;
    let seq_len = 50;

    let mut backend_run_count = 0;

    #[cfg(feature = "ndarray_backend")]
    {
        println!("--- Running benchmarks with NdArray backend (CPU) ---");
        let device = NdArrayDevice::default();
        // NdArray backend type: NdArray<FloatElement>
        type BackendNdArray = NdArray<f32>;
        // Autodiff for NdArray was removed as it was unused.
        // type AutodiffBackendNdArray = AutodiffNdArray<BackendNdArray>;

        run_benchmark_for_backend::<BackendNdArray>(&config, batch_size, seq_len, &device);
        backend_run_count += 1;
        println!("----------------------------------------------------");
    }

    #[cfg(feature = "wgpu_backend")]
    {
        println!("\n--- Running benchmarks with WGPU backend ---");
        let device = WgpuDevice::default(); 
        
        // WGPU backend type: Wgpu<FloatType, IntType>
        // Defaults to Wgpu<f32, i32> if not specified, assuming AutoGraphicsApi
        type BackendWgpu = Wgpu<f32, i32>;
        // Autodiff for WGPU was removed as it was unused.
        // type AutodiffBackendWgpu = AutodiffWgpu<BackendWgpu>;


        println!("Using WGPU Device: {:?}", device);
        run_benchmark_for_backend::<BackendWgpu>(&config, batch_size, seq_len, &device);
        backend_run_count += 1;
        println!("--------------------------------------------------");
    }

    if backend_run_count == 0 {
        println!("No backend feature (ndarray_backend or wgpu_backend) enabled. Skipping benchmarks.");
        println!("Please run with --features ndarray_backend or --features wgpu_backend (or both)");
    }
}

fn run_benchmark_for_backend<B: Backend>(
    config: &LinossLayerConfig, // Updated to LinossLayerConfig
    batch_size: usize,
    seq_len: usize,
    device: &B::Device,
)
where // Add these trait bounds
    B::FloatElem: From<f32> + std::ops::Mul<Output = B::FloatElem> + Copy,
{
    println!("Benchmarking for Backend: {:?}", std::any::type_name::<B>());
    let layer = config.init::<B>(device); // Specify type B for init
    let input_sequence = generate_input_sequence::<B>(config, batch_size, seq_len, device);

    benchmark_forward_methods::<B>(&layer, &input_sequence, device);
    // benchmark_scan_algorithms function removed as it's no longer viable
}

fn generate_input_sequence<B: Backend>(
    config: &LinossLayerConfig, // Updated to LinossLayerConfig
    batch_size: usize,
    seq_len: usize,
    device: &B::Device,
) -> Tensor<B, 3> {
    let input_dim = config.d_input_p; // Updated to d_input_p
    let mut rng = StdRng::from_seed(Default::default());
    
    // Initialize with random data first
    let mut initial_input_data = Vec::with_capacity(batch_size * seq_len * input_dim);
    for _ in 0..(batch_size * seq_len * input_dim) {
        initial_input_data.push(rng.gen::<f32>() * 2.0 - 1.0); // Using rng.gen() instead of rng.random()
    }
    let mut input_tensor = Tensor::<B, 3>::from_data(
        TensorData::new(initial_input_data, Shape::new([batch_size, seq_len, input_dim])),
        device
    );

    // Optional: Add structure to input like in compare_scan_methods
    for i in 0..seq_len {
        let time_step = i as f32;
        let sine_wave = (time_step * 0.5).sin() * 0.5;
        let mut current_slice_data_vec = vec![0.0; batch_size * input_dim];
        for b_idx in 0..batch_size {
            current_slice_data_vec[b_idx * input_dim] = sine_wave;
            for d_idx in 1..input_dim {
                current_slice_data_vec[b_idx * input_dim + d_idx] = (time_step * 0.2 + d_idx as f32 * 0.1).cos() * 0.3;
            }
        }
        let slice_data_tensor = Tensor::<B,2>::from_data(TensorData::new(current_slice_data_vec, Shape::new([batch_size, input_dim])), device);
        
        input_tensor = input_tensor.slice_assign(
            [0..batch_size, i..(i+1), 0..input_dim], 
            slice_data_tensor.unsqueeze_dim::<3>(1) 
        );
    }
    input_tensor
}


fn benchmark_forward_methods<B: Backend>(
    layer: &LinossLayer<B>, // Updated to LinossLayer
    input_sequence: &Tensor<B, 3>,
    device: &B::Device,
)
where
    B::FloatElem: From<f32> + std::ops::Mul<Output = B::FloatElem> + Copy, // Add trait bounds
{
    let [batch_size, seq_len, _] = input_sequence.dims();
    
    // Initialize hidden state for the layer (only y_state is managed by current LinossLayer)
    let mut y_state = Tensor::<B, 2>::zeros([batch_size, layer.d_state_m()], device);
    // z_state was removed as it's not used by the current LinossLayer::forward_step
    
    let mut outputs = Vec::with_capacity(seq_len);

    // Benchmark manual sequential scan using forward_step
    let start_seq = Instant::now();
    for i in 0..seq_len {
        let u_t = input_sequence.clone().slice([0..batch_size, i..i + 1, 0..layer.d_input_p()]).squeeze(1);
        // Pass current y_state as Option, LinossLayer::forward_step returns LinossOutput { output, hidden_state: Some(next_y_state) }
        let result = layer.forward_step(u_t, Some(y_state.clone()));
        let x_t = result.output;
        outputs.push(x_t.unsqueeze_dim(1)); // Add sequence dim back
        if let Some(next_y_state) = result.hidden_state {
            y_state = next_y_state;
        } else {
            // Should not happen if LinossLayer always returns a state
            panic!("LinossLayer::forward_step did not return a hidden state");
        }
    }
    let _output_sequence: Tensor<B, 3> = Tensor::cat(outputs, 1); 
    let duration_seq = start_seq.elapsed();
    println!("  LinossLayer.forward_step (manual sequential scan): {:?}", duration_seq);

    // Removed benchmarks for forward_parallel_scan, forward_tree_scan, and forward_work_efficient_scan
    // as these methods are no longer part of LinossLayer\'s public API for direct sequence processing.
}
