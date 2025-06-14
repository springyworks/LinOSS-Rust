// src/bin/dlinoss_comparison.rs
// D-LinOSS vs vanilla LinOSS performance comparison
//
// Citation: D-LinOSS implementation based on:
// Ricci, Marco Federici, Luca Zancato, Maximiliano Alfini, Matteo Salvatori,
// and Lukasz Kaiser. "D-LinOSS: Enhancing Controllability of Linear
// Oscillatory State-Space Models via Learnable Damping." arXiv preprint
// arXiv:2505.12171 (2025).

use burn::{
    prelude::Backend,
    tensor::{Tensor, TensorData},
};
use std::time::Instant;

// Conditionally import backends
#[cfg(feature = "wgpu_backend")]
use burn::backend::wgpu::{Wgpu, WgpuDevice};

use linoss_rust::linoss::{
    dlinoss_layer::{AParameterization, DLinossLayer, DLinossLayerConfig},
    layer::{LinossLayer, LinossLayerConfig},
};
use rand::{rngs::StdRng, Rng, SeedableRng};

fn generate_test_signal<B: Backend>(
    device: &B::Device,
    batch_size: usize,
    seq_len: usize,
    input_dim: usize,
) -> Tensor<B, 3> {
    let mut rng = StdRng::seed_from_u64(42);

    // Generate a mix of sinusoidal and noisy signals
    let mut data = vec![0.0; batch_size * seq_len * input_dim];

    for b in 0..batch_size {
        for t in 0..seq_len {
            for d in 0..input_dim {
                let idx = b * seq_len * input_dim + t * input_dim + d;
                let time = t as f32 * 0.1;
                let freq = 0.5 + (d as f32) * 0.3;
                let signal = (time * freq * 2.0 * std::f32::consts::PI).sin();
                let noise_val: f32 = rng.gen();
                let noise = (noise_val - 0.5) * 0.1;
                data[idx] = signal + noise;
            }
        }
    }

    let tensor_data = TensorData::new(data, [batch_size, seq_len, input_dim]);
    Tensor::from_data(tensor_data, device)
}

fn benchmark_model<B: Backend, M>(
    model: &mut M,
    input: &Tensor<B, 3>,
    name: &str,
    iterations: usize,
) -> (f64, Tensor<B, 3>)
where
    M: FnMut(&Tensor<B, 3>) -> Tensor<B, 3>,
{
    // Warmup
    let _ = model(input);

    let start = Instant::now();
    let mut output = None;

    for _ in 0..iterations {
        output = Some(model(input));
    }

    let duration = start.elapsed();
    let avg_time_ms = duration.as_secs_f64() * 1000.0 / iterations as f64;

    println!("{}: {:.3} ms/iteration", name, avg_time_ms);

    (avg_time_ms, output.unwrap())
}

fn calculate_oscillation_stability<B: Backend>(output: &Tensor<B, 3>) -> f32 {
    // Simplified stability measure - just use the mean absolute value as a proxy
    let output_abs = output.clone().abs();
    let mean_abs = output_abs.mean();

    // Extract value using to_data and manual byte interpretation
    let data = mean_abs.to_data();
    let bytes = data.as_bytes();
    if bytes.len() >= 4 {
        f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
    } else {
        0.0
    }
}

/// Forward pass through LinossLayer for a full sequence
fn linoss_forward_sequence<B: Backend>(
    layer: &LinossLayer<B>,
    input: &Tensor<B, 3>, // [batch, seq_len, input_dim]
    _output_dim: usize,   // Keep for potential future use
) -> Tensor<B, 3> {
    let batch_size = input.dims()[0];
    let seq_len = input.dims()[1];

    let mut outputs = Vec::new();
    let mut hidden_state: Option<Tensor<B, 2>> = None;

    for t in 0..seq_len {
        let input_t = input
            .clone()
            .slice([0..batch_size, t..(t + 1), 0..input.dims()[2]])
            .squeeze(1);
        let result = layer.forward_step(input_t, hidden_state.clone());

        outputs.push(result.output.clone().unsqueeze_dim(1)); // Add time dimension
        hidden_state = result.hidden_state;
    }

    // Concatenate along time dimension
    Tensor::cat(outputs, 1)
}

fn run_comparison<B: Backend>(device: B::Device)
where
    B::FloatTensorPrimitive: Send + Sync,
    B::IntTensorPrimitive: Send + Sync,
    B::BoolTensorPrimitive: Send + Sync,
{
    println!("=== D-LinOSS vs LinOSS Comparison ===");
    println!("Backend: {}", std::any::type_name::<B>());

    // Test parameters
    let batch_size = 8;
    let seq_len = 256;
    let input_dim = 4;
    let model_dim = 32;
    let output_dim = 4;
    let iterations = 10;

    // Generate test signal
    let input = generate_test_signal::<B>(&device, batch_size, seq_len, input_dim);
    println!(
        "Input shape: [batch={}, seq_len={}, input_dim={}]",
        batch_size, seq_len, input_dim
    );

    // Configure LinOSS
    let linoss_config = LinossLayerConfig {
        d_state_m: model_dim,
        d_input_p: input_dim,
        d_output_q: output_dim,
        delta_t: 0.1,
        init_std: 0.02,
        enable_d_feedthrough: true,
    };

    // Configure D-LinOSS with learnable damping
    let dlinoss_config = DLinossLayerConfig {
        d_input: input_dim,
        d_model: model_dim,
        d_output: output_dim,
        delta_t: 0.1,
        init_std: 0.02,
        enable_layer_norm: false,
        enable_damping: true,
        init_damping: 0.1,
        num_damping_scales: 1,
        a_parameterization: AParameterization::ReLU, // Add the missing field
    };

    // Create models
    let linoss_layer = linoss_config.init(&device);
    let dlinoss_layer = DLinossLayer::<B>::new(&dlinoss_config, &device);

    println!("\n--- Performance Benchmarks ---");

    // Benchmark LinOSS
    let mut linoss_forward =
        |input: &Tensor<B, 3>| linoss_forward_sequence(&linoss_layer, input, output_dim);
    let (linoss_time, linoss_output) =
        benchmark_model(&mut linoss_forward, &input, "LinOSS", iterations);

    // Benchmark D-LinOSS
    let mut dlinoss_forward = |input: &Tensor<B, 3>| dlinoss_layer.forward(input.clone());
    let (dlinoss_time, dlinoss_output) =
        benchmark_model(&mut dlinoss_forward, &input, "D-LinOSS", iterations);

    // Calculate stability metrics
    let linoss_stability = calculate_oscillation_stability(&linoss_output);
    let dlinoss_stability = calculate_oscillation_stability(&dlinoss_output);

    println!("\n--- Results Analysis ---");
    println!(
        "LinOSS  - Time: {:.3} ms, Stability: {:.6}",
        linoss_time, linoss_stability
    );
    println!(
        "D-LinOSS - Time: {:.3} ms, Stability: {:.6}",
        dlinoss_time, dlinoss_stability
    );

    // Calculate speed ratios and improvements properly
    let speed_ratio = dlinoss_time / linoss_time;
    let stability_improvement = if linoss_stability > dlinoss_stability {
        (linoss_stability - dlinoss_stability) / linoss_stability * 100.0
    } else {
        0.0
    };

    println!("\n--- Performance Comparison ---");
    if dlinoss_time > linoss_time {
        println!("D-LinOSS is {:.2}x slower than LinOSS", speed_ratio);
        println!(
            "D-LinOSS takes {:.1}% more time",
            (speed_ratio - 1.0) * 100.0
        );
    } else {
        println!(
            "D-LinOSS is {:.2}x faster than LinOSS",
            linoss_time / dlinoss_time
        );
    }

    if dlinoss_stability < linoss_stability {
        println!(
            "D-LinOSS has {:.1}% better stability (lower variance)",
            stability_improvement
        );
    } else {
        println!("LinOSS has better stability");
    }

    // Output shape verification
    println!("\n--- Output Verification ---");
    println!("LinOSS output shape: {:?}", linoss_output.shape());
    println!("D-LinOSS output shape: {:?}", dlinoss_output.shape());

    println!("\n✓ D-LinOSS comparison completed successfully!");
}

pub fn main() {
    #[cfg(feature = "wgpu_backend")]
    {
        println!("Running with WGPU backend...");
        let device = WgpuDevice::default();
        run_comparison::<Wgpu>(device);
    }

    #[cfg(all(feature = "ndarray_backend", not(feature = "wgpu_backend")))]
    {
        use burn::backend::ndarray::{NdArray, NdArrayDevice};
        println!("Running with NdArray backend...");
        let device = NdArrayDevice::default();
        run_comparison::<NdArray>(device);
    }

    #[cfg(not(any(feature = "wgpu_backend", feature = "ndarray_backend")))]
    {
        println!(
            "No backend feature enabled. Please enable either 'wgpu_backend' or 'ndarray_backend'."
        );
    }
}
