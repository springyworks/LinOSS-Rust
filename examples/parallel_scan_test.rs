// examples/parallel_scan_test.rs
// Test example for the parallel scan implementation in LinOSS

/*
use burn::prelude::*;
use burn::tensor::TensorData;

// Corrected path to LinossLayer and LinossLayerConfig
// Assuming these are now top-level under linoss
use linoss_rust::linoss::{
    layer::{LinossLayer, LinossLayerConfig},
    // parallel_scan::{self, RecurrencePair}, // Import module directly and RecurrencePair // This path is likely incorrect now
};

// --- Backend Selection (Simplified for example) ---
// Default to NdArray for this example
type MyBackend = burn::backend::NdArray<f32>;
type MyDevice = burn::backend::ndarray::NdArrayDevice;
// --- End Backend Selection ---

fn main() {
    let device = MyDevice::default();

    // Configuration
    let d_input = 2;
    let d_state_m = 4;
    let d_output_q = 2;
    let delta_t = 0.1;
    let batch_size = 1;
    let seq_len = 3;

    let layer_config = LinossLayerConfig {
        d_input_p: d_input,
        d_state_m,
        d_output_q,
        delta_t,
        init_std: 0.02,
        has_d_term: true,
    };
    let layer: LinossLayer<MyBackend> = layer_config.init(&device);

    // Create dummy input sequence: (batch, seq_len, d_input)
    let input_data_vec: Vec<f32> = (0..(batch_size * seq_len * d_input)).map(|i| i as f32).collect();
    let input_sequence = Tensor::<MyBackend, 3>::from_data(
        TensorData::new(input_data_vec, [batch_size, seq_len, d_input]),
        &device,
    );

    // Create initial states: (batch, d_state_m) and (batch, d_output_q)
    let initial_y_state = Tensor::<MyBackend, 2>::zeros([batch_size, d_state_m], &device);
    let initial_z_state = Tensor::<MyBackend, 2>::zeros([batch_size, d_output_q], &device);

    println!("Running Sequential Scan (Reference):");
    let ((final_y_seq, final_z_seq), output_seq_sequential) = layer.forward(
        input_sequence.clone(),
        initial_y_state.clone(),
        initial_z_state.clone(),
    );
    println!("  Output (Sequential):\n{:?}", output_seq_sequential.to_data());
    println!("  Final Y State (Sequential):\n{:?}", final_y_seq.to_data());
    println!("  Final Z State (Sequential):\n{:?}", final_z_seq.to_data());

    // --- Parallel Scan Section (Commented out due to refactoring) ---
    /*
    println!("\nRunning Parallel Scan:");

    // Prepare inputs for parallel_scan_linoss
    // This requires adapting the layer's parameters (A, B, C, D matrices)
    // and the input sequence into the format expected by parallel_scan_linoss.

    // Example: Extracting A_matrix (delta_A_expanded)
    // This is a simplification. The actual matrices might need to be cloned or accessed differently.
    // let delta_a_expanded = layer.delta_a_expanded.clone(); // Assuming public access or a getter

    // The `recurrences` would be constructed from the input sequence and B, D matrices.
    // The `initial_state` for parallel scan is typically just the hidden state 'h' (or 'y_state' here).

    // This part needs significant adaptation based on the current parallel_scan API.
    // The `parallel_scan_linoss` function and `RecurrencePair` would need to be correctly used.

    // let (output_parallel, final_hidden_state_parallel) = parallel_scan::parallel_scan_linoss(
    //     delta_a_expanded, // This is likely not the only matrix needed
    //     // ... other matrices (B, C, D related parts) ...
    //     input_sequence_adapted_for_parallel, // Input needs to be shaped into RecurrencePairs
    //     initial_y_state.clone(), // Or a relevant part of it
    // );

    // println!("  Output (Parallel):\n{:?}", output_parallel.to_data());
    // println!("  Final Hidden State (Parallel):\n{:?}", final_hidden_state_parallel.to_data());

    // Add assertions to compare sequential and parallel results if possible
    */
    println!("\nParallel scan example needs to be updated due to API changes in parallel_scan module.");
}
*/
fn main() {
    println!("parallel_scan_test.rs is currently commented out due to significant API changes in the parallel_scan module. It needs to be updated.");
}
