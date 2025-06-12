use burn::backend::NdArray;
use burn::tensor::{Shape, Tensor, TensorData};
use linoss_rust::linoss::{
    model::FullLinossModel, model::FullLinossModelConfig, block::LinossBlockConfig, layer::LinossLayerConfig,
};
use linoss_rust::{LinossError, Vector}; // TensorData for creation, Shape for dimensions // Example backend

fn main() -> Result<(), LinossError> {
    println!("Linoss Model Simulation Start - SIMPLIFIED TEST");

    // Define dimensions (simplified for testing)
    let q_dim = 1; // Dimension of input to the model
    let model_dim = 2; // Internal model dimension (d_model, d_state_m)
    let output_dim = 1; // Dimension of output from the model
    let num_layers_l = 1; // Number of LinossBlock layers
    let sequence_len_n = 2; // Length of the input sequence

    println!(
        "Test Config: q_dim={}, model_dim={}, output_dim={}, num_layers={}, sequence_len={}",
        q_dim, model_dim, output_dim, num_layers_l, sequence_len_n
    );

    if num_layers_l == 0 {
        println!("Error: num_layers_L must be at least 1.");
        return Ok(());
    }

    type MyBackend = NdArray<f32>;
    let device = Default::default();

    let layer_config = LinossLayerConfig {
        d_input_p: model_dim, // Input to the layer within the block is d_model
        d_state_m: model_dim, // State dim of the layer within the block is d_model
        d_output_q: model_dim, // Output of the layer within the block is d_model
        delta_t: 0.001,
        init_std: 0.02,
        enable_d_feedthrough: true, // Corrected field name
    };

    // Define the feed-forward dimension for the block
    let d_ff_block = model_dim * 2; // Example: 2x model dimension for GLU

    let block_config = LinossBlockConfig {
        d_state_m: layer_config.d_state_m, // State dimension for the LinossLayer inside the block
        d_ff: d_ff_block,                  // Feed-forward dimension for GLU
        delta_t: layer_config.delta_t,         // Delta_t for the LinossLayer
        init_std: layer_config.init_std,       // Init_std for the LinossLayer and other linears
        enable_d_feedthrough: layer_config.enable_d_feedthrough,   // Ensure this matches the (now snake_case) field in LinossBlockConfig and sources from the snake_case field in LinossLayerConfig
    };

    let model_config = FullLinossModelConfig {
        d_input: q_dim,
        d_model: model_dim,
        d_output: output_dim,
        n_layers: num_layers_l,
        linoss_block_config: block_config,
    };

    let linoss_model: FullLinossModel<MyBackend> = model_config.init(&device);
    println!("LinossModel instance created.");

    // Create a fixed, simple input sequence
    let mut input_sequence_nalgebra_vec = Vec::new();
    // Time step 0: input value 0.5
    input_sequence_nalgebra_vec.push(Vector::from_vec(vec![0.5f64]));
    // Time step 1: input value 1.0
    input_sequence_nalgebra_vec.push(Vector::from_vec(vec![1.0f64]));

    println!("Fixed Input Sequence (Nalgebra):");
    for (i, vec) in input_sequence_nalgebra_vec.iter().enumerate() {
        println!("  Time step {}: {:?}", i, vec.as_slice());
    }

    if let Some(first_input) = input_sequence_nalgebra_vec.first() {
        println!(
            "Sample Nalgebra Input u_0 (dim {}x{}):",
            first_input.nrows(),
            first_input.ncols()
        );
    }

    let batch_size = 1;
    let mut flat_data_f32: Vec<f32> = Vec::with_capacity(batch_size * sequence_len_n * q_dim);
    for vector_nalgebra in &input_sequence_nalgebra_vec {
        for val in vector_nalgebra.iter() {
            flat_data_f32.push(*val as f32);
        }
    }

    let tensor_shape = Shape::new([batch_size, sequence_len_n, q_dim]);
    // Create TensorData struct
    let tensor_data = TensorData::new(flat_data_f32, tensor_shape);
    // Create tensor from TensorData struct
    let model_input: Tensor<MyBackend, 3> =
        Tensor::from_data(tensor_data.convert::<f32>(), &device);

    println!(
        "Converted input to Burn Tensor with shape: {:?}",
        model_input.dims()
    );

    println!("Performing forward pass...");
    let output_tensor = linoss_model.forward(model_input);
    println!("Forward pass completed.");
    println!("Output tensor shape: {:?}", output_tensor.dims());

    // Convert Burn Tensor output back to Vec<Vector> (nalgebra) for inspection
    // Use to_data() which returns TensorData<E, D>, then call .into_vec()
    let output_tensor_data = output_tensor.to_data();
    let output_data_f32: Vec<f32> = output_tensor_data.into_vec().unwrap();
    let mut output_sequence_nalgebra_vec: Vec<Vector> = Vec::new();

    for i in 0..sequence_len_n {
        let start_index = i * output_dim;
        let end_index = start_index + output_dim;
        if end_index <= output_data_f32.len() {
            let slice = &output_data_f32[start_index..end_index];
            let nalgebra_vector_data: Vec<f64> = slice.iter().map(|&x| x as f64).collect();
            println!(
                "  Raw Nalgebra Output o_{} (before push): {:?}",
                i, nalgebra_vector_data
            );
            output_sequence_nalgebra_vec.push(Vector::from_vec(nalgebra_vector_data));
        }
    }

    println!("Processed Output Sequence (Nalgebra):");
    if !output_sequence_nalgebra_vec.is_empty() {
        for (i, vec) in output_sequence_nalgebra_vec.iter().enumerate() {
            println!("  Time step {}: {:?}", i, vec.as_slice());
        }
    } else {
        println!("Output sequence is empty or conversion failed.");
    }

    Ok(())
}