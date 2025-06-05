use linoss_rust::linoss::{LinossModel, LinossBlockParams};
use linoss_rust::{Vector, Matrix, LinossError};

fn main() -> Result<(), LinossError> {
    println!("LinOSS Model Simulation Start");

    // Define dimensions
    let q_dim = 4;         // Dimension of each input vector uᵢ in the sequence (e.g., embedding dim)
    let model_dim = 8;     // Internal dimension of the model (uˡ, xˡ, yˡ within blocks)
                           // Also output of W_enc, and input to W_dec (from yL)
    let ode_output_dim = model_dim; // Dimension of yˡ (output of ODE solver).
                                   // For simplicity, assumed to be model_dim. If different, C matrix dims change.
    let output_dim = 3;    // Dimension of each output vector oᵢ in the sequence
    let num_layers_l = 2;  // Number of LinOSS blocks (L >= 1)
    let sequence_len_n = 5; // Number of vectors in the input/output sequence

    if num_layers_l == 0 {
        println!("Error: num_layers_L must be at least 1.");
        return Ok(()); // Or return an error
    }

    // --- Initialize Parameters (Dummy random values) ---

    // Encoder: u⁰ ← Wenc * u + benc
    // W_enc: model_dim x q_dim
    // b_enc: model_dim x 1
    let w_enc = Matrix::new_random(model_dim, q_dim);
    let b_enc = Vector::new_random(model_dim);

    // LinossBlockParams for L layers
    let mut block_params_vec = Vec::new();
    for layer_num in 0..num_layers_l {
        // Linear readout: xl ← Cyl + Dul−1
        // C: model_dim x ode_output_dim
        // D: model_dim x model_dim (input uˡ⁻¹ has model_dim)
        let c_matrix = Matrix::new_random(model_dim, ode_output_dim);
        let d_matrix = Matrix::new_random(model_dim, model_dim);

        // GLU weights: GLU(x) = sigmoid(W₁x) ◦ W₂x
        // x (input to GLU) has model_dim. Output of GLU also model_dim for residual.
        // W₁, W₂: model_dim x model_dim
        let glu_w1 = Matrix::new_random(model_dim, model_dim);
        let glu_w2 = Matrix::new_random(model_dim, model_dim);

        println!("Initializing parameters for Layer {}", layer_num + 1);
        block_params_vec.push(LinossBlockParams::new(
            c_matrix, d_matrix, glu_w1, glu_w2,
        ));
    }

    // Decoder: o ← Wdec * yL + bdec
    // yL (output of last block's ODE solver) has ode_output_dim (assumed model_dim)
    // W_dec: output_dim x ode_output_dim
    // b_dec: output_dim x 1
    let w_dec = Matrix::new_random(output_dim, ode_output_dim);
    let b_dec = Vector::new_random(output_dim);

    // Create LinossModel instance
    println!("Creating LinossModel instance...");
    let linoss_model = LinossModel::new(
        w_enc,
        b_enc,
        block_params_vec,
        w_dec,
        b_dec,
    )?;
    println!("LinossModel instance created.");

    // Create a dummy input sequence u = [u₁, u₂, ..., uɴ]
    // Each uᵢ is q_dim x 1
    let mut input_sequence_u = Vec::new();
    for i in 0..sequence_len_n {
        // Adding a small value based on index for slight variation
        input_sequence_u.push(Vector::new_random(q_dim) + Vector::from_element(q_dim, i as f64 * 0.05));
    }

    if let Some(first_input) = input_sequence_u.first() {
        println!("Sample Input u_0 (dim {}x{}):", first_input.nrows(), first_input.ncols());
        for (i, val) in first_input.iter().enumerate() {
            println!("  u_0[{}]: {:.1}", i, val);
        }
    }
    
    // Perform the forward pass
    println!("Performing forward pass...");
    let output_sequence_o = linoss_model.forward(&input_sequence_u)?;
    println!("Forward pass completed.");

    if let Some(first_output) = output_sequence_o.first() {
         println!("Sample Output o_0 (dim {}x{}):", first_output.nrows(), first_output.ncols());
         for (i, val) in first_output.iter().enumerate() {
            println!("  o_0[{}]: {:.1}", i, val);
        }
    }
    println!("Full output sequence length: {}", output_sequence_o.len());
    if !output_sequence_o.is_empty() {
        assert_eq!(output_sequence_o[0].nrows(), output_dim, "Output vector dimension mismatch");
    }
    assert_eq!(output_sequence_o.len(), sequence_len_n, "Output sequence length mismatch");

    println!("LinOSS Model Simulation End");
    Ok(())
}