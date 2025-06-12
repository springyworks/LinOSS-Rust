// src/linoss/layer.rs
// Implementation of the LinossLayer for LinOSS-IM

use burn::prelude::Config;
use burn::{
    module::{Module, Param},
    tensor::{
        backend::Backend, 
        Tensor, 
        Distribution, 
    },
};
use crate::linoss::LinossOutput; // Corrected import path for LinossOutput

/// Configuration for the `LinossLayer` using LinOSS-IM.
#[derive(Config, Debug)] // Removed manual Clone, Config should provide it or it's not needed for config
pub struct LinossLayerConfig {
    /// Dimension of the state vectors y and z (denoted as 'm' in the paper).
    pub d_state_m: usize,
    /// Dimension of the input vector u (denoted as 'p' in the paper).
    pub d_input_p: usize,
    /// Dimension of the output vector x (denoted as 'q' in the paper).
    pub d_output_q: usize,
    /// Time step for discretization.
    pub delta_t: f32, // Made non-optional, will be set by generated `with_delta_t` or direct struct init
    /// Standard deviation for initializing learnable parameters.
    #[config(default = 0.02)]
    pub init_std: f64,
    /// Whether to include the direct feedthrough D matrix (D * u_input term) in the output.
    #[config(default = true)]
    pub enable_d_feedthrough: bool, 
}

impl LinossLayerConfig {
    // Removed manual new() and with_...() methods.
    // #[derive(Config)] will generate with_<field_name>() methods.
    // For fields without a default, they must be set during struct initialization.
    // Example: LinossLayerConfig { d_state_m: ..., d_input_p: ..., d_output_q: ..., delta_t: ..., ..Default::default() }
    // or by using the generated `with_...` methods after a default construction if possible.

    /// Initializes a `LinossLayer` module from this configuration.
    /// This is the method that `#[derive(Config)]` expects for module initialization.
    pub fn init<B: Backend>(&self, device: &B::Device) -> LinossLayer<B> {
        let init_std_val = self.init_std;

        // A_diag_mat (diagonal of matrix A_kk) is a vector.
        let a_diag_mat = Param::from_tensor( 
            Tensor::random([self.d_state_m], Distribution::Normal(0.0, init_std_val), device)
        );
        // B_matrix (matrix B_k)
        let b_matrix = Param::from_tensor(
            Tensor::random([self.d_state_m, self.d_input_p], Distribution::Normal(0.0, init_std_val), device)
        );
        // bias_b (bias for B_matrix * u_input)
        let bias_b = Param::from_tensor(
            Tensor::random([self.d_state_m], Distribution::Normal(0.0, init_std_val), device)
        );
        // C_matrix (matrix C_k)
        let c_matrix = Param::from_tensor(
            Tensor::random([self.d_output_q, self.d_state_m], Distribution::Normal(0.0, init_std_val), device)
        );
        // D_matrix (matrix D_k), optional direct feedthrough.
        let d_matrix = if self.enable_d_feedthrough {
            Some(Param::from_tensor(
                Tensor::random([self.d_output_q, self.d_input_p], Distribution::Normal(0.0, init_std_val), device)
            ))
        } else {
            None
        };

        LinossLayer {
            a_diag_mat, 
            b_matrix,   
            bias_b,     
            c_matrix,   
            d_matrix,   
            d_state_m: self.d_state_m,
            d_input_p: self.d_input_p,
            d_output_q: self.d_output_q,
            delta_t: self.delta_t,
            enable_d_feedthrough: self.enable_d_feedthrough,
        }
    }
}

/// Represents a single Linear Oscillatory State-Space (LinOSS) layer
/// implementing the LinOSS-IM (Implicit Time Integration) method.
/// The combined state is x = [z, y], with dim 2*m.
#[derive(Module, Debug)]
// Removed #[allow(non_snake_case)] as fields will be snake_case
pub struct LinossLayer<B: Backend> { 
    a_diag_mat: Param<Tensor<B, 1>>, // Diagonal of matrix A (A_kk), m elements.
    b_matrix: Param<Tensor<B, 2>>,   // Matrix B (B_k), m x p.
    bias_b: Param<Tensor<B, 1>>,     // Bias for B_matrix * u_input, m elements.

    c_matrix: Param<Tensor<B, 2>>,   // Matrix C (C_k), q x m.
    d_matrix: Option<Param<Tensor<B, 2>>>, // Optional matrix D (D_k), q x p.

    d_state_m: usize,
    d_input_p: usize,
    d_output_q: usize,
    delta_t: f32,
    enable_d_feedthrough: bool, 
}

impl<Be: Backend> LinossLayer<Be> { 
    /// Forward pass for a single time step of the LinOSS layer.
    ///
    /// Args:
    ///     input: Input tensor u_t of shape [batch_size, d_input_p].
    ///     hidden_state_option: Optional previous hidden state y_{t-1} of shape [batch_size, d_state_m].
    ///                          If None, a zero state is used.
    ///
    /// Returns:
    ///     LinossOutput containing:
    ///         - output: Output tensor x_t of shape [batch_size, d_output_q].
    ///         - hidden_state: Next hidden state y_t of shape [batch_size, d_state_m].
    pub fn forward_step(
        &self,
        input: Tensor<Be, 2>,                // u_t, shape [batch, P]
        hidden_state_option: Option<Tensor<Be, 2>>, // y_{t-1}, shape [batch, M]
    ) -> LinossOutput<Be, 2> { // Returns LinossOutput struct
        let device = input.device(); // Get device from input tensor
        let batch_size = input.dims()[0];

        // Initialize hidden_state if not provided
        let y_prev_state = hidden_state_option
            .unwrap_or_else(|| Tensor::zeros([batch_size, self.d_state_m], &device));

        // Get actual tensors from Parameter wrappers
        let a_diag_elements = self.a_diag_mat.val(); // Shape [M]
        let b_matrix_tensor = self.b_matrix.val();   // Shape [M, P]
        let bias_b_tensor = self.bias_b.val();       // Shape [M]
        let c_matrix_tensor = self.c_matrix.val();   // Shape [Q, M]

        // LinOSS-IM equations for y_next_state (internal state)
        // y_next_state = y_prev_state * (I - dt/2 * A_diag)^-1 * (I + dt/2 * A_diag) + 
        //                dt * (I - dt/2 * A_diag)^-1 * (B_matrix * input + bias_b)
        // For simplicity in this step, we'll use the explicit Euler-like update as before,
        // assuming A_diag represents the discretized A matrix directly.
        // A proper IM implementation would involve solving a linear system or using the matrix inverse.
        // The current A_diag is more like (I + dt*A_cont), B is dt*B_cont.

        // y_next_state = y_prev_state * A_diag (element-wise) + input @ B_matrix_tensor^T + bias_b_tensor
        let y_next_state_term1 = y_prev_state.mul(a_diag_elements.clone().unsqueeze::<2>()); // [batch, M] * [1, M] -> [batch, M]
        
        let y_next_state_term2 = input.clone().matmul(b_matrix_tensor.transpose());    // [batch, P] @ [P,M] -> [batch, M]
        
        let y_next_state_intermediate = y_next_state_term1.add(y_next_state_term2);
        let bias_b_broadcastable = bias_b_tensor.unsqueeze::<2>(); // Shape [1, M]
        let y_next_state = y_next_state_intermediate.add(bias_b_broadcastable); // [batch, M] + [1, M] (broadcasts)

        // Calculate output x_t
        // x_t = C_matrix @ y_next_state^T (if y_next_state is row vectors)
        // or x_t = y_next_state @ C_matrix^T (if y_next_state is row vectors and C_matrix is [Q, M])
        // x_t = C_matrix.matmul(y_next_state) if y_next_state is [M, batch_size] (column vectors)
        // Current: y_next_state is [batch, M], C_matrix is [Q, M]
        // So, x_output = y_next_state.matmul(c_matrix_tensor.transpose()) -> [batch, M] @ [M, Q] -> [batch, Q]
        let mut x_output = y_next_state.clone().matmul(c_matrix_tensor.transpose());

        // Add direct feedthrough term D * u_input if enabled
        if self.enable_d_feedthrough {
            if let Some(d_param) = &self.d_matrix {
                let d_tensor = d_param.val(); // Shape [Q, P]
                // input is [batch, P], d_tensor is [Q, P]
                // D_term = input.matmul(d_tensor.transpose()) -> [batch, P] @ [P, Q] -> [batch, Q]
                let d_term = input.matmul(d_tensor.transpose());
                x_output = x_output.add(d_term);
            }
        }
        
        LinossOutput {
            output: x_output,
            hidden_state: Some(y_next_state),
        }
    }

    // Accessor methods for dimensions and delta_t
    pub fn d_state_m(&self) -> usize { self.d_state_m }
    pub fn d_input_p(&self) -> usize { self.d_input_p }
    pub fn d_output_q(&self) -> usize { self.d_output_q }
    pub fn delta_t(&self) -> f32 { self.delta_t }
}
