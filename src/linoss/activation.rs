use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{backend::Backend, Tensor, activation},
};

/// Gated Linear Unit (GLU) implementation for Burn
#[derive(Module, Debug)]
pub struct GLU<B: Backend> {
    /// First linear projection (for gate)
    pub w1: Linear<B>,
    /// Second linear projection (for value)
    pub w2: Linear<B>,
}

impl<B: Backend> GLU<B> {
    /// Create a new GLU layer
    pub fn new(input_dim: usize, output_dim: usize, device: &B::Device) -> Self {
        let w1 = LinearConfig::new(input_dim, output_dim).init(device);
        let w2 = LinearConfig::new(input_dim, output_dim).init(device);
        
        Self { w1, w2 }
    }
    
    /// Forward pass: GLU(x) = sigmoid(W1 * x) ⊙ (W2 * x)
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let gate = activation::sigmoid(self.w1.forward(input.clone()));
        let value = self.w2.forward(input);
        gate * value
    }
}

// Legacy GLU functions for compatibility
use crate::{Vector, Matrix, LinossError}; // Ensure this is the only `use` for these types here

// GELU (Gaussian Error Linear Unit) approximation
// GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
pub fn gelu(x_vec: &Vector) -> Vector {
    x_vec.map(|x| {
        0.5 * x * (1.0 + (2.0f64 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh()
    })
}

// Sigmoid function (element-wise for a vector)
fn sigmoid_vec(v: &Vector) -> Vector {
    v.map(|val| 1.0 / (1.0 + (-val).exp()))
}

// GLU (Gated Linear Unit)
// GLU(x_input, W1, W2) = sigmoid(W1 * x_input) ◦ (W2 * x_input)
// x_input: d_in x 1
// W1: d_out x d_in
// W2: d_out x d_in
// Output: d_out x 1
pub fn glu(
    x_input: &Vector,
    w1: &Matrix,
    w2: &Matrix,
) -> Result<Vector, LinossError> {
    if w1.ncols() != x_input.nrows() || w2.ncols() != x_input.nrows() {
        return Err(LinossError::DimensionMismatch(
            format!("GLU: W1 ncols ({}) / W2 ncols ({}) must match x_input nrows ({}).", w1.ncols(), w2.ncols(), x_input.nrows())
        ));
    }
    if w1.nrows() != w2.nrows() {
        return Err(LinossError::DimensionMismatch(
            format!("GLU: W1 nrows ({}) must match W2 nrows ({}).", w1.nrows(), w2.nrows())
        ));
    }

    let gate = sigmoid_vec(&(w1 * x_input));
    let val = w2 * x_input;

    if gate.nrows() != val.nrows() {
        return Err(LinossError::DimensionMismatch(
            format!("GLU: Output of sigmoid(W1*x) nrows ({}) must match W2*x nrows ({}). This should not happen if previous checks passed.", gate.nrows(), val.nrows())
        ));
    }

    Ok(gate.component_mul(&val)) // Element-wise multiplication
}