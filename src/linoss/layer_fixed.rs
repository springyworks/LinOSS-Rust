// src/linoss/layer_fixed.rs
// Fixed LinOSS layer implementation based on Python reference

use burn::prelude::Config;
use burn::{
    module::{Module, Param},
    tensor::{
        backend::Backend, 
        Tensor, 
        Distribution,
        activation::relu,
    },
};
use num_traits::Float;

/// Configuration for the fixed LinOSS layer
#[derive(Config, Debug)]
pub struct FixedLinossLayerConfig {
    /// SSM size (number of oscillators, P in the paper)
    pub ssm_size: usize,
    /// Input/output dimension (H in the paper)  
    pub h_dim: usize,
    /// Time step discretization
    pub delta_t: f64,
    /// Standard deviation for initialization
    #[config(default = 0.02)]
    pub init_std: f64,
}

impl FixedLinossLayerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> FixedLinossLayer<B> {
        let ssm_size = self.ssm_size;
        let h_dim = self.h_dim;
        let init_std = self.init_std;
        
        // A_diag: uniform random initialization (like Python)
        let a_diag = Param::from_tensor(
            Tensor::random([ssm_size], Distribution::Uniform(-1.0, 1.0), device)
        );
        
        // B matrix: complex-like structure [ssm_size, h_dim, 2] for real/imaginary parts
        // Initialized with proper scaling: std = 1/sqrt(h_dim)
        let b_std = 1.0 / (h_dim as f64).sqrt();
        let b_matrix = Param::from_tensor(
            Tensor::random([ssm_size, h_dim, 2], Distribution::Uniform(-b_std, b_std), device)
        );
        
        // C matrix: complex-like structure [h_dim, ssm_size, 2] for real/imaginary parts  
        // Initialized with proper scaling: std = 1/sqrt(ssm_size)
        let c_std = 1.0 / (ssm_size as f64).sqrt();
        let c_matrix = Param::from_tensor(
            Tensor::random([h_dim, ssm_size, 2], Distribution::Uniform(-c_std, c_std), device)
        );
        
        // D matrix: direct feedthrough
        let d_matrix = Param::from_tensor(
            Tensor::random([h_dim], Distribution::Normal(0.0, init_std), device)
        );
        
        // Steps: discretization time steps
        let steps = Param::from_tensor(
            Tensor::random([ssm_size], Distribution::Uniform(0.0, 1.0), device)
        );
        
        FixedLinossLayer {
            a_diag,
            b_matrix,
            c_matrix, 
            d_matrix,
            steps,
            ssm_size,
            h_dim,
            delta_t: self.delta_t,
        }
    }
}

/// Fixed LinOSS layer matching Python implementation
#[derive(Module, Debug)]
pub struct FixedLinossLayer<B: Backend> {
    a_diag: Param<Tensor<B, 1>>,      // [P] - diagonal A matrix
    b_matrix: Param<Tensor<B, 3>>,    // [P, H, 2] - complex B matrix
    c_matrix: Param<Tensor<B, 3>>,    // [H, P, 2] - complex C matrix  
    d_matrix: Param<Tensor<B, 1>>,    // [H] - direct feedthrough
    steps: Param<Tensor<B, 1>>,       // [P] - discretization steps
    
    ssm_size: usize,
    h_dim: usize,
    delta_t: f64,
}

impl<B: Backend> FixedLinossLayer<B> 
where
    B::FloatElem: Float,
{
    /// Forward pass implementing LinOSS-IM discretization
    pub fn forward(&self, input_sequence: Tensor<B, 2>) -> Tensor<B, 2> {
        let [seq_len, _h_dim] = input_sequence.dims();
        
        // Apply ReLU to A_diag for stability (like Python)
        let a_diag_pos = relu(self.a_diag.val());
        
        // Apply sigmoid to steps for normalization (like Python)  
        let steps_norm = burn::tensor::activation::sigmoid(self.steps.val());
        
        // Convert B and C to complex representation
        let b_real = self.b_matrix.val().clone().slice([0..self.ssm_size, 0..self.h_dim, 0..1]).squeeze(2);
        let b_imag = self.b_matrix.val().clone().slice([0..self.ssm_size, 0..self.h_dim, 1..2]).squeeze(2);
        
        let c_real = self.c_matrix.val().clone().slice([0..self.h_dim, 0..self.ssm_size, 0..1]).squeeze(2);
        let c_imag = self.c_matrix.val().clone().slice([0..self.h_dim, 0..self.ssm_size, 1..2]).squeeze(2);
        
        // Compute Bu for each time step
        let mut bu_elements = Vec::new();
        for t in 0..seq_len {
            let u_t = input_sequence.clone().slice([t..t+1, 0..self.h_dim]).squeeze(0);
            
            // Complex multiplication: (B_real + i*B_imag) @ u
            let bu_real = b_real.clone().matmul(u_t.clone());
            let bu_imag = b_imag.clone().matmul(u_t);
            
            bu_elements.push((bu_real, bu_imag));
        }
        
        // LinOSS-IM discretization (following Python implementation)
        let schur_comp = (steps_norm.clone().powf_scalar(2.0) * a_diag_pos.clone() + 1.0).recip();
        
        let m_im_11 = -steps_norm.clone().powf_scalar(2.0) * a_diag_pos.clone() * schur_comp.clone() + 1.0;
        let m_im_12 = -steps_norm.clone() * a_diag_pos.clone() * schur_comp.clone();
        let m_im_21 = steps_norm.clone() * schur_comp.clone();
        let m_im_22 = schur_comp.clone();
        
        // Initial state
        let device = input_sequence.device();
        let mut state_real = Tensor::zeros([self.ssm_size], &device);
        let mut state_imag = Tensor::zeros([self.ssm_size], &device);
        
        let mut outputs = Vec::new();
        
        // Sequential processing (simplified version of parallel scan)
        for (bu_real, bu_imag) in bu_elements {
            // Update states using LinOSS-IM transition
            let f1_real = m_im_11.clone() * bu_real.clone() * steps_norm.clone();
            let f1_imag = m_im_11.clone() * bu_imag.clone() * steps_norm.clone();
            let f2_real = m_im_21.clone() * bu_real * steps_norm.clone();  
            let f2_imag = m_im_21.clone() * bu_imag * steps_norm.clone();
            
            let new_state_real = m_im_11.clone() * state_real.clone() + m_im_12.clone() * state_imag.clone() + f1_real;
            let new_state_imag = m_im_21.clone() * state_real + m_im_22.clone() * state_imag + f1_imag;
            
            // Output state is the second component (like Python xs[:, A_diag.shape[0]:])
            let output_state_real = f2_real + new_state_real.clone();
            let output_state_imag = f2_imag + new_state_imag.clone();
            
            // Complex output: C @ (output_state_real + i*output_state_imag), take real part
            let y_real = c_real.clone().matmul(output_state_real.clone()) - c_imag.clone().matmul(output_state_imag.clone());
            let _y_imag = c_real.clone().matmul(output_state_imag) + c_imag.clone().matmul(output_state_real);
            
            // Take real part (like Python .real)
            outputs.push(y_real);
            
            state_real = new_state_real;
            state_imag = new_state_imag;
        }
        
        // Stack outputs and add direct feedthrough
        let output_tensor = Tensor::stack(outputs, 0);
        let d_u = input_sequence * self.d_matrix.val().unsqueeze_dim(0);
        
        output_tensor + d_u
    }
}
