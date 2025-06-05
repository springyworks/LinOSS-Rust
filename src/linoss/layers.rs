use crate::{Matrix}; // Vector not used here directly but good for consistency

#[derive(Debug, Clone)] // Clone might be useful if params are copied or for easy setup
pub struct LinossBlockParams {
    pub c_matrix: Matrix,       // C matrix for linear readout
    pub d_matrix: Matrix,       // D matrix for linear readout
    pub glu_w1: Matrix,         // W₁ weight matrix for GLU
    pub glu_w2: Matrix,         // W₂ weight matrix for GLU
    // Potentially ODE parameters specific to this block if they vary per layer
    // pub ode_params_for_block: Option<OdeParamsType>, 
}

impl LinossBlockParams {
    pub fn new(
        c_matrix: Matrix,
        d_matrix: Matrix,
        glu_w1: Matrix,
        glu_w2: Matrix,
    ) -> Self {
        // Basic dimension checks could be added here during construction if desired,
        // though they will also be caught during operations in the model's forward pass.
        LinossBlockParams {
            c_matrix,
            d_matrix,
            glu_w1,
            glu_w2,
        }
    }
}