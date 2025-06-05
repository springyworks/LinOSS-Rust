use crate::{Vector, Matrix, LinossError};
use super::activation::{gelu, glu};
use super::ode_solver::solve_ode_parallel_scan;
use super::layers::LinossBlockParams;

pub struct LinossModel {
    w_enc: Matrix,
    b_enc: Vector,
    blocks: Vec<LinossBlockParams>, // Parameters for L blocks
    w_dec: Matrix,
    b_dec: Vector,
    // num_layers_l is implicitly self.blocks.len()
}

impl LinossModel {
    pub fn new(
        w_enc: Matrix,
        b_enc: Vector,
        block_params_vec: Vec<LinossBlockParams>,
        w_dec: Matrix,
        b_dec: Vector,
    ) -> Result<Self, LinossError> {
        if block_params_vec.is_empty() {
             return Err(LinossError::Other("Number of layers L (derived from block_params_vec) must be at least 1.".to_string()));
        }
        // Further dimension compatibility checks for W_enc, b_enc, W_dec, b_dec
        // against expected model dimensions (derived from block_params if consistent)
        // could be added here. For example, b_enc.nrows() should match w_enc.nrows().

        Ok(LinossModel {
            w_enc,
            b_enc,
            blocks: block_params_vec,
            w_dec,
            b_dec,
        })
    }

    /// Processes an input sequence through the LinOSS model.
    /// Input: u_sequence = [u₁, u₂, ..., uɴ] where uᵢ ∈ ℝ۹ (input dim)
    /// Output: o_sequence = [o₁, o₂, ..., oɴ] (output dim)
    pub fn forward(&self, input_sequence_u: &[Vector]) -> Result<Vec<Vector>, LinossError> {
        if input_sequence_u.is_empty() {
            return Ok(Vec::new());
        }

        // Encode input sequence: u⁰_i ← Wenc * u_i + benc (applied element-wise)
        // Output dimension of u⁰_i is self.w_enc.nrows() (model_dim)
        let mut current_u_sequence: Vec<Vector> = input_sequence_u
            .iter()
            .map(|u_i| {
                if self.w_enc.ncols() != u_i.nrows() {
                    return Err(LinossError::DimensionMismatch(format!(
                        "Encoding: W_enc ncols ({}) != u_i nrows ({}).",
                        self.w_enc.ncols(), u_i.nrows()
                    )));
                }
                if self.w_enc.nrows() != self.b_enc.nrows() {
                     return Err(LinossError::DimensionMismatch(format!(
                        "Encoding: W_enc nrows ({}) != b_enc nrows ({}).",
                        self.w_enc.nrows(), self.b_enc.nrows()
                    )));
                }
                Ok(&self.w_enc * u_i + &self.b_enc)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut final_y_sequence_for_decoding: Option<Vec<Vector>> = None;

        // for l = 1, ..., L do (0-indexed loop for Vec, corresponds to l=1..L)
        for (l_idx, block_params) in self.blocks.iter().enumerate() {
            let mut next_u_sequence = Vec::with_capacity(current_u_sequence.len());
            let mut y_l_sequence_for_this_layer = Vec::with_capacity(current_u_sequence.len());

            for u_prev_i in current_u_sequence.iter() { // u_prev_i is uˡ⁻¹ for element i
                // yˡ_i ← solution of ODE in (1) with input uˡ⁻¹_i via parallel scan
                let y_l_i = solve_ode_parallel_scan(u_prev_i, l_idx + 1)?; // l_idx+1 for 1-based layer num
                // Assume y_l_i has dimension ode_output_dim (which we assume is model_dim for now)

                // xˡ_i ← C * yˡ_i + D * uˡ⁻¹_i
                if block_params.c_matrix.ncols() != y_l_i.nrows() {
                    return Err(LinossError::DimensionMismatch(format!(
                        "Layer {}: C_matrix ncols ({}) != y_l_i nrows ({}).",
                        l_idx + 1, block_params.c_matrix.ncols(), y_l_i.nrows()
                    )));
                }
                if block_params.d_matrix.ncols() != u_prev_i.nrows() {
                     return Err(LinossError::DimensionMismatch(format!(
                        "Layer {}: D_matrix ncols ({}) != u_prev_i nrows ({}).",
                        l_idx + 1, block_params.d_matrix.ncols(), u_prev_i.nrows()
                    )));
                }
                let cy = &block_params.c_matrix * &y_l_i;
                let du = &block_params.d_matrix * u_prev_i;
                if cy.nrows() != du.nrows() {
                     return Err(LinossError::DimensionMismatch(format!(
                        "Layer {}: Output nrows of C*y ({}) and D*u_prev ({}) for x_l_i_linear do not match.",
                        l_idx + 1, cy.nrows(), du.nrows()
                    )));
                }
                let x_l_i_linear = cy + du; // Dimension is model_dim

                // xˡ_i ← GELU(xˡ_i)
                let x_l_i_activated = gelu(&x_l_i_linear); // Preserves model_dim

                // uˡ_i ← GLU(xˡ_i) + uˡ⁻¹_i
                let glu_out = glu(
                    &x_l_i_activated, // model_dim
                    &block_params.glu_w1, // model_dim x model_dim
                    &block_params.glu_w2, // model_dim x model_dim
                )?; // Output is model_dim
                if glu_out.nrows() != u_prev_i.nrows() { // Both should be model_dim
                    return Err(LinossError::DimensionMismatch(format!(
                        "Layer {}: GLU output nrows ({}) != u_prev_i nrows ({}). Residual connection mismatch.",
                        l_idx + 1, glu_out.nrows(), u_prev_i.nrows()
                    )));
                }
                let u_l_i = glu_out + u_prev_i; // Result is model_dim

                next_u_sequence.push(u_l_i);
                y_l_sequence_for_this_layer.push(y_l_i);
            }
            current_u_sequence = next_u_sequence;
            if l_idx == self.blocks.len() - 1 { // This is yL (from the last block)
                final_y_sequence_for_decoding = Some(y_l_sequence_for_this_layer);
            }
        }

        // o_i ← Wdec * yL_i + bdec (applied element-wise to the y sequence from the *last* block)
        let y_l_sequence = final_y_sequence_for_decoding
            .ok_or_else(|| LinossError::Other("yL sequence not available for decoding. This should not happen if L >= 1.".to_string()))?;

        let output_o_sequence: Vec<Vector> = y_l_sequence
            .iter()
            .map(|y_l_i| { // y_l_i has ode_output_dim (assumed model_dim)
                if self.w_dec.ncols() != y_l_i.nrows() {
                     return Err(LinossError::DimensionMismatch(format!(
                        "Decoding: W_dec ncols ({}) != y_l_i nrows ({}).",
                        self.w_dec.ncols(), y_l_i.nrows()
                    )));
                }
                if self.w_dec.nrows() != self.b_dec.nrows() { // Output dim consistency
                     return Err(LinossError::DimensionMismatch(format!(
                        "Decoding: W_dec nrows ({}) != b_dec nrows ({}).",
                        self.w_dec.nrows(), self.b_dec.nrows()
                    )));
                }
                Ok(&self.w_dec * y_l_i + &self.b_dec) // Output is output_dim
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(output_o_sequence)
    }
}