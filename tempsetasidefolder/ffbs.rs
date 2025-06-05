use nalgebra::{DMatrix, DVector};
use rand::Rng;
use rand_distr::{Distribution, MultivariateNormal};

#[derive(Debug)]
pub enum FFBSError {
    MatrixInversionError(String),
    DistributionCreationError(String),
    DimensionMismatch(String),
    TimeVaryingParameterMismatch(String),
}

/// Helper function to ensure covariance matrix is suitable for MultivariateNormal
/// It ensures symmetry and attempts to make it positive definite by adding small jitter if Cholesky fails.
fn condition_covariance(cov: DMatrix<f64>, dim: usize, step_name: &str) -> Result<DMatrix<f64>, FFBSError> {
    let mut c = (cov.clone() + cov.transpose()) * 0.5; // Ensure symmetry

    // Check if Cholesky decomposition succeeds, if not, add jitter.
    // nalgebra's MultivariateNormal constructor does this check internally.
    // However, we might want to add jitter proactively if we encounter issues.
    // For now, let's rely on MultivariateNormal's check.
    // If it fails, it will return an error which we will propagate.
    // A small positive definite check:
    if c.cholesky().is_none() {
        println!("Warning: Covariance matrix at {} not positive definite. Adding jitter.", step_name);
        c += DMatrix::identity(dim, dim) * 1e-9; // Add small diagonal jitter
        if c.cholesky().is_none() {
            return Err(FFBSError::DistributionCreationError(format!(
                "Covariance matrix at {} still not positive definite after adding jitter: {:?}", step_name, c
            )));
        }
    }
    Ok(c)
}


/// Implements the Forward Filtering Backward Sampling (FFBS) algorithm for a Dynamic Linear Model.
///
/// # Arguments
/// * `y_series`: A vector of observation vectors `DVector<f64>`, representing `y_1, ..., y_T`.
/// * `m0`: Initial state mean `m_0`.
/// * `C0`: Initial state covariance `C_0`.
/// * `G_series`: State transition matrices `G_t`. If constant, pass a Vec of one matrix.
/// * `F_series`: Observation matrices `F_t`. If constant, pass a Vec of one matrix.
/// * `W_series`: State noise covariances `W_t`. If constant, pass a Vec of one matrix.
/// * `V_series`: Observation noise covariances `V_t`. If constant, pass a Vec of one matrix.
///
/// # Returns
/// A `Result` containing a vector of smoothed state samples `x_0, ..., x_T`, or an `FFBSError`.
pub fn ffbs_dlm(
    y_series: &[DVector<f64>],
    m0: &DVector<f64>,
    C0: &DMatrix<f64>,
    G_series: &[DMatrix<f64>],
    F_series: &[DMatrix<f64>],
    W_series: &[DMatrix<f64>],
    V_series: &[DMatrix<f64>],
) -> Result<Vec<DVector<f64>>, FFBSError> {
    let T = y_series.len();
    if T == 0 {
        // Handle empty observation series: sample from prior x0 ~ N(m0, C0)
        let mut rng = rand::thread_rng();
        let C0_cond = condition_covariance(C0.clone(), m0.nrows(), "C0 for empty y_series")?;
        let dist_x0 = MultivariateNormal::new(m0.clone(), C0_cond.clone())
            .map_err(|e| FFBSError::DistributionCreationError(format!("x0 (empty y): {}", e)))?;
        return Ok(vec![dist_x0.sample(&mut rng)]);
    }

    let dim_x = m0.nrows();
    // let dim_y = y_series[0].nrows(); // Assuming y_series is not empty

    // Helper to get time-varying or constant matrix
    let get_matrix = |series: &[DMatrix<f64>], t: usize, name: &str| -> Result<&DMatrix<f64>, FFBSError> {
        if series.len() == 1 {
            Ok(&series[0])
        } else if series.len() == T {
            Ok(&series[t]) // t is 0-indexed for y_series, G_series[t] is G_{t+1}
        } else {
            Err(FFBSError::TimeVaryingParameterMismatch(format!(
                "Length of {} series ({}) must be 1 or T ({})",
                name, series.len(), T
            )))
        }
    };
    
    // --- Forward Filtering ---
    let mut m_vec: Vec<DVector<f64>> = Vec::with_capacity(T + 1);
    let mut C_vec: Vec<DMatrix<f64>> = Vec::with_capacity(T + 1);
    let mut a_vec: Vec<DVector<f64>> = Vec::with_capacity(T); // a_1, ..., a_T
    let mut R_vec: Vec<DMatrix<f64>> = Vec::with_capacity(T); // R_1, ..., R_T

    m_vec.push(m0.clone());
    C_vec.push(C0.clone());

    for t in 0..T { // Corresponds to time t=1, ..., T in algorithm notation
        let G_t = get_matrix(G_series, t, "G")?; // G_{t+1} in math, or G if constant
        let F_t = get_matrix(F_series, t, "F")?; // F_{t+1} in math, or F if constant
        let W_t = get_matrix(W_series, t, "W")?; // W_{t+1} in math, or W if constant
        let V_t = get_matrix(V_series, t, "V")?; // V_{t+1} in math, or V if constant

        let m_prev = &m_vec[t]; // m_{t} in math (or m_0 for first iteration)
        let C_prev = &C_vec[t]; // C_{t} in math (or C_0 for first iteration)

        // Prediction (prior for x_{t+1})
        // a_{t+1} = G_{t+1} * m_t
        let a_next = G_t * m_prev;
        // R_{t+1} = G_{t+1} * C_t * G_{t+1}^T + W_{t+1}
        let R_next = G_t * C_prev * G_t.transpose() + W_t;

        // Forecast (for y_{t+1})
        // f_{t+1} = F_{t+1} * a_{t+1}
        let f_next = F_t * &a_next;
        // Q_{t+1} = F_{t+1} * R_{t+1} * F_{t+1}^T + V_{t+1}
        let Q_next = F_t * &R_next * F_t.transpose() + V_t;
        
        let Q_next_inv = Q_next.clone().try_inverse().ok_or_else(|| {
            FFBSError::MatrixInversionError(format!("Q_next at t={} not invertible. Q_next: {:?}", t, Q_next))
        })?;

        // Update (posterior for x_{t+1})
        // A_{t+1} = R_{t+1} * F_{t+1}^T * Q_{t+1}^{-1}
        let A_next = &R_next * F_t.transpose() * &Q_next_inv;
        
        // y_obs is y_{t+1} from input y_series[t]
        let y_obs = &y_series[t]; 
        
        // m_{t+1} = a_{t+1} + A_{t+1} * (y_{t+1} - f_{t+1})
        let m_curr = &a_next + &A_next * (y_obs - &f_next);
        // C_{t+1} = R_{t+1} - A_{t+1} * F_{t+1} * R_{t+1}
        let C_curr = &R_next - &A_next * F_t * &R_next;

        m_vec.push(m_curr);
        C_vec.push(condition_covariance(C_curr, dim_x, &format!("C_curr at t={}", t))?);
        a_vec.push(a_next);
        R_vec.push(condition_covariance(R_next, dim_x, &format!("R_next at t={}",t))?);
    }

    // --- Backward Sampling ---
    let mut x_smoothed: Vec<DVector<f64>> = vec![DVector::zeros(dim_x); T + 1];
    let mut rng = rand::thread_rng();

    // Sample x_T ~ N(m_T, C_T)
    // m_T is m_vec[T], C_T is C_vec[T]
    let dist_T = MultivariateNormal::new(m_vec[T].clone(), C_vec[T].clone())
        .map_err(|e| FFBSError::DistributionCreationError(format!("x_T: {}", e)))?;
    x_smoothed[T] = dist_T.sample(&mut rng);

    // For t = T-1, ..., 0
    for t_idx in (0..T).rev() { // t_idx corresponds to state x_{t_idx}
        // We need:
        // m_t (m_vec[t_idx]), C_t (C_vec[t_idx])
        // G_{t+1} (G_series for index t_idx, as it's G for transition from t_idx to t_idx+1)
        // R_{t+1} (R_vec[t_idx], which is R for state at t_idx+1 prior)
        // a_{t+1} (a_vec[t_idx], which is a for state at t_idx+1 prior)
        // x_{t+1} (x_smoothed[t_idx+1])

        let G_next = get_matrix(G_series, t_idx, "G_backward")?; // G_{t_idx+1}
        
        let C_t = &C_vec[t_idx];
        let m_t = &m_vec[t_idx];
        let R_t_plus_1 = &R_vec[t_idx]; // R_{t_idx+1}
        let a_t_plus_1 = &a_vec[t_idx]; // a_{t_idx+1}
        let x_t_plus_1_smoothed = &x_smoothed[t_idx + 1];

        let R_t_plus_1_inv = R_t_plus_1.clone().try_inverse().ok_or_else(|| {
            FFBSError::MatrixInversionError(format!("R_{{t+1}} at t_idx={} not invertible. R: {:?}", t_idx, R_t_plus_1))
        })?;

        // B_t = C_t * G_{t+1}^T * R_{t+1}^{-1}
        let B_t = C_t * G_next.transpose() * &R_t_plus_1_inv;
        
        // h_t = m_t + B_t * (x_{t+1} - a_{t+1})
        let h_t = m_t + &B_t * (x_t_plus_1_smoothed - a_t_plus_1);
        
        // H_t = C_t - B_t * G_{t+1} * C_t
        let H_t_uncond = C_t - &B_t * G_next * C_t;
        let H_t = condition_covariance(H_t_uncond, dim_x, &format!("H_t at t_idx={}", t_idx))?;


        let dist_t = MultivariateNormal::new(h_t.clone(), H_t.clone())
            .map_err(|e| FFBSError::DistributionCreationError(format!("x_{}: {}", t_idx, e)))?;
        x_smoothed[t_idx] = dist_t.sample(&mut rng);
    }

    Ok(x_smoothed)
}