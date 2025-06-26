// src/linoss/parallel_scan.rs
// Implementation of the parallel scan algorithm for LinOSS recurrence

use burn::tensor::{Tensor, backend::Backend};

/// Struct to hold the pair (M, F) for LinOSS recurrence
#[derive(Debug, Clone)]
pub struct RecurrencePair<B: Backend> {
    pub matrix_m: Tensor<B, 2>, // Matrix M (or M^-1 for IM variant)
    pub vector_f: Tensor<B, 2>, // Vector F
}

impl<B: Backend> RecurrencePair<B> {
    pub fn new(matrix_m: Tensor<B, 2>, vector_f: Tensor<B, 2>) -> Self {
        Self { matrix_m, vector_f }
    }

    /// Associative operation for parallel scan: (a1, a2) · (b1, b2) = (b1·a1, b1·a2 + b2)
    /// Where a1, b1 are matrices, and a2, b2 are vectors
    // Made this a public static method so it can be used by standalone functions
    pub fn associative_op(
        pair1: &Self,
        pair2: &Self,
        _device: &B::Device, // device is not strictly needed here if ops are on tensors
    ) -> Self {
        // Compute b1·a1 (matrix multiplication)
        let new_m = pair2.matrix_m.clone().matmul(pair1.matrix_m.clone());

        // Compute b1·a2 + b2
        let new_f = pair2.matrix_m.clone().matmul(pair1.vector_f.clone()) + pair2.vector_f.clone();

        Self::new(new_m, new_f)
    }
}

// Standalone scan functions that operate on a slice of RecurrencePair elements.
// These are intended for use by the LinOSSLayer.

/// Sequential scan for a sequence of (M, F) pairs.
/// x_n = M_n * x_{n-1} + F_n (Note: M is constant in LinOSS, so M_n = M)
pub fn perform_sequential_scan<B: Backend>(
    initial_x: Tensor<B, 2>,
    elements: &[RecurrencePair<B>], // Sequence of (M, F_t)
    _device: &B::Device,            // Mark as unused if not needed
) -> Vec<Tensor<B, 2>> {
    let seq_len = elements.len();
    let mut x_history = Vec::with_capacity(seq_len);
    let mut current_x = initial_x;

    for (_t, pair_t) in elements.iter().enumerate().take(seq_len) {
        // x_t = M_t * x_{t-1} + F_t
        // In LinOSS, M is constant across time steps, so pair_t.matrix_m is the same M_IM.
        let new_x = pair_t.matrix_m.clone().matmul(current_x.clone()) + pair_t.vector_f.clone();
        x_history.push(new_x.clone());
        current_x = new_x;
    }
    x_history
}

/// Parallel scan (recursive doubling style) for a sequence of (M, F) pairs.
pub fn perform_parallel_scan<B: Backend>(
    initial_x: Tensor<B, 2>,
    elements: &[RecurrencePair<B>], // Sequence of (M, F_t)
    device: &B::Device,
) -> Vec<Tensor<B, 2>> {
    let seq_len = elements.len();
    if seq_len == 0 {
        return Vec::new();
    }

    // Compute prefix sums of pairs using the associative operation
    let mut prefix_pairs = Vec::with_capacity(seq_len);
    prefix_pairs.push(elements[0].clone());

    for i in 1..seq_len {
        prefix_pairs.push(RecurrencePair::associative_op(
            &prefix_pairs[i - 1],
            &elements[i],
            device,
        ));
    }

    // Compute the output states by applying each prefix to the initial state
    let mut result = Vec::with_capacity(seq_len);
    let initial_state_cloned = initial_x.clone(); // Clone once before the loop

    for (_i, prefix_pair) in prefix_pairs.iter().enumerate().take(seq_len) {
        // x_i = M_prefix_i * x_initial + F_prefix_i
        // This is the application of the combined (M, F) to the initial state.
        let x_i = prefix_pair
            .matrix_m
            .clone()
            .matmul(initial_state_cloned.clone())
            + prefix_pair.vector_f.clone();
        result.push(x_i);
    }
    result
}

/// Tree-based parallel scan (conceptually similar to work-efficient for this structure)
pub fn perform_tree_scan<B: Backend>(
    initial_x: Tensor<B, 2>,
    elements: &[RecurrencePair<B>],
    device: &B::Device,
) -> Vec<Tensor<B, 2>> {
    let seq_len = elements.len();
    if seq_len == 0 {
        return Vec::new();
    }
    // For small sequences, a direct parallel scan (prefix sum) is fine.
    // A true tree-based scan has a more complex structure for combining elements.
    // The current `apply_tree_scan` in the original code was essentially a prefix sum.
    // We will use the same prefix sum logic here for consistency with the previous structure,
    // acknowledging that a "true" tree scan might differ in implementation details for optimization.

    // Fallback for very small sequences (optional, can be handled by the layer or example)
    // if seq_len <= 4 {
    //     return perform_sequential_scan(initial_x, elements, device);
    // }

    let mut prefix_pairs = Vec::with_capacity(seq_len);
    prefix_pairs.push(elements[0].clone());

    for i in 1..seq_len {
        prefix_pairs.push(RecurrencePair::associative_op(
            &prefix_pairs[i - 1],
            &elements[i],
            device,
        ));
    }

    let mut result = Vec::with_capacity(seq_len);
    let initial_state_cloned = initial_x.clone();

    for prefix_pair in prefix_pairs.iter().take(seq_len) {
        let x_i = prefix_pair
            .matrix_m
            .clone()
            .matmul(initial_state_cloned.clone())
            + prefix_pair.vector_f.clone();
        result.push(x_i);
    }
    result
}

/// Work-efficient parallel scan.
/// For the recurrence x_n = M_n * x_{n-1} + F_n, the associative operator (M', F') = (M_b M_a, M_b F_a + F_b)
/// allows a standard parallel prefix sum (scan) algorithm.
/// The implementation here will be the same as perform_parallel_scan, as it's already work-efficient
/// for this type of recurrence when implemented with prefix sums.
pub fn perform_work_efficient_scan<B: Backend>(
    initial_x: Tensor<B, 2>,
    elements: &[RecurrencePair<B>],
    device: &B::Device,
) -> Vec<Tensor<B, 2>> {
    // This is effectively the same as the recursive doubling / prefix sum approach for this problem.
    perform_parallel_scan(initial_x, elements, device)
}

// Removed the old apply_*_scan methods from RecurrencePair impl<B: Backend> block
// as they were confusing and not correctly structured for use with a sequence of varying F vectors.
// The standalone functions above (perform_*) should be used by the layer.
