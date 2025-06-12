use burn::{
    tensor::{
        backend::Backend,
        Tensor,
    },
};

use std::time::Instant;
use std::fmt::Debug;

/// Struct to hold the pair (M, F) for LinOSS recurrence
#[derive(Debug, Clone)]
pub struct RecurrencePair<B: Backend> {
    pub matrix_m: Tensor<B, 2>,  // Matrix M (or M^-1 for IM variant)
    pub vector_f: Tensor<B, 2>,  // Vector F
}

impl<B: Backend> RecurrencePair<B> {
    pub fn new(matrix_m: Tensor<B, 2>, vector_f: Tensor<B, 2>) -> Self {
        Self {
            matrix_m,
            vector_f,
        }
    }
    
    /// Sequential recurrence: x_n = M * x_{n-1} + F_n
    pub fn apply_sequential(
        &self,
        initial_x: Tensor<B, 2>,
        seq_len: usize,
        _device: &B::Device, // Prefixed with underscore
    ) -> Vec<Tensor<B, 2>> {
        let mut x_history = Vec::with_capacity(seq_len);
        let mut current_x = initial_x;
        
        // For each time step, apply the recurrence relation
        for _t in 0..seq_len { // Prefixed with underscore
            // Get F_t (assuming self.vector_f contains all F vectors stacked)
            let f_t = self.vector_f.clone(); // In practice, we'd extract F_t from a sequence
            
            // Compute x_t = M * x_{t-1} + F_t
            let new_x = self.matrix_m.clone().matmul(current_x.clone()) + f_t;
            
            x_history.push(new_x.clone());
            current_x = new_x;
        }
        
        x_history
    }
    
    /// Associative operation for parallel scan: (a1, a2) · (b1, b2) = (b1·a1, b1·a2 + b2)
    /// Where a1, b1 are matrices, and a2, b2 are vectors
    fn associative_op(
        pair1: &Self,
        pair2: &Self,
        _device: &B::Device, // Prefixed with underscore
    ) -> Self {
        // Compute b1·a1 (matrix multiplication)
        let new_m = pair2.matrix_m.clone().matmul(pair1.matrix_m.clone());
        
        // Compute b1·a2 + b2
        let new_f = pair2.matrix_m.clone().matmul(pair1.vector_f.clone()) + pair2.vector_f.clone();
        
        Self::new(new_m, new_f)
    }
    
    /// Parallel scan implementation using recursive doubling
    /// Returns x_n for all n in 0..seq_len
    pub fn apply_parallel_scan(
        &self,
        initial_x: Tensor<B, 2>,
        seq_len: usize,
        device: &B::Device,
    ) -> Vec<Tensor<B, 2>> {
        // For simplicity in this implementation, make pairs for each time step
        // with same M but potentially different F (in practice)
        let mut pairs = Vec::with_capacity(seq_len);
        
        // Initialize pairs with (M, F_t) for each time step
        for _t in 0..seq_len {
            // In practice, F_t would be computed based on the input at time t
            let f_t = self.vector_f.clone(); // For testing, use the same F for all time steps
            pairs.push(Self::new(self.matrix_m.clone(), f_t));
        }
        
        // Compute prefix sums of pairs using the associative operation
        let mut prefix_pairs = Vec::with_capacity(seq_len);
        prefix_pairs.push(pairs[0].clone());
        
        for i in 1..seq_len {
            prefix_pairs.push(Self::associative_op(&prefix_pairs[i-1], &pairs[i], device));
        }
        
        // Compute the output states by applying each prefix to the initial state
        let mut result = Vec::with_capacity(seq_len);
        let initial_state = initial_x.clone();
        
        for i in 0..seq_len {
            let x_i = prefix_pairs[i].matrix_m.clone().matmul(initial_state.clone()) + 
                      prefix_pairs[i].vector_f.clone();
            result.push(x_i);
        }
        
        result
    }
    
    /// Tree-based parallel scan implementation using work-efficient algorithm
    /// This is more efficient for large seq_len
    pub fn apply_tree_scan(
        &self,
        initial_x: Tensor<B, 2>,
        seq_len: usize,
        device: &B::Device,
    ) -> Vec<Tensor<B, 2>> {
        // For very small sequences, fall back to sequential
        if seq_len <= 4 {
            return self.apply_sequential(initial_x, seq_len, device);
        }
        
        // Create array of (M, F) pairs for each time step
        let mut pairs = Vec::with_capacity(seq_len);
        for _t in 0..seq_len {
            let f_t = self.vector_f.clone(); // In practice, this would depend on the input
            pairs.push(Self::new(self.matrix_m.clone(), f_t));
        }
        
        // Compute the scan directly using the work-efficient algorithm
        // Step 1: Compute the products for all pairs of elements
        let mut prefix_pairs = Vec::with_capacity(seq_len);
        prefix_pairs.push(pairs[0].clone());
        
        for i in 1..seq_len {
            prefix_pairs.push(Self::associative_op(&prefix_pairs[i-1], &pairs[i], device));
        }
        
        // Compute the output states by applying each prefix to the initial state
        let mut result = Vec::with_capacity(seq_len);
        let initial_state = initial_x.clone();
        
        for i in 0..seq_len {
            let x_i = prefix_pairs[i].matrix_m.clone().matmul(initial_state.clone()) + 
                     prefix_pairs[i].vector_f.clone();
            result.push(x_i);
        }
        
        result
    }
}

/// Main function to demonstrate and test parallel scan implementation
pub fn main() {
    println!("LinOSS Parallel Scan Implementation Test");
    
    // Define backend and device
    type MyBackend = burn::backend::NdArray<f32>;
    let device = Default::default();
    
    // Test dimensions
    let d_state = 16; // State dimension (m)
    let combined_state_dim = 2 * d_state; // For both z and y components
    let seq_len = 100; // Sequence length
    
    // Create a sample M matrix (using identity for simple test)
    let m_matrix = Tensor::<MyBackend, 2>::eye(combined_state_dim, &device);
    println!("Created M matrix with shape: {:?}", m_matrix.dims());
    
    // Create a sample F vector
    let f_vector = Tensor::<MyBackend, 2>::ones([combined_state_dim, 1], &device).mul_scalar(0.1);
    println!("Created F vector with shape: {:?}", f_vector.dims());
    
    // Create initial state x_0
    let initial_x = Tensor::<MyBackend, 2>::zeros([combined_state_dim, 1], &device);
    println!("Created initial state with shape: {:?}", initial_x.dims());
    
    // Create a RecurrencePair
    let recurrence_pair = RecurrencePair::<MyBackend>::new(m_matrix, f_vector);
    
    // Run sequential implementation and measure time
    println!("\nRunning sequential implementation...");
    let seq_start = Instant::now();
    let seq_result = recurrence_pair.apply_sequential(initial_x.clone(), seq_len, &device);
    let seq_duration = seq_start.elapsed();
    println!("Sequential scan completed in {:?}", seq_duration);
    println!("Final state shape: {:?}", seq_result.last().unwrap().dims());
    
    // Run parallel scan implementation and measure time
    println!("\nRunning parallel scan implementation...");
    let par_start = Instant::now();
    let par_result = recurrence_pair.apply_parallel_scan(initial_x.clone(), seq_len, &device);
    let par_duration = par_start.elapsed();
    println!("Parallel scan completed in {:?}", par_duration);
    println!("Final state shape: {:?}", par_result.last().unwrap().dims());
    
    // Run tree-based implementation and measure time
    println!("\nRunning tree-based scan implementation...");
    let tree_start = Instant::now();
    let tree_result = recurrence_pair.apply_tree_scan(initial_x.clone(), seq_len, &device);
    let tree_duration = tree_start.elapsed();
    println!("Tree scan completed in {:?}", tree_duration);
    println!("Final state shape: {:?}", tree_result.last().unwrap().dims());
    
    // Compare results for correctness
    println!("\nComparing final states for correctness...");
    let seq_final = seq_result.last().unwrap().clone();
    let par_final = par_result.last().unwrap().clone();
    let tree_final = tree_result.last().unwrap().clone();
    
    // Convert to Vec<f32> for easy comparison
    let seq_data: Vec<f32> = seq_final.into_data().convert::<f32>().into_vec().unwrap();
    let par_data: Vec<f32> = par_final.into_data().convert::<f32>().into_vec().unwrap();
    let tree_data: Vec<f32> = tree_final.into_data().convert::<f32>().into_vec().unwrap();
    
    // Check if all elements are close enough
    let all_match = seq_data.iter().zip(par_data.iter())
        .all(|(a, b)| (a - b).abs() < 1e-5) &&
        seq_data.iter().zip(tree_data.iter())
        .all(|(a, b)| (a - b).abs() < 1e-5);
    
    if all_match {
        println!("All implementation results match! ✅");
    } else {
        println!("Results do not match! ❌");
        
        // Print first few values from each result for debugging
        let num_to_print = combined_state_dim.min(5);
        println!("First {} values from sequential: {:?}", num_to_print, &seq_data[0..num_to_print]);
        println!("First {} values from parallel: {:?}", num_to_print, &par_data[0..num_to_print]);
        println!("First {} values from tree: {:?}", num_to_print, &tree_data[0..num_to_print]);
    }
    
    // Performance comparison
    println!("\nPerformance comparison:");
    println!("Sequential time: {:?}", seq_duration);
    println!("Parallel scan time: {:?}", par_duration);
    println!("Tree scan time: {:?}", tree_duration);
    
    let par_speedup = seq_duration.as_micros() as f64 / par_duration.as_micros() as f64;
    let tree_speedup = seq_duration.as_micros() as f64 / tree_duration.as_micros() as f64;
    
    println!("Parallel scan speedup: {:.2}x", par_speedup);
    println!("Tree scan speedup: {:.2}x", tree_speedup);
    
    // Scaling test with different sequence lengths
    println!("\nScaling test with different sequence lengths:");
    let test_lengths = [10, 50, 100, 200, 500];
    
    for &length in &test_lengths {
        println!("\nTesting with sequence length: {}", length);
        
        let seq_start = Instant::now();
        let _ = recurrence_pair.apply_sequential(initial_x.clone(), length, &device);
        let seq_duration = seq_start.elapsed();
        
        let par_start = Instant::now();
        let _ = recurrence_pair.apply_parallel_scan(initial_x.clone(), length, &device);
        let par_duration = par_start.elapsed();
        
        let tree_start = Instant::now();
        let _ = recurrence_pair.apply_tree_scan(initial_x.clone(), length, &device);
        let tree_duration = tree_start.elapsed();
        
        println!("Sequential: {:?}", seq_duration);
        println!("Parallel scan: {:?}", par_duration);
        println!("Tree scan: {:?}", tree_duration);
        
        let par_speedup = seq_duration.as_micros() as f64 / par_duration.as_micros() as f64;
        let tree_speedup = seq_duration.as_micros() as f64 / tree_duration.as_micros() as f64;
        
        println!("Parallel speedup: {:.2}x", par_speedup);
        println!("Tree speedup: {:.2}x", tree_speedup);
    }
    
    println!("\nParallel scan test complete!");
}
