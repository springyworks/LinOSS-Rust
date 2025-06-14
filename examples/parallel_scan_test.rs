// examples/parallel_scan_test.rs
// Test parallel scan functionality

fn main() -> anyhow::Result<()> {
    println!("Parallel scan test starting...");

    // Create a simple test case
    let n = 1000;
    let mut state = vec![0.1; n];
    let input = vec![0.01; n];

    // Test sequential scan
    let start = std::time::Instant::now();
    let result_seq = sequential_scan(&mut state.clone(), &input);
    let seq_time = start.elapsed();

    // Test parallel scan (if available)
    let start = std::time::Instant::now();
    let result_par = parallel_scan(&mut state, &input);
    let par_time = start.elapsed();

    println!("Sequential scan time: {:?}", seq_time);
    println!("Parallel scan time: {:?}", par_time);

    // Verify results are similar
    let diff: f64 = result_seq
        .iter()
        .zip(result_par.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f64>()
        / result_seq.len() as f64;

    println!("Average difference: {:.6}", diff);

    if diff < 1e-6 {
        println!("✅ Parallel scan test passed!");
    } else {
        println!("❌ Parallel scan test failed - results differ significantly");
    }

    Ok(())
}

// Placeholder implementations - replace with actual linoss scan functions
fn sequential_scan(state: &mut [f64], input: &[f64]) -> Vec<f64> {
    let mut result = Vec::with_capacity(state.len());
    for (i, &inp) in input.iter().enumerate() {
        state[i] += inp;
        result.push(state[i]);
    }
    result
}

fn parallel_scan(state: &mut [f64], input: &[f64]) -> Vec<f64> {
    // For now, just use sequential - replace with actual parallel implementation
    sequential_scan(state, input)
}
