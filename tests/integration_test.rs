// /tests/integration_test.rs
use burn::tensor::{Shape, Tensor, TensorData, backend::Backend};
use linoss_rust::linoss::{
    block::LinossBlockConfig, // Corrected name
    model::{FullLinossModel, FullLinossModelConfig}, // Corrected names
    DLinossLayer, DLinossLayerConfig, // Add D-LinOSS imports (correct path)
    ProductionDLinossBlock, ProductionDLinossConfig, // Add Production D-LinOSS imports for Euler testing
};
use log::{debug, info};

// Define a test backend (e.g., NdArray)
type TestBackend = burn::backend::ndarray::NdArray<f32>;

// Helper function to initialize the logger for tests
fn init_test_logger() {
    // env_logger::init() can be called multiple times, but it will only initialize once.
    // try_init is better as it doesn't panic if already initialized.
    let _ = env_logger::builder().is_test(true).try_init();
}

#[test]
fn test_model_initialization_and_forward_pass() {
    init_test_logger(); // Initialize logger
    info!("--- Running Integration Test: Model Initialization and Forward Pass ---"); // Changed from println!
    let device: <TestBackend as Backend>::Device = Default::default();

    // Configuration
    let d_input = 4;
    let d_model = 8;
    let d_output = 2;
    let n_layers = 1;
    let batch_size = 1;
    let seq_len = 3;

    // Note: LinossLayerConfig is used by LinossBlockConfig internally.
    // We define the parameters that LinossBlockConfig will use to create its LinossLayerConfig.
    // The d_input for the layer within the block is d_model of the block.
    // The d_output_q for the layer within the block is also d_model of the block.
    let block_conf = LinossBlockConfig {
        d_state_m: d_model * 2, // Example state dimension for the layer in the block
        d_ff: d_model * 4,      // Example feed-forward dimension for the block
        delta_t: 1.0,
        init_std: 0.02,
        enable_d_feedthrough: true,       // Assuming the layer within the block has a D term
    };

    let model_config = FullLinossModelConfig {
        d_input,
        d_model,
        d_output,
        n_layers,
        linoss_block_config: block_conf, // Corrected name
    };

    info!("Integration Test: Initializing model..."); // Changed from println!
    let model: FullLinossModel<TestBackend> = model_config.init(&device); // Corrected name
    info!("Integration Test: Model initialized."); // Changed from println!

    // Input
    let input_elem_count = batch_size * seq_len * d_input;
    let input_vec: Vec<f32> = (0..input_elem_count).map(|x| x as f32 * 0.1).collect();
    let input_data = TensorData::new(input_vec, Shape::new([batch_size, seq_len, d_input]));
    let input_tensor: Tensor<TestBackend, 3> =
        Tensor::from_data(input_data.convert::<f32>(), &device);
    debug!(
        "Integration Test: Input tensor created with shape: {:?}",
        input_tensor.dims()
    ); // Changed from println!

    // Forward pass
    info!("Integration Test: Performing forward pass..."); // Changed from println!
    let output_tensor = model.forward(input_tensor);
    debug!(
        "Integration Test: Forward pass completed. Output shape: {:?}",
        output_tensor.dims()
    ); // Changed from println!

    // Assertions
    let expected_shape_dims = [batch_size, seq_len, d_output]; // Use array for direct comparison
    assert_eq!(
        output_tensor.dims(),
        expected_shape_dims,
        "Output tensor shape mismatch in integration test."
    );

    let output_data_vec: Vec<f32> = output_tensor
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .unwrap();
    assert_eq!(
        output_data_vec.len(),
        batch_size * seq_len * d_output,
        "Output data length mismatch in integration test."
    );

    let mut all_zeros = true;
    let mut has_nan = false;
    for val in output_data_vec.iter() {
        if *val != 0.0 {
            all_zeros = false;
        }
        if val.is_nan() {
            has_nan = true;
        }
    }
    assert!(
        !all_zeros,
        "Output tensor is all zeros in integration test."
    );
    assert!(
        !has_nan,
        "Output tensor contains NaN values in integration test."
    );

    info!("--- Integration Test: Model Initialization and Forward Pass Completed Successfully ---"); // Changed from println!
}

#[test]
fn test_minimal_model_with_sine_input() {
    init_test_logger();
    info!("--- Running Integration Test: Minimal Model with Sine Wave Input ---");

    let device: <TestBackend as Backend>::Device = Default::default();

    // Minimal Configuration
    let d_input = 1;
    let d_model = 4;
    let d_output = 1;
    let n_layers = 1;
    let batch_size = 1;
    let seq_len = 50;

    // Configure LinossBlockConfig directly
    let block_conf = LinossBlockConfig {
        d_state_m: d_model, // Example: state_m related to d_model
        d_ff: d_model * 2,  // Example: feed-forward dimension
        delta_t: 0.5,
        init_std: 0.5,
        enable_d_feedthrough: true,
    };

    let model_config = FullLinossModelConfig {
        d_input,
        d_model,
        d_output,
        n_layers,
        linoss_block_config: block_conf, // Corrected name
    };

    info!("Integration Test (Sine): Initializing minimal model...");
    let model: FullLinossModel<TestBackend> = model_config.init(&device); // Corrected name
    info!("Integration Test (Sine): Minimal model initialized.");

    // Sine wave Input
    let mut input_vec: Vec<f32> = Vec::with_capacity(batch_size * seq_len * d_input);
    for i in 0..(batch_size * seq_len * d_input) {
        // Create a sine wave: sin(t * frequency_factor)
        // For d_input = 1, batch_size = 1, this iterates seq_len times.
        let t = i as f32 * 0.2; // Adjust 0.2 to change frequency if needed
        input_vec.push(t.sin());
    }

    let input_data = TensorData::new(input_vec, Shape::new([batch_size, seq_len, d_input]));
    let input_tensor: Tensor<TestBackend, 3> =
        Tensor::from_data(input_data.convert::<f32>(), &device);
    debug!(
        "Integration Test (Sine): Input tensor created with shape: {:?}, first 5 values: {:?}",
        input_tensor.dims(),
        input_tensor
            .clone()
            .slice([0..1, 0..5, 0..1])
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .unwrap()
    );

    // Forward pass
    info!("Integration Test (Sine): Performing forward pass...");
    let output_tensor = model.forward(input_tensor);
    debug!(
        "Integration Test (Sine): Forward pass completed. Output shape: {:?}",
        output_tensor.dims()
    );

    // Assertions
    let expected_shape_dims = [batch_size, seq_len, d_output]; // Use array for direct comparison
    assert_eq!(
        output_tensor.dims(),
        expected_shape_dims,
        "Output tensor shape mismatch in sine test."
    );

    let output_data_vec: Vec<f32> = output_tensor
        .clone()
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .unwrap();
    assert_eq!(
        output_data_vec.len(),
        batch_size * seq_len * d_output,
        "Output data length mismatch in sine test."
    );

    debug!(
        "Integration Test (Sine): First 10 output values: {:?}",
        &output_data_vec[0..output_data_vec.len().min(10)]
    );
    debug!(
        "Integration Test (Sine): Last 10 output values: {:?}",
        &output_data_vec[(output_data_vec.len().saturating_sub(10))..]
    );

    let mut all_zeros = true;
    let mut has_nan = false;
    let mut has_inf = false;
    for &val in output_data_vec.iter() {
        if val != 0.0 {
            all_zeros = false;
        }
        if val.is_nan() {
            has_nan = true;
        }
        if val.is_infinite() {
            has_inf = true;
        }
    }
    assert!(!all_zeros, "Output tensor is all zeros in sine test.");
    assert!(!has_nan, "Output tensor contains NaN values in sine test.");
    assert!(!has_inf, "Output tensor contains Inf values in sine test.");

    // Check for some variation in output - a very basic check
    if output_data_vec.len() > 1 {
        let first_val = output_data_vec[0];
        let mut all_same = true;
        for &val in output_data_vec.iter().skip(1) {
            if (val - first_val).abs() > 1e-6 {
                // Check if values differ significantly
                all_same = false;
                break;
            }
        }
        assert!(
            !all_same,
            "Output tensor values are all (nearly) identical, expected some variation with sine input."
        );
    }

    info!("--- Integration Test: Minimal Model with Sine Wave Input Completed Successfully ---");
}

#[test]
fn test_dlinoss_euler_discretization() {
    init_test_logger();
    info!("--- Running Integration Test: D-LinOSS Euler Discretization ---");

    let device: <TestBackend as Backend>::Device = Default::default();

    // Test configuration for D-LinOSS with Euler discretization
    let d_input = 2;
    let d_model = 4;
    let d_output = 2;
    let batch_size = 2;
    let seq_len = 10;

    // Create D-LinOSS layer configuration
    let config = DLinossLayerConfig::new_dlinoss(d_input, d_model, d_output);

    info!("Integration Test (Euler): Initializing D-LinOSS layer...");
    let layer: DLinossLayer<TestBackend> = DLinossLayer::new(&config, &device);
    info!("Integration Test (Euler): D-LinOSS layer initialized.");

    // Create test input tensor
    let input_vec: Vec<f32> = (0..(batch_size * seq_len * d_input))
        .map(|i| (i as f32 * 0.1).sin()) // Use sine wave for interesting dynamics
        .collect();
    
    let input_data = TensorData::new(input_vec, Shape::new([batch_size, seq_len, d_input]));
    let input_tensor: Tensor<TestBackend, 3> = Tensor::from_data(input_data.convert::<f32>(), &device);
    
    debug!(
        "Integration Test (Euler): Input tensor shape: {:?}",
        input_tensor.dims()
    );

    // Forward pass - this uses the internal discretization scheme of D-LinOSS
    info!("Integration Test (Euler): Performing forward pass...");
    let output_tensor = layer.forward(input_tensor.clone());
    
    debug!(
        "Integration Test (Euler): Forward pass completed. Output shape: {:?}",
        output_tensor.dims()
    );

    // Assertions for D-LinOSS forward pass
    let expected_shape = [batch_size, seq_len, d_output];
    assert_eq!(
        output_tensor.dims(),
        expected_shape,
        "D-LinOSS output shape mismatch"
    );

    // Check output characteristics
    let output_data: Vec<f32> = output_tensor
        .clone()
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .unwrap();
    
    // Verify no NaN or Inf values (stability check)
    let mut has_nan = false;
    let mut has_inf = false;
    let mut all_zeros = true;
    
    for &val in output_data.iter() {
        if val.is_nan() {
            has_nan = true;
        }
        if val.is_infinite() {
            has_inf = true;
        }
        if val.abs() > 1e-6 {
            all_zeros = false;
        }
    }
    
    assert!(!has_nan, "D-LinOSS produced NaN values");
    assert!(!has_inf, "D-LinOSS produced infinite values");
    assert!(!all_zeros, "D-LinOSS output is all zeros");

    // Test that output has reasonable magnitude (not exploding)
    let max_abs_val = output_data.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    assert!(
        max_abs_val < 100.0,
        "D-LinOSS may be unstable - max absolute value: {}",
        max_abs_val
    );

    // Verify output has some variation (dynamics are working)
    if output_data.len() > 1 {
        let mean = output_data.iter().sum::<f32>() / output_data.len() as f32;
        let variance = output_data.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / output_data.len() as f32;
        
        assert!(
            variance > 1e-8,
            "D-LinOSS output has no variation - variance: {}",
            variance
        );
    }

    debug!(
        "Integration Test (Euler): Output statistics - min: {:.6}, max: {:.6}, mean: {:.6}",
        output_data.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
        output_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)),
        output_data.iter().sum::<f32>() / output_data.len() as f32
    );

    // Test that running the same input twice gives the same output (deterministic)
    let output_tensor2 = layer.forward(input_tensor);
    let output_data2: Vec<f32> = output_tensor2
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .unwrap();
    
    let max_diff = output_data.iter()
        .zip(output_data2.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    
    assert!(
        max_diff < 1e-6,
        "D-LinOSS is not deterministic - max difference: {}",
        max_diff
    );

    info!("--- Integration Test: D-LinOSS Layer Forward Pass Completed Successfully ---");
}

#[test]
fn test_production_dlinoss_euler_discretization() {
    init_test_logger();
    info!("--- Running Integration Test: Production D-LinOSS Euler Discretization ---");

    let device: <TestBackend as Backend>::Device = Default::default();

    // Test configuration for Production D-LinOSS with Euler discretization
    let d_model = 64;
    let d_inner = 128;
    let batch_size = 4;
    let seq_len = 32;

    // Create Production D-LinOSS layer with Euler discretization
    let mut config = ProductionDLinossConfig::default();
    config.d_model = d_model;
    config.d_inner = d_inner;
    config.discretization = "euler".to_string(); // Test Euler specifically
    config.activation = "gelu".to_string();
    config.learnable_damping = true;

    info!("Integration Test (Production Euler): Initializing Production D-LinOSS block with Euler discretization...");
    let block: ProductionDLinossBlock<TestBackend> = ProductionDLinossBlock::new(config, &device);
    info!("Integration Test (Production Euler): Production D-LinOSS block initialized.");

    // Create test input tensor with interesting dynamics
    let input_vec: Vec<f32> = (0..(batch_size * seq_len * d_model))
        .map(|i| {
            let t = i as f32 * 0.05;
            // Mix of frequencies to test oscillatory dynamics
            0.5 * (t * 2.0).sin() + 0.3 * (t * 5.0).cos() + 0.2 * (t * 0.5).sin()
        })
        .collect();
    
    let input_data = TensorData::new(input_vec, Shape::new([batch_size, seq_len, d_model]));
    let input_tensor: Tensor<TestBackend, 3> = Tensor::from_data(input_data.convert::<f32>(), &device);
    
    debug!(
        "Integration Test (Production Euler): Input tensor shape: {:?}",
        input_tensor.dims()
    );

    // Forward pass with Production D-LinOSS using Euler discretization
    info!("Integration Test (Production Euler): Performing forward pass with Euler discretization...");
    let output_tensor = block.forward(input_tensor.clone());
    
    debug!(
        "Integration Test (Production Euler): Forward pass completed. Output shape: {:?}",
        output_tensor.dims()
    );

    // Assertions for Production D-LinOSS Euler discretization
    let expected_shape = [batch_size, seq_len, d_model];
    assert_eq!(
        output_tensor.dims(),
        expected_shape,
        "Production D-LinOSS Euler discretization output shape mismatch"
    );

    // Check output characteristics
    let output_data: Vec<f32> = output_tensor
        .clone()
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .unwrap();
    
    // Verify no NaN or Inf values (critical for Euler stability)
    let mut has_nan = false;
    let mut has_inf = false;
    let mut all_zeros = true;
    
    for &val in output_data.iter() {
        if val.is_nan() {
            has_nan = true;
        }
        if val.is_infinite() {
            has_inf = true;
        }
        if val.abs() > 1e-6 {
            all_zeros = false;
        }
    }
    
    assert!(!has_nan, "Production D-LinOSS Euler discretization produced NaN values");
    assert!(!has_inf, "Production D-LinOSS Euler discretization produced infinite values");
    assert!(!all_zeros, "Production D-LinOSS Euler discretization output is all zeros");

    // Test Euler discretization stability - should not explode
    let max_abs_val = output_data.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    assert!(
        max_abs_val < 50.0,
        "Euler discretization may be unstable - max absolute value: {}",
        max_abs_val
    );

    // Test that Euler produces meaningful dynamics (not constant output)
    if output_data.len() > 10 {
        let first_batch_seq = &output_data[0..seq_len * d_model];
        let differences: Vec<f32> = first_batch_seq.windows(d_model)
            .map(|window| window.iter().map(|x| x.abs()).sum::<f32>())
            .collect::<Vec<_>>()
            .windows(2)
            .map(|pair| (pair[1] - pair[0]).abs())
            .collect();
        
        let avg_difference = differences.iter().sum::<f32>() / differences.len() as f32;
        assert!(
            avg_difference > 1e-6,
            "Euler discretization output too static - avg change: {}",
            avg_difference
        );
    }

    // Test energy properties with damping
    if seq_len > 5 {
        let energy_start = output_data[0..d_model].iter().map(|x| x * x).sum::<f32>();
        let mid_idx = (seq_len / 2) * d_model;
        let energy_mid = output_data[mid_idx..mid_idx + d_model].iter().map(|x| x * x).sum::<f32>();
        
        debug!(
            "Integration Test (Production Euler): Energy start: {:.6}, mid: {:.6}",
            energy_start, energy_mid
        );
        
        // With damping enabled, we expect some energy management (not unlimited growth)
        assert!(
            energy_mid <= energy_start * 3.0, // Allow some growth but not explosion
            "Euler discretization with damping shows excessive energy growth - start: {}, mid: {}",
            energy_start, energy_mid
        );
    }

    debug!(
        "Integration Test (Production Euler): Output statistics - min: {:.6}, max: {:.6}, mean: {:.6}, std: {:.6}",
        output_data.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
        output_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)),
        output_data.iter().sum::<f32>() / output_data.len() as f32,
        {
            let mean = output_data.iter().sum::<f32>() / output_data.len() as f32;
            (output_data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / output_data.len() as f32).sqrt()
        }
    );

    // Test deterministic behavior (Euler should be deterministic)
    let output_tensor2 = block.forward(input_tensor);
    let output_data2: Vec<f32> = output_tensor2
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .unwrap();
    
    let max_diff = output_data.iter()
        .zip(output_data2.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    
    assert!(
        max_diff < 1e-6,
        "Production D-LinOSS Euler discretization is not deterministic - max difference: {}",
        max_diff
    );

    info!("--- Integration Test: Production D-LinOSS Euler Discretization Completed Successfully ---");
}
