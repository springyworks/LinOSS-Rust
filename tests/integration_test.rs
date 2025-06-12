// /tests/integration_test.rs
use burn::tensor::{Shape, Tensor, TensorData, backend::Backend};
use linoss_rust::linoss::{
    block::LinossBlockConfig, // Corrected name
    model::{FullLinossModel, FullLinossModelConfig}, // Corrected names
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
