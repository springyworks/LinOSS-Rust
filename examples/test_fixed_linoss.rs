//! Test for the fixed LinOSS layer
//! This should perform much better than our original implementation

use burn::{
    tensor::{Tensor, TensorData},
};
use linoss_rust::linoss::layer_fixed::FixedLinossLayerConfig;

type MyBackend = burn::backend::NdArray<f32>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Testing Fixed LinOSS Layer ===");
    
    let device = <MyBackend as burn::tensor::backend::Backend>::Device::default();
    
    // Create fixed LinOSS layer (matching Python test parameters)
    let config = FixedLinossLayerConfig {
        ssm_size: 4,  // P=4 oscillators
        h_dim: 1,     // H=1 input/output channels
        delta_t: 0.1,
        init_std: 0.02,
    };
    
    let layer = config.init(&device);
    
    // Test data (same as our analytical tests)
    let seq_len = 20;
    let inputs: Vec<f32> = (0..seq_len).map(|i| i as f32 / (seq_len - 1) as f32).collect();
    
    // Test functions
    let test_cases = [
        ("identity", inputs.clone()),
        ("scaling", inputs.iter().map(|&x| 2.0 * x).collect()),
        ("sine", inputs.iter().map(|&x| (2.0 * std::f32::consts::PI * x).sin()).collect()),
    ];
    
    for (name, targets) in test_cases.iter() {
        println!("\n=== Testing {} ===", name);
        
        // Create tensors [seq_len, h_dim]
        let input_tensor = Tensor::<MyBackend, 2>::from_data(
            TensorData::new(inputs.clone(), [seq_len, 1]),
            &device,
        );
        
        let target_tensor = Tensor::<MyBackend, 2>::from_data(
            TensorData::new(targets.clone(), [seq_len, 1]),
            &device,
        );
        
        // Forward pass
        let output = layer.forward(input_tensor);
        
        // Compute loss
        let loss = (output.clone() - target_tensor).powf_scalar(2.0).mean();
        let loss_val: f32 = loss.into_scalar();
        
        // Get output values
        let output_vals: Vec<f32> = output.into_data().convert::<f32>().into_vec().unwrap();
        
        println!("Input: {:?}", &inputs[0..5]);
        println!("Target: {:?}", &targets[0..5]);
        println!("Output: {:?}", &output_vals[0..5]);
        println!("Loss: {:.6}", loss_val);
    }
    
    Ok(())
}
