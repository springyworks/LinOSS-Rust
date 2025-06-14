//! Single Block Analytical Test
//!
//! Tests individual LinOSS and dLinOSS blocks against known analytical functions

use burn::{
    optim::{AdamConfig, Optimizer, GradientsParams},
    tensor::{
        backend::AutodiffBackend,
        ElementConversion, Tensor, TensorData,
    },
};
use linoss_rust::linoss::{
    model::{FullLinossModel, FullLinossModelConfig},
    block::LinossBlockConfig,
    dlinoss_layer::{AParameterization, DLinossLayer, DLinossLayerConfig},
};
use std::{
    fs::{OpenOptions, create_dir_all},
    io::Write,
};
use num_traits::FloatConst;

// Test parameters
const SEQ_LEN: usize = 20;  // Shorter for simpler test
const BATCH_SIZE: usize = 1;
const LEARNING_RATE: f64 = 1e-3;  // Lower learning rate for stability
const N_EPOCHS: usize = 100;  // More epochs for better learning
const D_MODEL: usize = 8;    // Smaller model

type MyBackend = burn::backend::NdArray<f32>;
type MyAutodiffBackend = burn::backend::Autodiff<MyBackend>;

/// Simple single-layer LinOSS model
#[derive(burn::module::Module, Debug)]
pub struct SingleLinossTest<B: burn::tensor::backend::Backend> {
    model: FullLinossModel<B>,
}

impl<B: burn::tensor::backend::Backend> SingleLinossTest<B> 
where 
    B::FloatElem: FloatConst + From<f32> + std::ops::Mul<Output = B::FloatElem> + Copy,
{
    pub fn new(device: &B::Device) -> Self {
        let block_config = LinossBlockConfig {
            d_state_m: D_MODEL * 2,
            d_ff: D_MODEL,
            delta_t: 0.1,
            init_std: 0.02,
            enable_d_feedthrough: false,
        };
        
        let config = FullLinossModelConfig {
            d_input: 1,
            d_model: D_MODEL,
            d_output: 1,
            n_layers: 1,  // Single layer test
            linoss_block_config: block_config,
        };
        
        Self {
            model: config.init(device),
        }
    }
    
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        self.model.forward(input)
    }
}

/// Simple single-layer dLinOSS model  
#[derive(burn::module::Module, Debug)]
pub struct SingleDLinossTest<B: burn::tensor::backend::Backend> {
    input_linear: burn::nn::Linear<B>,
    dlinoss_layer: DLinossLayer<B>,
    output_linear: burn::nn::Linear<B>,
}

impl<B: burn::tensor::backend::Backend> SingleDLinossTest<B> {
    pub fn new(device: &B::Device) -> Self {
        let config = DLinossLayerConfig {
            d_input: D_MODEL,
            d_model: D_MODEL,
            d_output: D_MODEL,
            delta_t: 0.1,
            init_std: 0.02,
            enable_layer_norm: false,
            enable_damping: false,
            init_damping: 0.0,
            num_damping_scales: 1,
            a_parameterization: AParameterization::GELU,
        };
        
        Self {
            input_linear: burn::nn::LinearConfig::new(1, D_MODEL).init(device),
            dlinoss_layer: DLinossLayer::new(&config, device),
            output_linear: burn::nn::LinearConfig::new(D_MODEL, 1).init(device),
        }
    }
    
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let processed = self.input_linear.forward(input);
        let dlinoss_out = self.dlinoss_layer.forward(processed);
        self.output_linear.forward(dlinoss_out)
    }
}

/// Test cases
#[derive(Debug, Clone, Copy)]
enum TestCase {
    Identity,    // f(x) = x
    Scaling,     // f(x) = 2x  
    Step,        // f(x) = step function
    Sine,        // f(x) = sin(2πx)
    Exponential, // f(x) = exp(-2x)
    Quadratic,   // f(x) = x²
}

impl TestCase {
    fn name(&self) -> &'static str {
        match self {
            TestCase::Identity => "Identity",
            TestCase::Scaling => "Scaling", 
            TestCase::Step => "Step",
            TestCase::Sine => "Sine",
            TestCase::Exponential => "Exponential",
            TestCase::Quadratic => "Quadratic",
        }
    }
    
    fn generate_data(&self) -> (Vec<f32>, Vec<f32>) {
        let inputs: Vec<f32> = (0..SEQ_LEN).map(|i| i as f32 / (SEQ_LEN - 1) as f32).collect();
        
        let targets: Vec<f32> = inputs.iter().map(|&x| {
            match self {
                TestCase::Identity => x,
                TestCase::Scaling => 2.0 * x,
                TestCase::Step => if x > 0.5 { 1.0 } else { 0.0 },
                TestCase::Sine => (2.0 * std::f32::consts::PI * x).sin(),
                TestCase::Exponential => (-2.0 * x).exp(),
                TestCase::Quadratic => x * x,
            }
        }).collect();
        
        (inputs, targets)
    }
}

fn debug_log(message: &str) {
    // Ensure logs directory exists
    if let Err(_) = create_dir_all("logs") {
        println!("Warning: Could not create logs directory");
    }
    
    if let Ok(mut file) = OpenOptions::new()
        .create(true)
        .append(true)
        .open("logs/single_block_test.log")
    {
        let _ = writeln!(file, "{}", message);
    }
    println!("{}", message);
}

fn test_single_block<AB: AutodiffBackend>(
    test_case: TestCase,
    use_dlinoss: bool,
    device: &AB::Device,
) where
    AB::FloatElem: From<f32> + std::ops::Mul<Output = AB::FloatElem> + Copy + ElementConversion + num_traits::FloatConst,
{
    debug_log(&format!("=== Testing {} with {} ===", 
        test_case.name(), 
        if use_dlinoss { "dLinOSS" } else { "LinOSS" }
    ));
    
    // Generate test data
    let (input_data, target_data) = test_case.generate_data();
    
    // Create tensors
    let input_tensor = Tensor::<AB, 2>::from_data(
        TensorData::new(input_data.clone(), [BATCH_SIZE, SEQ_LEN]),
        device,
    ).reshape([BATCH_SIZE, SEQ_LEN, 1]);
    
    let target_tensor = Tensor::<AB, 2>::from_data(
        TensorData::new(target_data.clone(), [BATCH_SIZE, SEQ_LEN]),
        device,
    ).reshape([BATCH_SIZE, SEQ_LEN, 1]);
    
    debug_log(&format!("Input: {:.3?}", &input_data[0..5]));
    debug_log(&format!("Target: {:.3?}", &target_data[0..5]));
    
    // Training
    if use_dlinoss {
        let mut model = SingleDLinossTest::new(device);
        let mut optimizer = AdamConfig::new().init();
        
        for epoch in 0..N_EPOCHS {
            let output = model.forward(input_tensor.clone());
            let loss = (output.clone() - target_tensor.clone()).powf_scalar(2.0).mean();
            let loss_val = loss.clone().into_scalar().elem::<f64>();
            
            if epoch % 20 == 0 || epoch == N_EPOCHS - 1 {
                let output_data: Vec<f32> = output.clone()
                    .into_data()
                    .convert::<f32>()
                    .into_vec()
                    .unwrap();
                
                debug_log(&format!("Epoch {}: Loss={:.6}, Output: {:.3?}", 
                    epoch, loss_val, &output_data[0..5]));
            }
            
            let grads_raw = loss.backward();
            let grads = GradientsParams::from_grads(grads_raw, &model);
            model = optimizer.step(LEARNING_RATE, model, grads);
        }
    } else {
        let mut model = SingleLinossTest::new(device);
        let mut optimizer = AdamConfig::new().init();
        
        for epoch in 0..N_EPOCHS {
            let output = model.forward(input_tensor.clone());
            let loss = (output.clone() - target_tensor.clone()).powf_scalar(2.0).mean();
            let loss_val = loss.clone().into_scalar().elem::<f64>();
            
            if epoch % 20 == 0 || epoch == N_EPOCHS - 1 {
                let output_data: Vec<f32> = output.clone()
                    .into_data()
                    .convert::<f32>()
                    .into_vec()
                    .unwrap();
                
                debug_log(&format!("Epoch {}: Loss={:.6}, Output: {:.3?}", 
                    epoch, loss_val, &output_data[0..5]));
            }
            
            let grads_raw = loss.backward();
            let grads = GradientsParams::from_grads(grads_raw, &model);
            model = optimizer.step(LEARNING_RATE, model, grads);
        }
    }
    debug_log("");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Ensure logs directory exists
    let _ = create_dir_all("logs");
    
    // Clear log
    let _ = std::fs::remove_file("logs/single_block_test.log");
    
    debug_log("=== Single Block Analytical Test ===");
    
    let device = <MyAutodiffBackend as burn::tensor::backend::Backend>::Device::default();
    
    // Test simple cases
    let test_cases = [
        TestCase::Identity, 
        TestCase::Scaling, 
        TestCase::Step,
        TestCase::Sine,
        TestCase::Exponential,
        TestCase::Quadratic,
    ];
    
    for test_case in test_cases.iter() {
        test_single_block::<MyAutodiffBackend>(*test_case, false, &device);  // LinOSS
        test_single_block::<MyAutodiffBackend>(*test_case, true, &device);   // dLinOSS
    }
    
    debug_log("=== Test Complete ===");
    
    Ok(())
}
