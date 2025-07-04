use burn::{
    backend::Autodiff,
    module::Module,
    tensor::{backend::Backend, Data, Tensor},
};
use burn_wgpu::WgpuBackend;

use linoss_rust::linoss::production_dlinoss::ProductionDLinossModel;

type MyBackend = WgpuBackend<f32, i32>;

#[derive(Module, Debug)]
pub struct SimpleMnistClassifier<B: Backend> {
    dlinoss: ProductionDLinossModel<B>,
}

impl<B: Backend> SimpleMnistClassifier<B> {
    pub fn new(device: &B::Device) -> Self {
        Self {
            dlinoss: ProductionDLinossModel::new(
                128,  // d_model
                2,    // num_layers  
                10,   // num_classes (MNIST)
                device,
            ),
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        // Simple approach: pad or project input to match d_model
        let batch_size = input.dims()[0];
        
        // Take first 128 features or pad to 128
        let processed_input = if input.dims()[1] >= 128 {
            input.slice([0..batch_size, 0..128])
        } else {
            // Pad with zeros if input is smaller than d_model
            let padding_size = 128 - input.dims()[1];
            let zeros = Tensor::zeros([batch_size, padding_size], &input.device());
            Tensor::cat(vec![input, zeros], 1)
        };
        
        // Reshape to sequence format [batch, seq_len=1, features]
        let input_seq = processed_input.reshape([batch_size, 1, 128]);
        
        // Forward through D-LinOSS model
        self.dlinoss.forward(input_seq)
    }
}

pub fn simple_demo() {
    println!("🚀 Simple D-LinOSS MNIST Demo");
    
    let device = burn_wgpu::WgpuDevice::default();
    let model = SimpleMnistClassifier::new(&device);
    
    // Create synthetic batch
    let batch_size = 4;
    let input_data: Vec<Vec<f32>> = (0..batch_size)
        .map(|i| (0..784).map(|j| ((i + j) % 10) as f32 / 10.0).collect())
        .collect();
    
    let input_tensor = Tensor::<MyBackend, 2>::from_data(
        Data::from(input_data), 
        &device
    );
    
    println!("Input shape: {:?}", input_tensor.dims());
    
    let output = model.forward(input_tensor);
    println!("Output shape: {:?}", output.dims());
    
    let output_data: Vec<f32> = output.into_data().to_vec().unwrap();
    println!("Sample outputs: {:?}", &output_data[..10]);
    
    println!("✅ D-LinOSS MNIST demo completed successfully!");
}

fn main() {
    simple_demo();
}
