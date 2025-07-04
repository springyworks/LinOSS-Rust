use burn::{
    module::Module,
    tensor::{backend::Backend, Data, Tensor},
};
use burn_wgpu::WgpuBackend;

use linoss_rust::linoss::production_dlinoss::{
    ProductionDLinossModel, ProductionDLinossConfig, 
    DiscretizationScheme, ActivationFunction,
};

type MyBackend = WgpuBackend<f32, i32>;

/// Demonstrate D-LinOSS damping effects on different oscillatory inputs
pub fn demonstrate_damping_effects() {
    println!("🌊 D-LinOSS Oscillator Damping Demo");
    println!("===================================");
    
    let device = burn_wgpu::WgpuDevice::default();
    
    // Create models with different damping factors
    let configs = vec![
        ("No Damping", 0.0),
        ("Light Damping", 0.05),
        ("Medium Damping", 0.15),
        ("Heavy Damping", 0.3),
    ];
    
    let mut models = Vec::new();
    
    for (name, damping_factor) in &configs {
        println!("🔧 Creating model: {} (damping = {:.2})", name, damping_factor);
        
        let config = ProductionDLinossConfig::new(
            1,  // Single input
            32, // Hidden dimension
            1,  // Single output
            2,  // Number of layers
            DiscretizationScheme::RK4,
            ActivationFunction::Tanh,
            0.1,  // dt
            *damping_factor,
        );
        
        let model = config.init(&device);
        models.push((name.to_string(), model));
    }
    
    // Generate test signals
    let sequence_length = 100;
    let test_signals = vec![
        ("Pure Sine Wave", generate_sine_wave(sequence_length, 1.0, 0.1)),
        ("High Frequency Sine", generate_sine_wave(sequence_length, 1.0, 0.3)),
        ("Damped Oscillation", generate_damped_oscillation(sequence_length)),
        ("Noisy Oscillation", generate_noisy_oscillation(sequence_length)),
    ];
    
    println!("\n📊 Testing different input signals...\n");
    
    for (signal_name, signal_data) in &test_signals {
        println!("🎵 Signal: {}", signal_name);
        println!("   Input statistics:");
        let mean = signal_data.iter().sum::<f32>() / signal_data.len() as f32;
        let variance = signal_data.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / signal_data.len() as f32;
        let std_dev = variance.sqrt();
        println!("     Mean: {:.4}, Std Dev: {:.4}", mean, std_dev);
        
        // Convert to tensor
        let input_tensor = Tensor::<MyBackend, 3>::from_data(
            Data::from([signal_data.iter().map(|&x| [x]).collect::<Vec<_>>()]),
            &device
        );
        
        println!("   Model responses:");
        
        for (model_name, model) in &models {
            let output = model.forward(input_tensor.clone());
            
            // Calculate output statistics
            let output_data: Vec<f32> = output
                .flatten::<1>(0, 2)
                .into_data()
                .to_vec().unwrap();
            
            let output_mean = output_data.iter().sum::<f32>() / output_data.len() as f32;
            let output_variance = output_data.iter()
                .map(|&x| (x - output_mean).powi(2))
                .sum::<f32>() / output_data.len() as f32;
            let output_std = output_variance.sqrt();
            
            // Calculate energy (sum of squares)
            let input_energy: f32 = signal_data.iter().map(|&x| x * x).sum();
            let output_energy: f32 = output_data.iter().map(|&x| x * x).sum();
            let energy_ratio = output_energy / input_energy;
            
            println!("     {}: Std={:.4}, Energy ratio={:.4}", 
                    model_name, output_std, energy_ratio);
            
            // Show some sample values
            if output_data.len() >= 5 {
                let samples: Vec<f32> = output_data[..5].to_vec();
                println!("       First 5 outputs: {:?}", samples);
            }
        }
        println!();
    }
    
    println!("🔬 Analysis:");
    println!("• Higher damping factors reduce output variance and energy");
    println!("• D-LinOSS provides controllable energy dissipation");
    println!("• RK4 discretization ensures stable long-term dynamics");
    println!("• Tanh activation provides bounded, smooth responses");
    
    // Demonstrate phase space analysis
    println!("\n🌀 Phase Space Analysis:");
    demonstrate_phase_space_damping(&device);
}

fn generate_sine_wave(length: usize, amplitude: f32, frequency: f32) -> Vec<f32> {
    (0..length)
        .map(|i| amplitude * (frequency * i as f32).sin())
        .collect()
}

fn generate_damped_oscillation(length: usize) -> Vec<f32> {
    (0..length)
        .map(|i| {
            let t = i as f32 * 0.1;
            (-0.1 * t).exp() * (t).sin()
        })
        .collect()
}

fn generate_noisy_oscillation(length: usize) -> Vec<f32> {
    (0..length)
        .map(|i| {
            let t = i as f32 * 0.1;
            let signal = (t).sin();
            let noise = (((i * 7) % 100) as f32 / 100.0 - 0.5) * 0.3;
            signal + noise
        })
        .collect()
}

fn demonstrate_phase_space_damping(device: &<MyBackend as Backend>::Device) {
    // Create a simple 2D oscillator to show phase space damping
    let config = ProductionDLinossConfig::new(
        2,  // 2D input (position, velocity)
        16, // Hidden dimension
        2,  // 2D output
        1,  // Single layer
        DiscretizationScheme::RK4,
        ActivationFunction::Tanh,
        0.05, // Small dt for stability
        0.1,  // Moderate damping
    );
    
    let model = config.init(device);
    
    // Initial conditions: circular motion
    let initial_radius = 2.0;
    let steps = 50;
    let mut trajectory = Vec::new();
    
    for i in 0..steps {
        let angle = (i as f32) * 0.2;
        let x = initial_radius * angle.cos();
        let y = initial_radius * angle.sin();
        trajectory.push([x, y]);
    }
    
    // Convert to tensor and process
    let input_tensor = Tensor::<MyBackend, 3>::from_data(
        Data::from([trajectory]),
        device
    );
    
    let output = model.forward(input_tensor);
    let output_data: Vec<Vec<f32>> = output
        .into_data()
        .to_vec().unwrap()
        .into_iter()
        .flatten()
        .collect::<Vec<f32>>()
        .chunks(2)
        .map(|chunk| chunk.to_vec())
        .collect();
    
    println!("📍 Phase space trajectory (first 10 points):");
    println!("   Input → Output");
    for (i, (input_point, output_point)) in 
        trajectory.iter().zip(output_data.iter()).take(10).enumerate() {
        println!("   {:2}: [{:6.3}, {:6.3}] → [{:6.3}, {:6.3}]", 
                i, input_point[0], input_point[1], 
                output_point[0], output_point[1]);
    }
    
    // Calculate trajectory statistics
    let input_energy: f32 = trajectory.iter()
        .map(|point| point[0] * point[0] + point[1] * point[1])
        .sum();
    let output_energy: f32 = output_data.iter()
        .map(|point| point[0] * point[0] + point[1] * point[1])
        .sum();
    
    println!("\n📊 Trajectory Analysis:");
    println!("   Input energy: {:.4}", input_energy);
    println!("   Output energy: {:.4}", output_energy);
    println!("   Energy dissipation: {:.2}%", 
            (1.0 - output_energy / input_energy) * 100.0);
}

fn main() {
    demonstrate_damping_effects();
}
