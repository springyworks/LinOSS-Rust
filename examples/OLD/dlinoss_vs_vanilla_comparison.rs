use burn::{
    module::Module,
    tensor::{backend::Backend, Data, Tensor},
};
use burn_wgpu::WgpuBackend;

use linoss_rust::linoss::production_dlinoss::{
    ProductionDLinossModel, ProductionDLinossConfig, 
    ProductionVanillaLinossModel, ProductionVanillaLinossConfig,
    DiscretizationScheme, ActivationFunction,
};

type MyBackend = WgpuBackend<f32, i32>;

/// Compare D-LinOSS (with damping) against vanilla LinOSS (no damping)
pub fn compare_dlinoss_vs_vanilla() {
    println!("⚡ D-LinOSS vs Vanilla LinOSS Comparison");
    println!("======================================");
    
    let device = burn_wgpu::WgpuDevice::default();
    
    // Model parameters
    let d_model = 32;
    let num_layers = 2;
    let input_size = 1;
    let output_size = 1;
    let dt = 0.1;
    
    // Create vanilla LinOSS (no damping)
    println!("🔧 Creating Vanilla LinOSS model...");
    let vanilla_config = ProductionVanillaLinossConfig::new(
        input_size,
        d_model,
        output_size,
        num_layers,
        DiscretizationScheme::RK4,
        ActivationFunction::GELU,
        dt,
    );
    let vanilla_model = vanilla_config.init(&device);
    
    // Create D-LinOSS models with different damping levels
    let damping_configs = vec![
        ("D-LinOSS (Light Damping)", 0.05),
        ("D-LinOSS (Medium Damping)", 0.15),
        ("D-LinOSS (Heavy Damping)", 0.3),
    ];
    
    let mut dlinoss_models = Vec::new();
    for (name, damping_factor) in &damping_configs {
        println!("🔧 Creating {} model (damping={:.2})...", name, damping_factor);
        let config = ProductionDLinossConfig::new(
            input_size,
            d_model,
            output_size,
            num_layers,
            DiscretizationScheme::RK4,
            ActivationFunction::GELU,
            dt,
            *damping_factor,
        );
        let model = config.init(&device);
        dlinoss_models.push((name.to_string(), model));
    }
    
    // Test signals
    let test_cases = vec![
        ("Step Input", generate_step_input(100)),
        ("Impulse Input", generate_impulse_input(100)),
        ("Sine Wave", generate_sine_input(100, 0.2)),
        ("Chirp Signal", generate_chirp_input(100)),
        ("Random Walk", generate_random_walk(100)),
    ];
    
    println!("\n🧪 Running comparison tests...\n");
    
    for (test_name, input_data) in &test_cases {
        println!("📊 Test Case: {}", test_name);
        
        // Convert to tensor
        let input_tensor = Tensor::<MyBackend, 3>::from_data(
            Data::from([input_data.iter().map(|&x| [x]).collect::<Vec<_>>()]),
            &device
        );
        
        // Test vanilla LinOSS
        let vanilla_output = vanilla_model.forward(input_tensor.clone());
        let vanilla_stats = calculate_signal_stats(&vanilla_output);
        
        println!("   Vanilla LinOSS:");
        print_signal_stats(&vanilla_stats, "     ");
        
        // Test D-LinOSS variants
        for (model_name, model) in &dlinoss_models {
            let output = model.forward(input_tensor.clone());
            let stats = calculate_signal_stats(&output);
            
            println!("   {}:", model_name);
            print_signal_stats(&stats, "     ");
            
            // Compare with vanilla
            let energy_reduction = (vanilla_stats.energy - stats.energy) / vanilla_stats.energy * 100.0;
            let variance_reduction = (vanilla_stats.variance - stats.variance) / vanilla_stats.variance * 100.0;
            
            println!("     Relative to Vanilla: Energy ↓{:.1}%, Variance ↓{:.1}%", 
                    energy_reduction, variance_reduction);
        }
        println!();
    }
    
    // Demonstrate long-term stability
    println!("🔬 Long-term Stability Analysis:");
    demonstrate_long_term_stability(&vanilla_model, &dlinoss_models, &device);
    
    // Demonstrate frequency response
    println!("\n🎵 Frequency Response Analysis:");
    demonstrate_frequency_response(&vanilla_model, &dlinoss_models, &device);
}

#[derive(Debug)]
struct SignalStats {
    mean: f32,
    variance: f32,
    std_dev: f32,
    energy: f32,
    max_abs: f32,
    peak_to_peak: f32,
}

fn calculate_signal_stats<B: Backend>(tensor: &Tensor<B, 3>) -> SignalStats {
    let data: Vec<f32> = tensor
        .flatten::<1>(0, 2)
        .into_data()
        .to_vec().unwrap();
    
    let n = data.len() as f32;
    let mean = data.iter().sum::<f32>() / n;
    let variance = data.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f32>() / n;
    let std_dev = variance.sqrt();
    let energy = data.iter().map(|&x| x * x).sum::<f32>();
    let max_abs = data.iter().map(|&x| x.abs()).fold(0.0, f32::max);
    let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let peak_to_peak = max_val - min_val;
    
    SignalStats {
        mean,
        variance,
        std_dev,
        energy,
        max_abs,
        peak_to_peak,
    }
}

fn print_signal_stats(stats: &SignalStats, indent: &str) {
    println!("{}Mean: {:.4}, Std: {:.4}, Energy: {:.4}", 
            indent, stats.mean, stats.std_dev, stats.energy);
    println!("{}Max |x|: {:.4}, Peak-to-Peak: {:.4}", 
            indent, stats.max_abs, stats.peak_to_peak);
}

// Signal generators
fn generate_step_input(length: usize) -> Vec<f32> {
    (0..length)
        .map(|i| if i > length / 4 { 1.0 } else { 0.0 })
        .collect()
}

fn generate_impulse_input(length: usize) -> Vec<f32> {
    (0..length)
        .map(|i| if i == length / 4 { 1.0 } else { 0.0 })
        .collect()
}

fn generate_sine_input(length: usize, frequency: f32) -> Vec<f32> {
    (0..length)
        .map(|i| (frequency * i as f32).sin())
        .collect()
}

fn generate_chirp_input(length: usize) -> Vec<f32> {
    (0..length)
        .map(|i| {
            let t = i as f32 / length as f32;
            let freq = 0.1 + 0.4 * t; // Frequency increases linearly
            (2.0 * std::f32::consts::PI * freq * i as f32).sin()
        })
        .collect()
}

fn generate_random_walk(length: usize) -> Vec<f32> {
    let mut value = 0.0;
    let mut result = Vec::with_capacity(length);
    
    for i in 0..length {
        let step = (((i * 17) % 100) as f32 / 100.0 - 0.5) * 0.2;
        value += step;
        result.push(value);
    }
    
    result
}

fn demonstrate_long_term_stability(
    vanilla_model: &ProductionVanillaLinossModel<MyBackend>,
    dlinoss_models: &[(String, ProductionDLinossModel<MyBackend>)],
    device: &<MyBackend as Backend>::Device,
) {
    println!("   Testing with long sequence (200 steps)...");
    
    // Generate a long noisy sine wave
    let long_input: Vec<f32> = (0..200)
        .map(|i| {
            let signal = (0.1 * i as f32).sin();
            let noise = (((i * 13) % 100) as f32 / 100.0 - 0.5) * 0.1;
            signal + noise
        })
        .collect();
    
    let input_tensor = Tensor::<MyBackend, 3>::from_data(
        Data::from([long_input.iter().map(|&x| [x]).collect::<Vec<_>>()]),
        device
    );
    
    // Test stability
    let vanilla_output = vanilla_model.forward(input_tensor.clone());
    let vanilla_final_energy = calculate_final_window_energy(&vanilla_output, 20);
    
    println!("     Vanilla LinOSS final energy (last 20 steps): {:.4}", vanilla_final_energy);
    
    for (name, model) in dlinoss_models {
        let output = model.forward(input_tensor.clone());
        let final_energy = calculate_final_window_energy(&output, 20);
        let stability_ratio = final_energy / vanilla_final_energy;
        
        println!("     {} final energy: {:.4} (ratio: {:.3})", 
                name, final_energy, stability_ratio);
    }
}

fn calculate_final_window_energy<B: Backend>(tensor: &Tensor<B, 3>, window_size: usize) -> f32 {
    let data: Vec<f32> = tensor
        .flatten::<1>(0, 2)
        .into_data()
        .to_vec().unwrap();
    
    let start_idx = data.len().saturating_sub(window_size);
    data[start_idx..]
        .iter()
        .map(|&x| x * x)
        .sum()
}

fn demonstrate_frequency_response(
    vanilla_model: &ProductionVanillaLinossModel<MyBackend>,
    dlinoss_models: &[(String, ProductionDLinossModel<MyBackend>)],
    device: &<MyBackend as Backend>::Device,
) {
    let frequencies = vec![0.05, 0.1, 0.2, 0.4, 0.8];
    
    for freq in frequencies {
        println!("   Frequency {:.2} Hz:", freq);
        
        let sine_input: Vec<f32> = (0..100)
            .map(|i| (freq * i as f32).sin())
            .collect();
        
        let input_tensor = Tensor::<MyBackend, 3>::from_data(
            Data::from([sine_input.iter().map(|&x| [x]).collect::<Vec<_>>()]),
            device
        );
        
        let vanilla_output = vanilla_model.forward(input_tensor.clone());
        let vanilla_gain = calculate_gain_ratio(&vanilla_output, &sine_input);
        
        println!("     Vanilla gain: {:.3}", vanilla_gain);
        
        for (name, model) in dlinoss_models {
            let output = model.forward(input_tensor.clone());
            let gain = calculate_gain_ratio(&output, &sine_input);
            let gain_ratio = gain / vanilla_gain;
            
            println!("     {} gain: {:.3} (ratio: {:.3})", 
                    name, gain, gain_ratio);
        }
    }
}

fn calculate_gain_ratio<B: Backend>(output_tensor: &Tensor<B, 3>, input_signal: &[f32]) -> f32 {
    let output_data: Vec<f32> = output_tensor
        .flatten::<1>(0, 2)
        .into_data()
        .to_vec().unwrap();
    
    // Calculate RMS of input and output
    let input_rms = (input_signal.iter().map(|&x| x * x).sum::<f32>() / input_signal.len() as f32).sqrt();
    let output_rms = (output_data.iter().map(|&x| x * x).sum::<f32>() / output_data.len() as f32).sqrt();
    
    if input_rms > 1e-10 {
        output_rms / input_rms
    } else {
        0.0
    }
}

fn main() {
    compare_dlinoss_vs_vanilla();
}
