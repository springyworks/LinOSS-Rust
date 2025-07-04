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

/// Comprehensive demonstration of D-LinOSS capabilities
pub fn showcase_dlinoss_capabilities() {
    println!("🎭 Comprehensive D-LinOSS Showcase");
    println!("=================================");
    println!("Demonstrating all production D-LinOSS features:");
    println!("• Multiple discretization schemes");
    println!("• Various activation functions");
    println!("• Learnable damping on multiple timescales");
    println!("• Energy dissipation control");
    println!("• Stable long-term dynamics\n");
    
    let device = burn_wgpu::WgpuDevice::default();
    
    // 1. Discretization Scheme Comparison
    demonstrate_discretization_schemes(&device);
    
    // 2. Activation Function Comparison
    demonstrate_activation_functions(&device);
    
    // 3. Damping Analysis
    demonstrate_damping_analysis(&device);
    
    // 4. Multi-timescale Dynamics
    demonstrate_multitimescale_dynamics(&device);
    
    // 5. Energy Conservation and Dissipation
    demonstrate_energy_dynamics(&device);
    
    println!("✨ D-LinOSS Showcase Complete!");
    println!("🚀 Ready for production neural ODE applications!");
}

fn demonstrate_discretization_schemes(device: &<MyBackend as Backend>::Device) {
    println!("1️⃣  Discretization Scheme Comparison");
    println!("   ================================");
    
    let schemes = vec![
        ("Euler", DiscretizationScheme::Euler),
        ("Midpoint", DiscretizationScheme::Midpoint),
        ("RK4", DiscretizationScheme::RK4),
    ];
    
    let mut models = Vec::new();
    for (name, scheme) in &schemes {
        let config = ProductionDLinossConfig::new(
            1, 16, 1, 2,
            scheme.clone(),
            ActivationFunction::Tanh,
            0.1, 0.1,
        );
        let model = config.init(device);
        models.push((name.to_string(), model));
    }
    
    // Test with a challenging signal (high frequency oscillation)
    let test_signal: Vec<f32> = (0..50)
        .map(|i| (0.5 * i as f32).sin() + 0.3 * (1.2 * i as f32).sin())
        .collect();
    
    let input_tensor = Tensor::<MyBackend, 3>::from_data(
        Data::from([test_signal.iter().map(|&x| [x]).collect::<Vec<_>>()]),
        device
    );
    
    println!("   Testing with complex oscillatory signal...");
    for (name, model) in &models {
        let output = model.forward(input_tensor.clone());
        let stats = calculate_stability_metrics(&output);
        
        println!("   {}: Stability={:.4}, Smoothness={:.4}, Energy={:.4}", 
                name, stats.stability, stats.smoothness, stats.energy);
    }
    println!();
}

fn demonstrate_activation_functions(device: &<MyBackend as Backend>::Device) {
    println!("2️⃣  Activation Function Effects");
    println!("   ============================");
    
    let activations = vec![
        ("ReLU", ActivationFunction::ReLU),
        ("GELU", ActivationFunction::GELU),
        ("Tanh", ActivationFunction::Tanh),
    ];
    
    let mut models = Vec::new();
    for (name, activation) in &activations {
        let config = ProductionDLinossConfig::new(
            1, 32, 1, 2,
            DiscretizationScheme::RK4,
            activation.clone(),
            0.1, 0.1,
        );
        let model = config.init(device);
        models.push((name.to_string(), model));
    }
    
    // Test with step input to see activation effects
    let step_input: Vec<f32> = (0..60)
        .map(|i| if i > 20 && i < 40 { 2.0 } else { 0.0 })
        .collect();
    
    let input_tensor = Tensor::<MyBackend, 3>::from_data(
        Data::from([step_input.iter().map(|&x| [x]).collect::<Vec<_>>()]),
        device
    );
    
    println!("   Testing with step input (amplitude=2.0)...");
    for (name, model) in &models {
        let output = model.forward(input_tensor.clone());
        let response = analyze_step_response(&output);
        
        println!("   {}: Peak={:.3}, Settle={:.3}, Overshoot={:.3}", 
                name, response.peak_value, response.settling_value, response.overshoot);
    }
    println!();
}

fn demonstrate_damping_analysis(device: &<MyBackend as Backend>::Device) {
    println!("3️⃣  Learnable Damping Analysis");
    println!("   ===========================");
    
    let damping_levels = vec![0.0, 0.05, 0.1, 0.2, 0.4];
    let mut models = Vec::new();
    
    for &damping in &damping_levels {
        let config = ProductionDLinossConfig::new(
            1, 24, 1, 3,
            DiscretizationScheme::RK4,
            ActivationFunction::GELU,
            0.1, damping,
        );
        let model = config.init(device);
        models.push((format!("γ={:.2}", damping), model));
    }
    
    // Impulse response test
    let impulse: Vec<f32> = (0..80)
        .map(|i| if i == 10 { 1.0 } else { 0.0 })
        .collect();
    
    let input_tensor = Tensor::<MyBackend, 3>::from_data(
        Data::from([impulse.iter().map(|&x| [x]).collect::<Vec<_>>()]),
        device
    );
    
    println!("   Impulse response analysis...");
    for (name, model) in &models {
        let output = model.forward(input_tensor.clone());
        let decay = analyze_impulse_decay(&output, 10);
        
        println!("   {}: Decay rate={:.4}, Final energy={:.4}", 
                name, decay.decay_rate, decay.final_energy);
    }
    println!();
}

fn demonstrate_multitimescale_dynamics(device: &<MyBackend as Backend>::Device) {
    println!("4️⃣  Multi-timescale Dynamics");
    println!("   =========================");
    
    // Create models with different dt values
    let dt_values = vec![0.05, 0.1, 0.2];
    let mut models = Vec::new();
    
    for &dt in &dt_values {
        let config = ProductionDLinossConfig::new(
            2, 32, 2, 2, // 2D system
            DiscretizationScheme::RK4,
            ActivationFunction::Tanh,
            dt, 0.1,
        );
        let model = config.init(device);
        models.push((format!("dt={:.2}", dt), model));
    }
    
    // Multi-frequency input
    let complex_signal: Vec<[f32; 2]> = (0..100)
        .map(|i| {
            let t = i as f32 * 0.1;
            [
                (0.1 * t).sin() + 0.5 * (0.5 * t).sin(), // Slow + medium freq
                0.3 * (1.5 * t).sin() + 0.2 * (3.0 * t).sin(), // Fast freqs
            ]
        })
        .collect();
    
    let input_tensor = Tensor::<MyBackend, 3>::from_data(
        Data::from([complex_signal]),
        device
    );
    
    println!("   Testing multi-frequency 2D signal...");
    for (name, model) in &models {
        let output = model.forward(input_tensor.clone());
        let spectrum = analyze_frequency_content(&output);
        
        println!("   {}: Low={:.3}, Mid={:.3}, High={:.3}", 
                name, spectrum.low_freq, spectrum.mid_freq, spectrum.high_freq);
    }
    println!();
}

fn demonstrate_energy_dynamics(device: &<MyBackend as Backend>::Device) {
    println!("5️⃣  Energy Conservation & Dissipation");
    println!("   ===================================");
    
    // Conservative vs dissipative models
    let configs = vec![
        ("Conservative (γ=0)", 0.0),
        ("Weakly Dissipative (γ=0.05)", 0.05),
        ("Strongly Dissipative (γ=0.2)", 0.2),
    ];
    
    let mut models = Vec::new();
    for (name, damping) in &configs {
        let config = ProductionDLinossConfig::new(
            2, 32, 2, 2,
            DiscretizationScheme::RK4,
            ActivationFunction::Tanh,
            0.1, *damping,
        );
        let model = config.init(device);
        models.push((name.to_string(), model));
    }
    
    // High-energy circular trajectory
    let trajectory: Vec<[f32; 2]> = (0..120)
        .map(|i| {
            let angle = i as f32 * 0.1;
            let radius = 2.0;
            [radius * angle.cos(), radius * angle.sin()]
        })
        .collect();
    
    let input_tensor = Tensor::<MyBackend, 3>::from_data(
        Data::from([trajectory]),
        device
    );
    
    let input_energy = calculate_total_energy_2d(&trajectory);
    
    println!("   Input energy: {:.4}", input_energy);
    for (name, model) in &models {
        let output = model.forward(input_tensor.clone());
        let output_energy = calculate_tensor_energy_2d(&output);
        let conservation_ratio = output_energy / input_energy;
        let dissipation = (1.0 - conservation_ratio) * 100.0;
        
        println!("   {}: Energy={:.4}, Conservation={:.3}, Dissipation={:.1}%", 
                name, output_energy, conservation_ratio, dissipation);
    }
    println!();
}

// Helper structs and functions
#[derive(Debug)]
struct StabilityMetrics {
    stability: f32,
    smoothness: f32,
    energy: f32,
}

#[derive(Debug)]
struct StepResponse {
    peak_value: f32,
    settling_value: f32,
    overshoot: f32,
}

#[derive(Debug)]
struct ImpulseDecay {
    decay_rate: f32,
    final_energy: f32,
}

#[derive(Debug)]
struct FrequencySpectrum {
    low_freq: f32,
    mid_freq: f32,
    high_freq: f32,
}

fn calculate_stability_metrics<B: Backend>(tensor: &Tensor<B, 3>) -> StabilityMetrics {
    let data: Vec<f32> = tensor.flatten::<1>(0, 2).into_data().to_vec().unwrap();
    
    // Stability: inverse of variance in second half
    let n = data.len();
    let second_half = &data[n/2..];
    let mean = second_half.iter().sum::<f32>() / second_half.len() as f32;
    let variance = second_half.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f32>() / second_half.len() as f32;
    let stability = 1.0 / (1.0 + variance);
    
    // Smoothness: inverse of total variation
    let total_variation: f32 = data.windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .sum();
    let smoothness = 1.0 / (1.0 + total_variation);
    
    // Energy
    let energy = data.iter().map(|&x| x * x).sum::<f32>();
    
    StabilityMetrics { stability, smoothness, energy }
}

fn analyze_step_response<B: Backend>(tensor: &Tensor<B, 3>) -> StepResponse {
    let data: Vec<f32> = tensor.flatten::<1>(0, 2).into_data().to_vec().unwrap();
    
    let peak_value = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let settling_value = data[data.len()*3/4..].iter().sum::<f32>() / (data.len()/4) as f32;
    let overshoot = (peak_value - settling_value).max(0.0);
    
    StepResponse { peak_value, settling_value, overshoot }
}

fn analyze_impulse_decay<B: Backend>(tensor: &Tensor<B, 3>, impulse_idx: usize) -> ImpulseDecay {
    let data: Vec<f32> = tensor.flatten::<1>(0, 2).into_data().to_vec().unwrap();
    
    if impulse_idx + 20 < data.len() {
        let response_window = &data[impulse_idx..impulse_idx + 20];
        let initial_amplitude = response_window[1].abs();
        let final_amplitude = response_window[19].abs();
        
        let decay_rate = if initial_amplitude > 1e-10 {
            -(final_amplitude / initial_amplitude).ln() / 19.0
        } else {
            0.0
        };
        
        let final_energy = data[data.len()-10..].iter()
            .map(|&x| x * x)
            .sum::<f32>();
        
        ImpulseDecay { decay_rate, final_energy }
    } else {
        ImpulseDecay { decay_rate: 0.0, final_energy: 0.0 }
    }
}

fn analyze_frequency_content<B: Backend>(tensor: &Tensor<B, 3>) -> FrequencySpectrum {
    let data: Vec<f32> = tensor.flatten::<1>(0, 2).into_data().to_vec().unwrap();
    
    // Simple frequency analysis using windowed differences
    let low_freq = data.windows(8).map(|w| {
        let mean1 = w[..4].iter().sum::<f32>() / 4.0;
        let mean2 = w[4..].iter().sum::<f32>() / 4.0;
        (mean2 - mean1).abs()
    }).sum::<f32>();
    
    let mid_freq = data.windows(4).map(|w| {
        let mean1 = w[..2].iter().sum::<f32>() / 2.0;
        let mean2 = w[2..].iter().sum::<f32>() / 2.0;
        (mean2 - mean1).abs()
    }).sum::<f32>();
    
    let high_freq = data.windows(2).map(|w| (w[1] - w[0]).abs()).sum::<f32>();
    
    FrequencySpectrum { low_freq, mid_freq, high_freq }
}

fn calculate_total_energy_2d(trajectory: &[[f32; 2]]) -> f32 {
    trajectory.iter()
        .map(|point| point[0] * point[0] + point[1] * point[1])
        .sum()
}

fn calculate_tensor_energy_2d<B: Backend>(tensor: &Tensor<B, 3>) -> f32 {
    let data: Vec<f32> = tensor.flatten::<1>(0, 2).into_data().to_vec().unwrap();
    data.chunks(2)
        .map(|chunk| chunk[0] * chunk[0] + chunk[1] * chunk[1])
        .sum()
}

fn main() {
    showcase_dlinoss_capabilities();
}
