//! Burn Profiler Demo - Live Neural Dynamics with MaxGraph Visualization
//! 
//! This example demonstrates the live burn profiler integration:
//! 1. Runs a D-LinOSS neural dynamics simulation
//! 2. Streams instrumentation data via FIFO pipe  
//! 3. Serves data to MaxGraph web interface via WebSocket
//!
//! Usage:
//! 1. Run this example: `cargo run --example burn_profiler_demo`
//! 2. Run WebSocket bridge: `cargo run --bin burn_profiler_bridge`  
//! 3. Open `burn_profiler_maxgraph.html` in browser
//! 4. Watch live neural dynamics visualization!

use burn::prelude::*;
use linoss_rust::linoss::dlinoss_layer::{AParameterization, DLinossLayer, DLinossLayerConfig};
use serde::{Deserialize, Serialize};
use std::{
    fs::OpenOptions,
    io::{BufWriter, Write},
    time::Instant,
};

// --- Backend Selection ---
#[cfg(feature = "wgpu_backend")]
type MyBackend = burn::backend::wgpu::Wgpu;
#[cfg(feature = "wgpu_backend")]
type MyAutodiffBackend = burn::backend::Autodiff<MyBackend>;

#[cfg(all(feature = "ndarray_backend", not(feature = "wgpu_backend")))]
type MyBackend = burn::backend::ndarray::NdArray<f32>;
#[cfg(all(feature = "ndarray_backend", not(feature = "wgpu_backend")))]
type MyAutodiffBackend = burn::backend::Autodiff<MyBackend>;

#[cfg(not(any(feature = "wgpu_backend", feature = "ndarray_backend")))]
type MyBackend = burn::backend::ndarray::NdArray<f32>;
#[cfg(not(any(feature = "wgpu_backend", feature = "ndarray_backend")))]
type MyAutodiffBackend = burn::backend::Autodiff<MyBackend>;

// Instrumentation data structures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralInstrumentationData {
    pub timestamp: u64,
    pub simulation_time: f64,
    pub regions: Vec<RegionData>,
    pub system_stats: SystemStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionData {
    pub name: String,
    pub position: (f64, f64, f64),
    pub activity_magnitude: f64,
    pub velocity: (f64, f64, f64),
    pub dlinoss_state: DLinossStateData,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DLinossStateData {
    pub damping_coefficient: f64,
    pub oscillation_frequency: f64,
    pub energy_level: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStats {
    pub total_operations: u64,
    pub memory_usage_mb: f64,
    pub fps: f64,
    pub coupling_strength: f64,
}

/// Neural region with D-LinOSS processing
struct NeuralRegion {
    name: String,
    position: (f64, f64, f64),
    velocity: (f64, f64, f64),
    dlinoss_layer: DLinossLayer<MyAutodiffBackend>,
    internal_state: Tensor<MyAutodiffBackend, 2>,
    activity_history: Vec<f64>,
}

impl NeuralRegion {
    fn new(name: String, position: (f64, f64, f64), device: &<MyAutodiffBackend as Backend>::Device) -> Self {
        let config = DLinossLayerConfig {
            d_input: 16,        // 16 input dimensions
            d_model: 16,        // 16 model dimensions (must be even)
            d_output: 16,       // 16 output dimensions
            delta_t: 0.1,       // Time step
            init_std: 0.02,     // Initialization standard deviation
            enable_layer_norm: true,
            enable_damping: true,
            init_damping: 0.1,
            num_damping_scales: 3,
            a_parameterization: AParameterization::ReLU,
        };
        
        let dlinoss_layer = DLinossLayer::new(&config, device);
        let internal_state = Tensor::zeros([1, 16], device);
        
        Self {
            name,
            position,
            velocity: (0.0, 0.0, 0.0),
            dlinoss_layer,
            internal_state,
            activity_history: Vec::new(),
        }
    }
    
    fn process_step(&mut self, input: Tensor<MyAutodiffBackend, 2>, coupling_input: Tensor<MyAutodiffBackend, 2>) -> Tensor<MyAutodiffBackend, 2> {
        // Combine external input with coupling from other regions
        let combined_input = input + coupling_input * 0.1;
        
        // D-LinOSS expects 3D input: [batch, seq_len, features]
        // Add sequence dimension: [1, 1, 16]
        let input_3d = combined_input.unsqueeze_dim(1);
        
        // Process through D-LinOSS layer
        let output_3d = self.dlinoss_layer.forward(input_3d);
        
        // Remove sequence dimension: [1, 1, 16] -> [1, 16]
        let output = output_3d.squeeze(1);
        
        // Update internal state (simplified)
        self.internal_state = output.clone() * 0.9 + self.internal_state.clone() * 0.1;
        
        // Calculate activity magnitude
        let activity = output.clone().sum().into_scalar() as f64;
        self.activity_history.push(activity);
        if self.activity_history.len() > 100 {
            self.activity_history.remove(0);
        }
        
        // Update position based on activity (simple dynamics)
        let dt = 0.01f64;
        let activity_force = activity as f64 * 0.1;
        
        self.velocity.0 += (activity_force - self.position.0 * 0.1) * dt;
        self.velocity.1 += (activity_force * 0.8 - self.position.1 * 0.1) * dt;
        self.velocity.2 += (activity_force * 0.6 - self.position.2 * 0.1) * dt;
        
        // Apply damping
        self.velocity.0 *= 0.98;
        self.velocity.1 *= 0.98;
        self.velocity.2 *= 0.98;
        
        // Update position
        self.position.0 += self.velocity.0 * dt;
        self.position.1 += self.velocity.1 * dt;
        self.position.2 += self.velocity.2 * dt;
        
        output
    }
    
    fn get_region_data(&self) -> RegionData {
        let activity_magnitude = self.activity_history.last().copied().unwrap_or(0.0).abs();
        
        RegionData {
            name: self.name.clone(),
            position: self.position,
            activity_magnitude,
            velocity: self.velocity,
            dlinoss_state: DLinossStateData {
                damping_coefficient: 0.1, // Would extract from actual D-LinOSS layer
                oscillation_frequency: 2.0,
                energy_level: activity_magnitude.powi(2),
            },
        }
    }
}

/// Instrumentation manager for the burn profiler
struct BurnProfilerInstrumentation {
    pipe_writer: Option<BufWriter<std::fs::File>>,
    pipe_path: String,
    tensor_ops_count: u64,
    start_time: Instant,
}

impl BurnProfilerInstrumentation {
    fn new() -> Self {
        let pipe_path = "/tmp/dlinoss_brain_pipe".to_string();
        
        // Try to create and open the pipe
        let pipe_writer = Self::create_pipe(&pipe_path);
        
        if pipe_writer.is_some() {
            println!("📡 Burn profiler instrumentation pipe created: {}", pipe_path);
            println!("💡 Run 'cargo run --bin burn_profiler_bridge' to start WebSocket server");
            println!("🌐 Then open burn_profiler_maxgraph.html in browser");
        } else {
            println!("⚠️ Could not create instrumentation pipe, data will be logged to console only");
        }
        
        Self {
            pipe_writer,
            pipe_path,
            tensor_ops_count: 0,
            start_time: Instant::now(),
        }
    }
    
    fn create_pipe(pipe_path: &str) -> Option<BufWriter<std::fs::File>> {
        // Remove existing pipe if it exists
        let _ = std::fs::remove_file(pipe_path);
        
        // Create named pipe
        let output = std::process::Command::new("mkfifo")
            .arg(pipe_path)
            .output();
            
        match output {
            Ok(result) if result.status.success() => {
                // Try to open pipe for writing
                match OpenOptions::new().write(true).open(pipe_path) {
                    Ok(file) => Some(BufWriter::new(file)),
                    Err(e) => {
                        eprintln!("Failed to open pipe for writing: {}", e);
                        None
                    }
                }
            }
            _ => {
                eprintln!("Failed to create named pipe with mkfifo");
                None
            }
        }
    }
    
    fn record_data(&mut self, regions: &[NeuralRegion]) {
        self.tensor_ops_count += regions.len() as u64 * 10; // Estimate ops per region
        
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
            
        let simulation_time = self.start_time.elapsed().as_secs_f64();
        
        let region_data: Vec<RegionData> = regions.iter().map(|r| r.get_region_data()).collect();
        
        let system_stats = SystemStats {
            total_operations: self.tensor_ops_count,
            memory_usage_mb: 45.0 + (simulation_time * 0.1).sin() * 5.0, // Mock memory usage
            fps: 100.0, // Fixed FPS for this demo
            coupling_strength: 0.1 + 0.05 * (simulation_time * 0.3).sin(),
        };
        
        let instrumentation_data = NeuralInstrumentationData {
            timestamp,
            simulation_time,
            regions: region_data,
            system_stats: system_stats.clone(),
        };
        
        // Write to pipe if available
        if let Some(ref mut writer) = self.pipe_writer {
            match serde_json::to_string(&instrumentation_data) {
                Ok(json_data) => {
                    if let Err(e) = writeln!(writer, "{}", json_data) {
                        eprintln!("Failed to write to instrumentation pipe: {}", e);
                        self.pipe_writer = None; // Disable pipe writing
                    } else {
                        let _ = writer.flush(); // Force immediate write
                    }
                }
                Err(e) => {
                    eprintln!("Failed to serialize instrumentation data: {}", e);
                }
            }
        }
        
        // Console logging for verification
        if (timestamp / 1000) % 2 == 0 { // Log every 2 seconds
            println!(
                "🧠 t={:.1}s | Regions: {} | Ops: {} | Memory: {:.1}MB",
                simulation_time,
                instrumentation_data.regions.len(),
                self.tensor_ops_count,
                system_stats.memory_usage_mb
            );
            
            for region in &instrumentation_data.regions {
                println!(
                    "   📍 {}: pos=({:.2},{:.2},{:.2}) activity={:.3} velocity=({:.3},{:.3},{:.3})",
                    region.name,
                    region.position.0, region.position.1, region.position.2,
                    region.activity_magnitude,
                    region.velocity.0, region.velocity.1, region.velocity.2
                );
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔥 Burn Profiler Demo - Live Neural Dynamics");
    println!("==============================================");
    println!("Running D-LinOSS neural simulation with live instrumentation...");
    
    let device = <MyAutodiffBackend as Backend>::Device::default();
    
    // Create neural regions with different D-LinOSS configurations
    let mut regions = vec![
        NeuralRegion::new("prefrontal".to_string(), (1.0, 0.0, 0.0), &device),
        NeuralRegion::new("dmn".to_string(), (-0.5, 0.866, 0.0), &device),
        NeuralRegion::new("thalamus".to_string(), (-0.5, -0.866, 0.0), &device),
    ];
    
    // Initialize instrumentation
    let mut instrumentation = BurnProfilerInstrumentation::new();
    
    println!("\n🎮 Starting neural dynamics simulation...");
    println!("Press Ctrl+C to stop\n");
    
    // Simulation loop
    let mut step = 0;
    let dt = 0.01; // 10ms timesteps = 100 FPS
    
    loop {
        let time = step as f64 * dt;
        
        // Generate input stimuli for each region
        let inputs: Vec<Tensor<MyAutodiffBackend, 2>> = regions.iter().enumerate().map(|(i, _)| {
            let phase = i as f64 * std::f64::consts::PI * 2.0 / 3.0;
            let stimulus_strength = 0.5 + 0.3 * (time * 0.5 + phase).sin();
            
            // Create random input tensor
            Tensor::random([1, 16], burn::tensor::Distribution::Normal(stimulus_strength, 0.1), &device)
        }).collect();
        
        // Calculate coupling between regions
        let mut coupling_inputs = vec![Tensor::zeros([1, 16], &device); regions.len()];
        
        for i in 0..regions.len() {
            for j in 0..regions.len() {
                if i != j {
                    // Simple coupling based on distance and activity
                    let distance = {
                        let pos_i = regions[i].position;
                        let pos_j = regions[j].position;
                        ((pos_i.0 - pos_j.0).powi(2) + (pos_i.1 - pos_j.1).powi(2) + (pos_i.2 - pos_j.2).powi(2)).sqrt()
                    };
                    
                    let coupling_strength = 0.1 * (-distance).exp();
                    let activity_j = regions[j].activity_history.last().copied().unwrap_or(0.0);
                    
                    let coupling_signal = Tensor::full([1, 16], activity_j * coupling_strength, &device);
                    coupling_inputs[i] = coupling_inputs[i].clone() + coupling_signal;
                }
            }
        }
        
        // Process each region
        for (i, region) in regions.iter_mut().enumerate() {
            let _output = region.process_step(inputs[i].clone(), coupling_inputs[i].clone());
        }
        
        // Record instrumentation data every 10 steps (10 FPS for visualization)
        if step % 10 == 0 {
            instrumentation.record_data(&regions);
        }
        
        step += 1;
        
        // Sleep to maintain real-time simulation
        std::thread::sleep(std::time::Duration::from_millis(10));
        
        // Stop after 2 minutes for demo
        if step > 12000 {
            break;
        }
    }
    
    println!("\n✅ Neural dynamics simulation completed!");
    println!("📊 Total tensor operations: {}", instrumentation.tensor_ops_count);
    println!("⏱️ Simulation time: {:.1}s", instrumentation.start_time.elapsed().as_secs_f64());
    
    Ok(())
}
