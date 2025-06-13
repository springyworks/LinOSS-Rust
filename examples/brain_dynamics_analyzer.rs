//! Brain Dynamics Analyzer - Data Collection Version
//! 
//! This version runs the simulation in the background and logs detailed data
//! about the dynamics so we can analyze the behavior together.

use burn::prelude::*;
use linoss_rust::linoss::dlinoss_layer::{DLinossLayer, DLinossLayerConfig};
use std::{
    fs::File,
    io::Write,
    time::Instant,
};
use serde::{Deserialize, Serialize};

// --- Backend Selection ---
#[cfg(feature = "wgpu_backend")]
mod gpu_backend {
    use burn::backend::wgpu::{Wgpu, WgpuDevice};
    pub type SelectedBackend = Wgpu<f32, i32>;
    pub fn get_device() -> WgpuDevice { WgpuDevice::default() }
}

#[cfg(all(feature = "ndarray_backend", not(feature = "wgpu_backend")))]
mod cpu_backend {
    use burn::backend::{NdArray, ndarray::NdArrayDevice};
    pub type SelectedBackend = NdArray<f32>;
    pub fn get_device() -> NdArrayDevice { NdArrayDevice::default() }
}

#[cfg(not(any(feature = "wgpu_backend", feature = "ndarray_backend")))]
mod cpu_backend {
    use burn::backend::{NdArray, ndarray::NdArrayDevice};
    pub type SelectedBackend = NdArray<f32>;
    pub fn get_device() -> NdArrayDevice { NdArrayDevice::default() }
}

#[cfg(feature = "wgpu_backend")]
use gpu_backend as chosen_backend;
#[cfg(all(feature = "ndarray_backend", not(feature = "wgpu_backend")))]
use cpu_backend as chosen_backend;
#[cfg(not(any(feature = "wgpu_backend", feature = "ndarray_backend")))]
use cpu_backend as chosen_backend;

type B = chosen_backend::SelectedBackend;

// Data structures for logging
#[derive(Serialize, Deserialize, Clone)]
struct TimeStep {
    time: f64,
    regions: Vec<RegionState>,
    coupling_strength: f64,
    dlinoss_active: bool,
}

#[derive(Serialize, Deserialize, Clone)]
struct RegionState {
    name: String,
    position: (f64, f64, f64),
    velocity: (f64, f64, f64),  // dx/dt, dy/dt, dz/dt
    energy: f64,                // Simple energy measure: x² + y² + z²
    dlinoss_influence: f64,     // Magnitude of dLinOSS perturbation
}

#[derive(Clone)]
struct LorenzParams {
    sigma: f64,
    rho: f64, 
    beta: f64,
    coupling: f64,
}

impl Default for LorenzParams {
    fn default() -> Self {
        Self {
            sigma: 10.0,
            rho: 28.0,
            beta: 8.0/3.0,
            coupling: 0.1,
        }
    }
}

#[derive(Clone)]
struct BrainRegion {
    name: String,
    params: LorenzParams,
    position: (f64, f64, f64),
    prev_position: (f64, f64, f64),  // For velocity calculation
    dlinoss_model: Option<DLinossLayer<B>>,
    dlinoss_input: Option<Tensor<B, 2>>,
    last_dlinoss_magnitude: f64,
}

impl BrainRegion {
    fn new(name: String, initial_pos: (f64, f64, f64), params: LorenzParams) -> Self {
        Self {
            name,
            params,
            position: initial_pos,
            prev_position: initial_pos,
            dlinoss_model: None,
            dlinoss_input: None,
            last_dlinoss_magnitude: 0.0,
        }
    }

    fn init_dlinoss(&mut self, device: &<B as Backend>::Device) {
        let layer_config = DLinossLayerConfig::new_dlinoss(3, 8, 3);
        self.dlinoss_model = Some(DLinossLayer::new(&layer_config, device));
        self.dlinoss_input = Some(Tensor::<B, 2>::from_floats(
            [[self.position.0 as f32, self.position.1 as f32, self.position.2 as f32]], 
            device,
        ));
    }

    fn lorenz_step(&mut self, dt: f64, coupling_input: (f64, f64, f64)) {
        self.prev_position = self.position;
        
        let (x, y, z) = self.position;
        let (cx, cy, cz) = coupling_input;
        
        let dx = self.params.sigma * (y - x) + self.params.coupling * cx;
        let dy = x * (self.params.rho - z) - y + self.params.coupling * cy;
        let dz = x * y - self.params.beta * z + self.params.coupling * cz;
        
        self.position.0 += dx * dt;
        self.position.1 += dy * dt;
        self.position.2 += dz * dt;
    }

    fn dlinoss_step(&mut self) -> f64 {
        if let (Some(model), Some(input)) = (&self.dlinoss_model, &self.dlinoss_input) {
            let sequence_input = input.clone().unsqueeze_dim(1);
            let output = model.forward(sequence_input);
            let output_squeezed = output.squeeze::<2>(1);
            
            let output_data = output_squeezed.into_data();
            let output_values = output_data.to_vec::<f32>().unwrap();
            
            // Calculate magnitude of dLinOSS influence
            let magnitude = (output_values[0].powi(2) + output_values[1].powi(2) + output_values[2].powi(2)).sqrt() as f64;
            
            // Apply dLinOSS output as perturbations
            let influence_factor = 0.01;
            self.position.0 += output_values[0] as f64 * influence_factor;
            self.position.1 += output_values[1] as f64 * influence_factor;
            self.position.2 += output_values[2] as f64 * influence_factor;

            // Update dLinOSS input
            self.dlinoss_input = Some(Tensor::<B, 2>::from_floats(
                [[self.position.0 as f32, self.position.1 as f32, self.position.2 as f32]], 
                &input.device(),
            ));
            
            self.last_dlinoss_magnitude = magnitude;
            magnitude
        } else {
            0.0
        }
    }

    fn get_velocity(&self) -> (f64, f64, f64) {
        (
            self.position.0 - self.prev_position.0,
            self.position.1 - self.prev_position.1,
            self.position.2 - self.prev_position.2,
        )
    }

    fn get_energy(&self) -> f64 {
        let (x, y, z) = self.position;
        x * x + y * y + z * z
    }

    fn to_region_state(&self) -> RegionState {
        RegionState {
            name: self.name.clone(),
            position: self.position,
            velocity: self.get_velocity(),
            energy: self.get_energy(),
            dlinoss_influence: self.last_dlinoss_magnitude,
        }
    }
}

struct BrainDynamicsLogger {
    regions: Vec<BrainRegion>,
    coupling_strength: f64,
    dlinoss_active: bool,
    time_series: Vec<TimeStep>,
    simulation_time: f64,
}

impl BrainDynamicsLogger {
    fn new() -> Self {
        let device = chosen_backend::get_device();
        
        let mut regions = vec![
            BrainRegion::new(
                "Prefrontal_Cortex".to_string(),
                (1.0, 1.0, 1.0),
                LorenzParams { sigma: 10.0, rho: 28.0, beta: 8.0/3.0, coupling: 0.05 },
            ),
            BrainRegion::new(
                "Default_Mode_Network".to_string(),
                (-1.0, -1.0, -1.0),
                LorenzParams { sigma: 16.0, rho: 45.6, beta: 4.0, coupling: 0.08 },
            ),
            BrainRegion::new(
                "Thalamus".to_string(),
                (0.5, -0.5, 1.5),
                LorenzParams { sigma: 12.0, rho: 35.0, beta: 3.0, coupling: 0.12 },
            ),
        ];

        for region in &mut regions {
            region.init_dlinoss(&device);
        }

        Self {
            regions,
            coupling_strength: 0.1,
            dlinoss_active: true,
            time_series: Vec::new(),
            simulation_time: 0.0,
        }
    }

    fn step(&mut self) {
        let dt = 0.01;
        self.simulation_time += dt;

        // Calculate coupling
        let coupling_matrix = self.calculate_coupling();

        // Update each region
        for (i, region) in self.regions.iter_mut().enumerate() {
            region.lorenz_step(dt, coupling_matrix[i]);
            
            if self.dlinoss_active {
                region.dlinoss_step();
            }
        }

        // Log this timestep
        let timestep = TimeStep {
            time: self.simulation_time,
            regions: self.regions.iter().map(|r| r.to_region_state()).collect(),
            coupling_strength: self.coupling_strength,
            dlinoss_active: self.dlinoss_active,
        };
        
        self.time_series.push(timestep);
    }

    fn calculate_coupling(&self) -> Vec<(f64, f64, f64)> {
        let mut coupling = vec![(0.0, 0.0, 0.0); self.regions.len()];
        
        for (i, coupling_ref) in coupling.iter_mut().enumerate().take(self.regions.len()) {
            let mut total_coupling = (0.0, 0.0, 0.0);
            
            for j in 0..self.regions.len() {
                if i != j {
                    let weight = self.coupling_strength * self.regions[j].params.coupling;
                    total_coupling.0 += weight * self.regions[j].position.0;
                    total_coupling.1 += weight * self.regions[j].position.1;
                    total_coupling.2 += weight * self.regions[j].position.2;
                }
            }
            
            *coupling_ref = total_coupling;
        }
        
        coupling
    }

    fn run_experiment(&mut self, duration_seconds: f64, phase_name: &str) -> std::io::Result<()> {
        let dt = 0.01;
        let steps = (duration_seconds / dt) as usize;
        
        println!("🧠 Running {} phase for {:.1}s ({} steps)...", phase_name, duration_seconds, steps);
        
        let start_time = Instant::now();
        
        for step in 0..steps {
            self.step();
            
            // Print progress every 1000 steps
            if step % 1000 == 0 {
                let progress = (step as f64 / steps as f64) * 100.0;
                println!("  Progress: {:.1}% (t={:.2}s)", progress, self.simulation_time);
                
                // Print current state
                for region in &self.regions {
                    let (x, y, z) = region.position;
                    println!("    {}: ({:.3}, {:.3}, {:.3}) | dLinOSS: {:.6}", 
                             region.name, x, y, z, region.last_dlinoss_magnitude);
                }
            }
        }
        
        let elapsed = start_time.elapsed();
        println!("✅ {} phase completed in {:.2}s", phase_name, elapsed.as_secs_f64());
        
        Ok(())
    }

    fn save_data(&self, filename: &str) -> std::io::Result<()> {
        let json_data = serde_json::to_string_pretty(&self.time_series)?;
        let mut file = File::create(filename)?;
        file.write_all(json_data.as_bytes())?;
        println!("📊 Data saved to {}", filename);
        Ok(())
    }

    fn analyze_dynamics(&self) {
        println!("\n🔍 DYNAMICS ANALYSIS");
        println!("====================");
        
        if self.time_series.is_empty() {
            println!("No data to analyze!");
            return;
        }

        let total_time = self.time_series.last().unwrap().time;
        println!("Total simulation time: {:.2}s", total_time);
        println!("Number of timesteps: {}", self.time_series.len());
        
        // Analyze each region
        for (region_idx, region_name) in ["Prefrontal_Cortex", "Default_Mode_Network", "Thalamus"].iter().enumerate() {
            println!("\n📈 {} Analysis:", region_name);
            
            let positions: Vec<(f64, f64, f64)> = self.time_series.iter()
                .map(|ts| ts.regions[region_idx].position)
                .collect();
            
            let energies: Vec<f64> = self.time_series.iter()
                .map(|ts| ts.regions[region_idx].energy)
                .collect();
            
            let dlinoss_influences: Vec<f64> = self.time_series.iter()
                .map(|ts| ts.regions[region_idx].dlinoss_influence)
                .collect();
            
            // Calculate statistics
            let mean_energy = energies.iter().sum::<f64>() / energies.len() as f64;
            let max_energy = energies.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let min_energy = energies.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            
            let mean_dlinoss = dlinoss_influences.iter().sum::<f64>() / dlinoss_influences.len() as f64;
            let max_dlinoss = dlinoss_influences.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            
            println!("  Energy - Mean: {:.3}, Min: {:.3}, Max: {:.3}, Range: {:.3}", 
                     mean_energy, min_energy, max_energy, max_energy - min_energy);
            println!("  dLinOSS Influence - Mean: {:.6}, Max: {:.6}", mean_dlinoss, max_dlinoss);
            
            // Calculate trajectory spread (how much it moves around)
            let x_coords: Vec<f64> = positions.iter().map(|p| p.0).collect();
            let y_coords: Vec<f64> = positions.iter().map(|p| p.1).collect();
            let z_coords: Vec<f64> = positions.iter().map(|p| p.2).collect();
            
            let x_range = x_coords.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)) - 
                         x_coords.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let y_range = y_coords.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)) - 
                         y_coords.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let z_range = z_coords.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)) - 
                         z_coords.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            
            println!("  Trajectory Spread - X: {:.3}, Y: {:.3}, Z: {:.3}", x_range, y_range, z_range);
        }
    }
}

fn main() -> std::io::Result<()> {
    println!("🚀 Starting Brain Dynamics Analysis");
    println!("===================================");
    
    let mut logger = BrainDynamicsLogger::new();
    
    // Run different experimental phases
    println!("\n🧪 EXPERIMENT: Effect of dLinOSS on Brain Dynamics");
    
    // Phase 1: Without dLinOSS (pure Lorenz)
    logger.dlinoss_active = false;
    logger.run_experiment(5.0, "Pure Lorenz Dynamics")?;
    logger.save_data("logs/brain_dynamics_no_dlinoss.json")?;
    logger.analyze_dynamics();
    
    // Reset simulation
    logger = BrainDynamicsLogger::new();
    
    // Phase 2: With dLinOSS damping
    logger.dlinoss_active = true;
    logger.run_experiment(5.0, "dLinOSS Damped Dynamics")?;
    logger.save_data("logs/brain_dynamics_with_dlinoss.json")?;
    logger.analyze_dynamics();
    
    println!("\n🎯 EXPERIMENT COMPLETE!");
    println!("========================");
    println!("✅ Data files created:");
    println!("  📄 logs/brain_dynamics_no_dlinoss.json");
    println!("  📄 logs/brain_dynamics_with_dlinoss.json");
    println!("\n💡 Now we can analyze these together to see the difference!");
    
    Ok(())
}
