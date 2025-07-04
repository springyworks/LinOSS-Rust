//! Pure dLinOSS Brain Dynamics - No RK4, Only Neural State-Space Models
//!
//! This version uses three stacked dLinOSS blocks to model brain dynamics with different
//! A matrix parameterizations based on Proposition 3.1 from the LinOSS paper:
//!
//! ## A Matrix Parameterizations (for stability: diagonal weights must be non-negative):
//! - **Block 1 (Fast)**: ReLU parameterization `A = ReLU(A_hat)` - allows complete dimension switch-off
//! - **Block 2 (Medium)**: GELU parameterization `A = GELU(A_hat)` - smooth activation
//! - **Block 3 (Slow)**: Squared parameterization `A = A_hat ⊙ A_hat` - guaranteed non-negativity
//!
//! ## Model Structure:
//! - Each block represents a different timescale/frequency of neural activity
//! - GLU and ReLU/GELU activations provide nonlinearity between blocks
//! - Damping can be controlled per block for energy dissipation
//! - Based on: "Learning to Dissipate Energy in Oscillatory State-Space Models" (dLinOSS paper)
//!            and "Oscillatory State-Space Models" (LinOSS paper)
//!
//! ## Ultra-Low Latency Streaming:
//! - FIFO pipes have kernel buffering (up to 1MB) causing latency
//! - Non-blocking I/O with O_NONBLOCK for immediate writes
//! - Frame dropping when buffer full to maintain real-time performance
//! - Optional: Shared memory mapping for zero-copy streaming

use burn::prelude::*;
use linoss_rust::linoss::dlinoss_layer::{AParameterization, DLinossLayer, DLinossLayerConfig};
use ratatui::{
    prelude::{Constraint, CrosstermBackend, Direction, Frame, Layout},
    style::{Color, Style},
    symbols,
    widgets::{Block as TuiBlock, Borders, Chart, Dataset, List, ListItem, Paragraph},
    Terminal,
};
use std::{
    collections::VecDeque,
    error::Error,
    io::{self, Stdout},
    time::{Duration, Instant},
};

use crossterm::{
    event::{self, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};

use log::{debug, info, warn};
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};

// --- Backend Selection ---
#[cfg(feature = "wgpu_backend")]
mod gpu_backend {
    use burn::backend::wgpu::{Wgpu, WgpuDevice};
    pub type SelectedBackend = Wgpu<f32, i32>;
    pub fn get_device() -> WgpuDevice {
        WgpuDevice::default()
    }
}

#[cfg(all(feature = "ndarray_backend", not(feature = "wgpu_backend")))]
mod cpu_backend {
    use burn::backend::{ndarray::NdArrayDevice, NdArray};
    pub type SelectedBackend = NdArray<f32>;
    pub fn get_device() -> NdArrayDevice {
        NdArrayDevice::default()
    }
}

#[cfg(not(any(feature = "wgpu_backend", feature = "ndarray_backend")))]
mod cpu_backend {
    use burn::backend::{ndarray::NdArrayDevice, NdArray};
    pub type SelectedBackend = NdArray<f32>;
    pub fn get_device() -> NdArrayDevice {
        NdArrayDevice::default()
    }
}

#[cfg(all(feature = "ndarray_backend", not(feature = "wgpu_backend")))]
use cpu_backend as chosen_backend;
#[cfg(not(any(feature = "wgpu_backend", feature = "ndarray_backend")))]
use cpu_backend as chosen_backend;
#[cfg(feature = "wgpu_backend")]
use gpu_backend as chosen_backend;

type B = chosen_backend::SelectedBackend;

// Constants for neural dynamics
const MAX_TRAJECTORY_POINTS: usize = 2000;
const SIMULATION_SPEED: u64 = 30;
const _STATE_DIM: usize = 3; // x, y, z coordinates
const HIDDEN_DIM: usize = 16; // Hidden dimension for dLinOSS
const GROUP_SIZE: usize = 3; // Each input/output group has 3 signals
const NUM_GROUPS: usize = 2; // gri1/gri2 and gro1/gro2
const TOTAL_INTER_SIGNALS: usize = GROUP_SIZE * NUM_GROUPS; // 6 signals total
const CANVAS_X_BOUNDS: [f64; 2] = [-50.0, 50.0];
const CANVAS_Y_BOUNDS: [f64; 2] = [-50.0, 50.0];

// Position interpretation:
// X-axis: Excitatory/Inhibitory balance (negative = more inhibitory, positive = more excitatory)
// Y-axis: Synchronization level (negative = desynchronized, positive = synchronized)
// Z-axis: Oscillation amplitude (larger = stronger oscillations)

// Instrumentation constants
const LOG_FILE_PATH: &str = "/tmp/dlinoss_brain_dynamics.log";
const DATA_PIPE_PATH: &str = "/tmp/dlinoss_brain_pipe";
const INSTRUMENTATION_INTERVAL: usize = 10; // Log every 10 iterations to reduce overhead

/// Instrumentation data structure for neural dynamics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralInstrumentationData {
    pub timestamp: f64,
    pub simulation_time: f64,
    pub regions: Vec<RegionData>,
    pub coupling_matrix: Vec<(f64, f64, f64)>,
    pub system_stats: SystemStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionData {
    pub name: String,
    pub position: (f64, f64, f64),
    pub activity_magnitude: f64,
    pub trajectory_length: usize,
    pub velocity: (f64, f64, f64),
    pub dlinoss_state: DLinossStateData,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DLinossStateData {
    pub fast_block_activity: f64,
    pub medium_block_activity: f64,
    pub slow_block_activity: f64,
    pub coupling_strength: f64,
    pub damping_factor: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStats {
    pub total_energy: f64,
    pub coupling_strength: f64,
    pub damping_enabled: bool,
    pub paused: bool,
    pub bounds: ((f64, f64), (f64, f64)), // ((x_min, x_max), (y_min, y_max))
}

/// Enhanced dLinOSS configuration with parameterization options
#[derive(Debug, Clone)]
pub struct EnhancedDLinossConfig {
    pub d_input: usize,
    pub d_model: usize,
    pub d_output: usize,
    pub delta_t: f64,
    pub init_std: f64,
    pub enable_layer_norm: bool,
    pub enable_damping: bool,
    pub init_damping: f64,
    pub num_damping_scales: usize,
    pub a_parameterization: AParameterization,
}

impl EnhancedDLinossConfig {
    /// Convert to standard DLinossLayerConfig
    pub fn to_standard_config(&self) -> DLinossLayerConfig {
        DLinossLayerConfig {
            d_input: self.d_input,
            d_model: self.d_model,
            d_output: self.d_output,
            delta_t: self.delta_t,
            init_std: self.init_std,
            enable_layer_norm: self.enable_layer_norm,
            enable_damping: self.enable_damping,
            init_damping: self.init_damping,
            num_damping_scales: self.num_damping_scales,
            a_parameterization: self.a_parameterization.clone(),
        }
    }
}

/// Enhanced dLinOSS block with grouped inputs/outputs
#[derive(Module, Debug)]
pub struct GroupedDLinossBlock<B: Backend> {
    // Core dLinOSS layer
    dlinoss_core: DLinossLayer<B>,
    
    // Input processing: combine gri1 (3) + gri2 (3) + local state (3) = 9 → hidden_dim
    input_processor: burn::nn::Linear<B>,
    
    // Output generation: hidden_dim → gro1 (3) + gro2 (3) + position (3) = 9
    output_generator: burn::nn::Linear<B>,
}

impl<B: Backend> GroupedDLinossBlock<B> {
    pub fn new(config: &DLinossLayerConfig, device: &B::Device) -> Self {
        let dlinoss_core = DLinossLayer::new(config, device);
        
        // Input: gri1(3) + gri2(3) + local_state(3) = 9 signals
        let input_processor = burn::nn::LinearConfig::new(3 * GROUP_SIZE, HIDDEN_DIM).init(device);
        
        // Output: gro1(3) + gro2(3) + position(3) = 9 signals  
        let output_generator = burn::nn::LinearConfig::new(HIDDEN_DIM, 3 * GROUP_SIZE).init(device);
        
        Self {
            dlinoss_core,
            input_processor,
            output_generator,
        }
    }
    
    /// Forward pass: (gri1, gri2, local_state) → (gro1, gro2, new_position)
    pub fn forward(&self, gri1: Tensor<B, 2>, gri2: Tensor<B, 2>, local_state: Tensor<B, 2>) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
        // Combine all inputs: gri1 + gri2 + local_state
        let combined_input = Tensor::cat(vec![gri1, gri2, local_state], 1);
        
        // Process through input layer
        let processed_input = self.input_processor.forward(combined_input);
        
        // Add sequence dimension for dLinOSS
        let dlinoss_input = processed_input.unsqueeze_dim(1);
        
        // Core dLinOSS processing
        let dlinoss_output = self.dlinoss_core.forward(dlinoss_input);
        let dlinoss_squeezed = dlinoss_output.squeeze::<2>(1);
        
        // Generate outputs
        let output_signals = self.output_generator.forward(dlinoss_squeezed);
        
        // Split into gro1(3), gro2(3), position(3)
        let gro1 = output_signals.clone().narrow(1, 0, GROUP_SIZE);
        let gro2 = output_signals.clone().narrow(1, GROUP_SIZE, GROUP_SIZE);
        let new_position = output_signals.narrow(1, 2 * GROUP_SIZE, GROUP_SIZE);
        
        (gro1, gro2, new_position)
    }
}

/// Fully connected 3-block brain model with gri1/gri2 → gro1/gro2 connectivity
#[derive(Module, Debug)]
pub struct DLinossBrainModel<B: Backend> {
    // Three enhanced dLinOSS blocks
    block1_fast: GroupedDLinossBlock<B>,      // Fast oscillations (gamma/beta waves)
    block2_medium: GroupedDLinossBlock<B>,    // Medium oscillations (alpha waves)  
    block3_slow: GroupedDLinossBlock<B>,      // Slow oscillations (theta/delta waves)
    
    // Full connectivity matrix: each block's outputs (gro1+gro2) → all blocks' inputs (gri1+gri2)
    // This creates a 6x6 connection matrix for each pair of blocks
    conn_matrix_1_to_2: burn::nn::Linear<B>,  // Block1 outputs → Block2 inputs
    conn_matrix_1_to_3: burn::nn::Linear<B>,  // Block1 outputs → Block3 inputs
    conn_matrix_2_to_1: burn::nn::Linear<B>,  // Block2 outputs → Block1 inputs
    conn_matrix_2_to_3: burn::nn::Linear<B>,  // Block2 outputs → Block3 inputs
    conn_matrix_3_to_1: burn::nn::Linear<B>,  // Block3 outputs → Block1 inputs
    conn_matrix_3_to_2: burn::nn::Linear<B>,  // Block3 outputs → Block2 inputs
}

impl<B: Backend> DLinossBrainModel<B> {
    pub fn new(device: &B::Device) -> Self {
        // Block 1: Fast oscillations (40-100 Hz, gamma waves) with ReLU parameterization
        let config1 = EnhancedDLinossConfig {
            d_input: HIDDEN_DIM,  // Takes processed input from input_processor
            d_model: HIDDEN_DIM,
            d_output: HIDDEN_DIM,
            delta_t: 0.01, // Small time step for fast dynamics
            init_std: 0.02,
            enable_layer_norm: true,
            enable_damping: true,
            init_damping: 0.05, // Light damping for fast oscillations
            num_damping_scales: 2,
            a_parameterization: AParameterization::ReLU,
        };

        // Block 2: Medium oscillations (8-30 Hz, alpha/beta waves) with GELU parameterization
        let config2 = EnhancedDLinossConfig {
            d_input: HIDDEN_DIM,
            d_model: HIDDEN_DIM,
            d_output: HIDDEN_DIM,
            delta_t: 0.05, // Medium time step
            init_std: 0.02,
            enable_layer_norm: true,
            enable_damping: true,
            init_damping: 0.1, // Medium damping
            num_damping_scales: 3,
            a_parameterization: AParameterization::GELU,
        };

        // Block 3: Slow oscillations (1-8 Hz, theta/delta waves) with Squared parameterization
        let config3 = EnhancedDLinossConfig {
            d_input: HIDDEN_DIM,
            d_model: HIDDEN_DIM,
            d_output: HIDDEN_DIM,
            delta_t: 0.1, // Large time step for slow dynamics
            init_std: 0.02,
            enable_layer_norm: true,
            enable_damping: true,
            init_damping: 0.2, // Strong damping for slow waves
            num_damping_scales: 4,
            a_parameterization: AParameterization::Squared,
        };

        // Create the three grouped dLinOSS blocks
        let block1_fast = GroupedDLinossBlock::new(&config1.to_standard_config(), device);
        let block2_medium = GroupedDLinossBlock::new(&config2.to_standard_config(), device);
        let block3_slow = GroupedDLinossBlock::new(&config3.to_standard_config(), device);

        // Full connectivity matrices: 6 outputs → 6 inputs for each block pair
        // Each block outputs gro1(3) + gro2(3) = 6 signals
        // Each block inputs gri1(3) + gri2(3) = 6 signals
        let conn_matrix_1_to_2 = burn::nn::LinearConfig::new(TOTAL_INTER_SIGNALS, TOTAL_INTER_SIGNALS).init(device);
        let conn_matrix_1_to_3 = burn::nn::LinearConfig::new(TOTAL_INTER_SIGNALS, TOTAL_INTER_SIGNALS).init(device);
        let conn_matrix_2_to_1 = burn::nn::LinearConfig::new(TOTAL_INTER_SIGNALS, TOTAL_INTER_SIGNALS).init(device);
        let conn_matrix_2_to_3 = burn::nn::LinearConfig::new(TOTAL_INTER_SIGNALS, TOTAL_INTER_SIGNALS).init(device);
        let conn_matrix_3_to_1 = burn::nn::LinearConfig::new(TOTAL_INTER_SIGNALS, TOTAL_INTER_SIGNALS).init(device);
        let conn_matrix_3_to_2 = burn::nn::LinearConfig::new(TOTAL_INTER_SIGNALS, TOTAL_INTER_SIGNALS).init(device);

        Self {
            block1_fast,
            block2_medium,
            block3_slow,
            conn_matrix_1_to_2,
            conn_matrix_1_to_3,
            conn_matrix_2_to_1,
            conn_matrix_2_to_3,
            conn_matrix_3_to_1,
            conn_matrix_3_to_2,
        }
    }

    /// Forward pass with full bidirectional connectivity between grouped dLinOSS blocks
    /// Each block has gri1(3) + gri2(3) inputs and gro1(3) + gro2(3) outputs
    /// All gro outputs connect to all gri inputs via 6x6 connectivity matrices
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let batch_size = input.dims()[0];
        let device = input.device();
        
        // Initialize inter-block signals (6 signals per block pair)
        // Start with zero signals - they'll be computed from previous outputs
        let _zeros_6: Tensor<B, 2> = Tensor::zeros([batch_size, TOTAL_INTER_SIGNALS], &device);
        let zeros_3: Tensor<B, 2> = Tensor::zeros([batch_size, GROUP_SIZE], &device);
        
        // Initial forward pass with zero inter-block signals
        let (gro1_1, gro2_1, _pos1) = self.block1_fast.forward(zeros_3.clone(), zeros_3.clone(), input.clone());
        let (gro1_2, gro2_2, _pos2) = self.block2_medium.forward(zeros_3.clone(), zeros_3.clone(), input.clone());
        let (gro1_3, gro2_3, _pos3) = self.block3_slow.forward(zeros_3.clone(), zeros_3.clone(), input.clone());
        
        // Combine outputs for inter-block connectivity
        let block1_outputs = Tensor::cat(vec![gro1_1, gro2_1], 1); // 6 signals
        let block2_outputs = Tensor::cat(vec![gro1_2, gro2_2], 1); // 6 signals  
        let block3_outputs = Tensor::cat(vec![gro1_3, gro2_3], 1); // 6 signals
        
        // Apply full connectivity matrices: each block's 6 outputs → each block's 6 inputs
        let signals_1_to_2 = self.conn_matrix_1_to_2.forward(block1_outputs.clone());
        let signals_1_to_3 = self.conn_matrix_1_to_3.forward(block1_outputs.clone());
        let signals_2_to_1 = self.conn_matrix_2_to_1.forward(block2_outputs.clone());
        let signals_2_to_3 = self.conn_matrix_2_to_3.forward(block2_outputs.clone());
        let signals_3_to_1 = self.conn_matrix_3_to_1.forward(block3_outputs.clone());
        let signals_3_to_2 = self.conn_matrix_3_to_2.forward(block3_outputs.clone());
        
        // Sum incoming signals for each block (bidirectional connectivity)
        let inputs_to_1 = signals_2_to_1 + signals_3_to_1; // From blocks 2&3 → block 1
        let inputs_to_2 = signals_1_to_2 + signals_3_to_2; // From blocks 1&3 → block 2  
        let inputs_to_3 = signals_1_to_3 + signals_2_to_3; // From blocks 1&2 → block 3
        
        // Split 6-signal inputs into gri1(3) + gri2(3) for each block
        let (gri1_1, gri2_1) = (inputs_to_1.clone().narrow(1, 0, GROUP_SIZE), inputs_to_1.narrow(1, GROUP_SIZE, GROUP_SIZE));
        let (gri1_2, gri2_2) = (inputs_to_2.clone().narrow(1, 0, GROUP_SIZE), inputs_to_2.narrow(1, GROUP_SIZE, GROUP_SIZE));
        let (gri1_3, gri2_3) = (inputs_to_3.clone().narrow(1, 0, GROUP_SIZE), inputs_to_3.narrow(1, GROUP_SIZE, GROUP_SIZE));
        
        // Second forward pass with inter-block signals
        let (_, _, final_pos1) = self.block1_fast.forward(gri1_1, gri2_1, input.clone());
        let (_, _, final_pos2) = self.block2_medium.forward(gri1_2, gri2_2, input.clone());
        let (_, _, final_pos3) = self.block3_slow.forward(gri1_3, gri2_3, input.clone());
        
        // Combine final positions (could be weighted combination)
        (final_pos1 + final_pos2 + final_pos3) / 3.0
    }
}

/// Brain region with pure dLinOSS dynamics (no RK4)
#[derive(Clone)]
struct PureBrainRegion {
    name: String,
    position: (f64, f64, f64),
    previous_position: (f64, f64, f64),
    velocity: (f64, f64, f64),
    trajectory: VecDeque<(f64, f64)>,
    color: Color,
    dlinoss_model: Option<DLinossBrainModel<B>>,

    // Neural activity states for each frequency band
    fast_state: Option<Tensor<B, 2>>,
    medium_state: Option<Tensor<B, 2>>,
    slow_state: Option<Tensor<B, 2>>,
}

impl PureBrainRegion {
    fn new(name: String, initial_pos: (f64, f64, f64), color: Color) -> Self {
        Self {
            name,
            position: initial_pos,
            previous_position: initial_pos,
            velocity: (0.0, 0.0, 0.0),
            trajectory: VecDeque::with_capacity(MAX_TRAJECTORY_POINTS),
            color,
            dlinoss_model: None,
            fast_state: None,
            medium_state: None,
            slow_state: None,
        }
    }

    fn init_dlinoss(&mut self, device: &<B as Backend>::Device) {
        self.dlinoss_model = Some(DLinossBrainModel::new(device));

        // Initialize neural states
        self.fast_state = Some(Tensor::<B, 2>::from_floats(
            [[
                self.position.0 as f32,
                self.position.1 as f32,
                self.position.2 as f32,
            ]],
            device,
        ));

        self.medium_state = Some(Tensor::<B, 2>::from_floats(
            [[0.0, 0.0, 0.0]], // Initialize medium state to zero
            device,
        ));

        self.slow_state = Some(Tensor::<B, 2>::from_floats(
            [[0.0, 0.0, 0.0]], // Initialize slow state to zero
            device,
        ));
    }

    /// Pure dLinOSS step - no RK4, only neural state-space evolution
    fn dlinoss_step(&mut self, coupling_input: (f64, f64, f64), time: f64) {
        if let (Some(model), Some(state)) = (&self.dlinoss_model, &self.fast_state) {
            // Store previous position for velocity calculation
            self.previous_position = self.position;

            // Add small random noise to simulate neural noise and make dynamics interesting
            let mut rng = rand::rng();
            let noise_scale = 0.1;
            let mut vec = Vec::new();
            vec.push(rng.random::<f32>() * 2.0 - 1.0);
            vec.push(rng.random::<f32>() * 2.0 - 1.0);
            vec.push(rng.random::<f32>() * 2.0 - 1.0);

            // Create dynamic input that changes over time with larger amplitude
            let time_factor = (time * 0.5).sin() as f32;
            let dynamic_input = Tensor::<B, 2>::from_floats(
                [[
                    time_factor * 2.0,
                    (time * 0.7).cos() as f32 * 1.5,
                    (time * 0.3).sin() as f32 * 1.8,
                ]],
                &state.device(),
            );

            // Add coupling as external input with noise
            let coupling_tensor = Tensor::<B, 2>::from_floats(
                [[
                    coupling_input.0 as f32 + vec[0] as f32 * noise_scale,
                    coupling_input.1 as f32 + vec[1] as f32 * noise_scale,
                    coupling_input.2 as f32 + vec[2] as f32 * noise_scale,
                ]],
                &state.device(),
            );

            // Combine current state with coupling and dynamic input (reduced state feedback)
            let input = state.clone() * 0.5 + coupling_tensor * 0.3 + dynamic_input * 0.2;

            // Forward through the three dLinOSS blocks
            let output = model.forward(input);

            // Extract new position and get device before moving
            let device = output.device();
            let output_data = output.into_data();
            let output_values = output_data.to_vec::<f32>().unwrap();

            // Update position based on dLinOSS output with less momentum for more dynamic behavior
            let momentum = 0.7; // Reduced momentum for more dynamic behavior
            self.position.0 =
                momentum * self.position.0 + (1.0 - momentum) * output_values[0] as f64 * 10.0;
            self.position.1 =
                momentum * self.position.1 + (1.0 - momentum) * output_values[1] as f64 * 10.0;
            self.position.2 =
                momentum * self.position.2 + (1.0 - momentum) * output_values[2] as f64 * 10.0;

            // Calculate velocity
            self.velocity.0 = self.position.0 - self.previous_position.0;
            self.velocity.1 = self.position.1 - self.previous_position.1;
            self.velocity.2 = self.position.2 - self.previous_position.2;

            // Update state for next iteration
            self.fast_state = Some(Tensor::<B, 2>::from_floats(
                [[
                    self.position.0 as f32,
                    self.position.1 as f32,
                    self.position.2 as f32,
                ]],
                &device,
            ));
        }
    }

    /// Generate instrumentation data for this region
    fn generate_instrumentation_data(&self) -> RegionData {
        RegionData {
            name: self.name.clone(),
            position: self.position,
            activity_magnitude: self.get_activity_magnitude(),
            trajectory_length: self.trajectory.len(),
            velocity: self.velocity,
            dlinoss_state: DLinossStateData {
                fast_block_activity: self.get_activity_magnitude(),
                medium_block_activity: self.get_activity_magnitude() * 0.7,
                slow_block_activity: self.get_activity_magnitude() * 0.4,
                coupling_strength: 0.1, // This should come from the app
                damping_factor: 0.1,    // This should come from the app
            },
        }
    }

    fn update_trajectory(&mut self) {
        let point = (self.position.0, self.position.1);

        if self.trajectory.len() >= MAX_TRAJECTORY_POINTS {
            self.trajectory.pop_front();
        }
        self.trajectory.push_back(point);
    }

    fn get_trajectory_data(&self) -> Vec<(f64, f64)> {
        self.trajectory.iter().cloned().collect()
    }

    fn get_activity_magnitude(&self) -> f64 {
        (self.position.0 * self.position.0
            + self.position.1 * self.position.1
            + self.position.2 * self.position.2)
            .sqrt()
    }
}

struct PureDLinossBrainApp {
    regions: Vec<PureBrainRegion>,
    paused: bool,
    status_message: String,
    session_id: String,  // Unique session identifier for multi-instance support
    simulation_time: f64,
    coupling_strength: f64,
    damping_enabled: bool,
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    instrumentation: Option<InstrumentationManager>,
    
    // Enhanced intercommunication visualization data  
    intercommunication_matrix: Vec<Vec<f64>>, // 3×16 matrix: [region][signal_idx] = strength
    signal_velocities: Vec<Vec<f64>>, // Velocity for each signal for spinning animation
    signal_history: VecDeque<Vec<f64>>, // History of 18 signals (6 per region)
    lissajous_phase: f64, // Phase for animated Lissajous patterns
}

impl PureDLinossBrainApp {
    fn new() -> Self {
        let device = chosen_backend::get_device();

        // Create regions with pure dLinOSS dynamics and distinct starting positions
        let mut regions = vec![
            PureBrainRegion::new(
                "Prefrontal Cortex".to_string(),
                (5.0, 3.0, 2.0), // Different starting position
                Color::Cyan,
            ),
            PureBrainRegion::new(
                "Default Mode Network".to_string(),
                (-3.0, -5.0, -1.0), // Different starting position
                Color::Yellow,
            ),
            PureBrainRegion::new(
                "Thalamus".to_string(),
                (2.0, -3.0, 4.0), // Different starting position
                Color::Magenta,
            ),
        ];

        for region in &mut regions {
            region.init_dlinoss(&device);
        }

        // Initialize instrumentation
        let instrumentation = match InstrumentationManager::new() {
            Ok(mgr) => Some(mgr),
            Err(e) => {
                eprintln!("Warning: Could not initialize instrumentation: {}", e);
                None
            }
        };

        Self {
            regions,
            paused: false,
            status_message: "NeuroBreeze v1.0 🧠✨ - Gentle Neural Winds - Press 'p' pause, 'd' damping, 'q' quit".to_string(),
            session_id: format!("breeze-{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() % 10000),
            simulation_time: 0.0,
            coupling_strength: 0.1,
            damping_enabled: true,
            x_min: f64::MAX, x_max: f64::MIN, y_min: f64::MAX, y_max: f64::MIN,
            instrumentation,
            intercommunication_matrix: vec![vec![0.0; 16]; 3], // 3 regions × 16 signals each
            signal_velocities: vec![vec![0.0; 16]; 3], // Velocities for spinning animation
            signal_history: VecDeque::with_capacity(100), // Store last 100 signal snapshots
            lissajous_phase: 0.0,
        }
    }

    fn on_tick(&mut self) {
        if self.paused {
            return;
        }

        self.simulation_time += 0.05; // Faster time step for more visible dynamics
        self.lissajous_phase += 0.15; // Faster phase increment for more dramatic spinning

        let coupling_matrix = self.calculate_coupling();

        // Capture all 16 individual intercommunication signals per region for detailed visualization
        let mut current_signals = Vec::with_capacity(18); // Keep 6 per region for backward compatibility

        for (i, region) in self.regions.iter_mut().enumerate() {
            // Pure dLinOSS step - no RK4 integration
            region.dlinoss_step(coupling_matrix[i], self.simulation_time);
            region.update_trajectory();

            // Generate 16 distinct intercommunication signals per region
            let activity = region.get_activity_magnitude();
            let vel = region.velocity;
            let pos = region.position;
            
            // Create rich intercommunication signal matrix (16 signals per region)
            let mut region_signals = Vec::with_capacity(16);
            let mut region_velocities = Vec::with_capacity(16);
            
            for signal_idx in 0..16 {
                let freq_base = 0.8 + signal_idx as f64 * 0.4; // More varied frequencies per signal
                let phase_offset = signal_idx as f64 * 0.8; // Larger phase offsets
                
                // Signal strength: mix of activity, position, and velocity components
                let signal_strength = match signal_idx % 5 {
                    0 => activity * (self.lissajous_phase * freq_base + phase_offset).sin() * 1.5,
                    1 => vel.0 * (self.lissajous_phase * freq_base + phase_offset).cos() * 2.0,
                    2 => vel.1 * (self.lissajous_phase * freq_base + phase_offset).sin() * 1.8,
                    3 => vel.2 * (self.lissajous_phase * freq_base + phase_offset).cos() * 1.7,
                    4 => (pos.0.abs() + pos.1.abs()) * (self.lissajous_phase * freq_base + phase_offset).sin() * 1.2,
                    _ => activity * (self.lissajous_phase * freq_base + phase_offset).sin(),
                };
                
                // Velocity of signal (for spinning animation) - make it more dramatic
                let signal_velocity = (vel.0.abs() + vel.1.abs() + vel.2.abs()) * 0.5 + activity * 0.2 + signal_idx as f64 * 0.1;
                
                region_signals.push(signal_strength);
                region_velocities.push(signal_velocity);
            }
            
            // Store in intercommunication matrix
            self.intercommunication_matrix[i] = region_signals;
            self.signal_velocities[i] = region_velocities;
            
            // Keep backward compatibility: extract 6 signals for signal_history
            current_signals.extend(vec![
                activity * (self.lissajous_phase + i as f64).sin(),           // gro1[0]
                activity * (self.lissajous_phase + i as f64 + 1.0).sin(),     // gro1[1] 
                activity * (self.lissajous_phase + i as f64 + 2.0).sin(),     // gro1[2]
                vel.0 * (self.lissajous_phase * 2.0 + i as f64).cos(),       // gro2[0]
                vel.1 * (self.lissajous_phase * 2.0 + i as f64 + 1.0).cos(), // gro2[1]
                vel.2 * (self.lissajous_phase * 2.0 + i as f64 + 2.0).cos(), // gro2[2]
            ]);

            // Update bounds for visualization
            let (x, y) = (region.position.0, region.position.1);
            self.x_min = self.x_min.min(x);
            self.x_max = self.x_max.max(x);
            self.y_min = self.y_min.min(y);
            self.y_max = self.y_max.max(y);
        }

        // Store signal history for Lissajous patterns
        self.signal_history.push_back(current_signals);
        if self.signal_history.len() > 100 {
            self.signal_history.pop_front();
        }

        // Record instrumentation data
        self.record_instrumentation_data(&coupling_matrix);
    }

    fn calculate_coupling(&self) -> Vec<(f64, f64, f64)> {
        let mut coupling = vec![(0.0, 0.0, 0.0); self.regions.len()];

        for (i, coupling_ref) in coupling.iter_mut().enumerate().take(self.regions.len()) {
            let mut total_coupling = (0.0, 0.0, 0.0);

            for (j, region) in self.regions.iter().enumerate() {
                if i != j {
                    let weight = self.coupling_strength;
                    total_coupling.0 += weight * region.position.0;
                    total_coupling.1 += weight * region.position.1;
                    total_coupling.2 += weight * region.position.2;
                }
            }

            *coupling_ref = total_coupling;
        }

        coupling
    }

    fn toggle_pause(&mut self) {
        self.paused = !self.paused;
        self.status_message = if self.paused {
            "PAUSED - Press 'p' to resume".to_string()
        } else {
            "NeuroBreeze v1.0 🧠✨ - Gentle Neural Winds - Press 'p' pause, 'd' damping, 'q' quit".to_string()
        };
    }

    fn toggle_damping(&mut self) {
        self.damping_enabled = !self.damping_enabled;
        self.status_message = format!(
            "Damping: {} - Pure dLinOSS: Fast+Medium+Slow blocks",
            if self.damping_enabled {
                "ENABLED"
            } else {
                "DISABLED"
            }
        );
    }

    /// Record instrumentation data during simulation
    fn record_instrumentation_data(&mut self, coupling_matrix: &[(f64, f64, f64)]) {
        if let Some(ref mut instrumentation) = self.instrumentation {
            let region_data: Vec<RegionData> = self
                .regions
                .iter()
                .map(|region| region.generate_instrumentation_data())
                .collect();

            let total_energy = self
                .regions
                .iter()
                .map(|region| region.get_activity_magnitude().powi(2))
                .sum::<f64>();

            let instrumentation_data = NeuralInstrumentationData {
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs_f64(),
                simulation_time: self.simulation_time,
                regions: region_data,
                coupling_matrix: coupling_matrix.to_vec(),
                system_stats: SystemStats {
                    total_energy,
                    coupling_strength: self.coupling_strength,
                    damping_enabled: self.damping_enabled,
                    paused: self.paused,
                    bounds: ((self.x_min, self.x_max), (self.y_min, self.y_max)),
                },
            };

            if let Err(e) = instrumentation.record_data(&instrumentation_data) {
                eprintln!("Instrumentation error: {}", e);
            }
        }
    }
}

/// Instrumentation manager for real-time data logging and streaming
pub struct InstrumentationManager {
    log_writer: BufWriter<File>,
    pipe_path: String,
    iteration_count: usize,
    start_time: Instant,
}

impl InstrumentationManager {
    pub fn new() -> Result<Self, Box<dyn Error>> {
        // Initialize logging to stderr to avoid interfering with ratatui
        env_logger::Builder::from_default_env()
            .target(env_logger::Target::Stderr)
            .init();

        // Create log file
        let log_file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(LOG_FILE_PATH)?;

        let log_writer = BufWriter::new(log_file);

        // Create named pipe for real-time data streaming
        let pipe_path = Self::create_named_pipe()?;

        info!("Instrumentation system initialized");
        info!("Log file: {}", LOG_FILE_PATH);
        info!(
            "Data pipe: {} (use 'tail -f {}' or 'cat {}' to stream data)",
            pipe_path, pipe_path, pipe_path
        );

        Ok(Self {
            log_writer,
            pipe_path,
            iteration_count: 0,
            start_time: Instant::now(),
        })
    }

    fn create_named_pipe() -> Result<String, Box<dyn Error>> {
        let pipe_path = DATA_PIPE_PATH.to_string();

        // Remove existing pipe if it exists
        let _ = std::fs::remove_file(&pipe_path);

        // Create named pipe using mkfifo
        let output = std::process::Command::new("mkfifo")
            .arg(&pipe_path)
            .output();

        match output {
            Ok(result) => {
                if result.status.success() {
                    info!("Created named pipe: {}", pipe_path);
                    Ok(pipe_path)
                } else {
                    let error_msg = String::from_utf8_lossy(&result.stderr);
                    warn!("mkfifo failed: {}", error_msg);
                    Ok(pipe_path) // Return path anyway, we'll handle write errors gracefully
                }
            }
            Err(e) => {
                warn!("Could not create named pipe: {}", e);
                Ok(pipe_path) // Return path anyway
            }
        }
    }

    pub fn record_data(&mut self, data: &NeuralInstrumentationData) -> Result<(), Box<dyn Error>> {
        self.iteration_count += 1;

        // Always write to log file (buffered)
        let log_entry = format!("{},{}\n", data.timestamp, serde_json::to_string(data)?);
        self.log_writer.write_all(log_entry.as_bytes())?;

        // Flush every iteration for real-time monitoring and detailed logging
        if self.iteration_count % INSTRUMENTATION_INTERVAL == 0 {
            self.log_writer.flush()?;

            debug!(
                "Neural dynamics snapshot at t={:.2}s:",
                data.simulation_time
            );
            for region in &data.regions {
                debug!(
                    "  {}: pos=({:.2},{:.2},{:.2}), activity={:.3}, velocity=({:.3},{:.3},{:.3})",
                    region.name,
                    region.position.0,
                    region.position.1,
                    region.position.2,
                    region.activity_magnitude,
                    region.velocity.0,
                    region.velocity.1,
                    region.velocity.2
                );
            }

            // Write to named pipe for real-time streaming
            self.write_to_pipe(data)?;
        } else {
            // Still flush periodically even if not doing detailed logging
            if self.iteration_count % 5 == 0 {
                self.log_writer.flush()?;
            }
        }

        Ok(())
    }

    fn write_to_pipe(&self, data: &NeuralInstrumentationData) -> Result<(), Box<dyn Error>> {
        // Ultra-low latency pipe writing with O_NONBLOCK
        let pipe_data = format!("{}\n", serde_json::to_string(data)?);

        // Use O_NONBLOCK to avoid any blocking behavior
        #[cfg(feature = "libc")]
        use std::os::unix::fs::OpenOptionsExt;
        #[cfg(feature = "libc")]
        use std::os::unix::io::AsRawFd;

        // Open with O_NONBLOCK flag for immediate return
        let result = {
            #[cfg(feature = "libc")]
            {
                OpenOptions::new()
                    .write(true)
                    .custom_flags(libc::O_NONBLOCK)
                    .open(&self.pipe_path)
            }
            #[cfg(not(feature = "libc"))]
            {
                OpenOptions::new()
                    .write(true)
                    .open(&self.pipe_path)
            }
        };
        match result
        {
            Ok(mut file) => {
                // Try to write immediately - if pipe buffer is full, it will return EAGAIN
                match file.write_all(pipe_data.as_bytes()) {
                    Ok(()) => {
                        // Force immediate kernel flush for minimal latency
                        let _ = file.flush();

                        // Optional: force kernel to flush pipe buffer immediately
                        #[cfg(feature = "libc")]
                        unsafe {
                            libc::fsync(file.as_raw_fd());
                        }
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        // Pipe buffer full - drop this frame to maintain real-time performance
                        debug!(
                            "Pipe buffer full, dropping frame to maintain real-time performance"
                        );
                    }
                    Err(_) => {
                        // Other error - ignore silently to not disrupt simulation
                    }
                }
            }
            Err(_) => {
                // Pipe not ready or no reader - silently ignore
            }
        }

        Ok(())
    }

    pub fn close(mut self) -> Result<(), Box<dyn Error>> {
        self.log_writer.flush()?;
        info!(
            "Instrumentation session completed. {} iterations logged.",
            self.iteration_count
        );
        info!(
            "Total runtime: {:.2}s",
            self.start_time.elapsed().as_secs_f64()
        );

        // Clean up named pipe
        let _ = std::fs::remove_file(&self.pipe_path);

        Ok(())
    }
}

// UI drawing function with Lissajous intercommunication visualization
fn draw_ui(f: &mut Frame, app: &PureDLinossBrainApp) {
    let main_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints(
            [
                Constraint::Percentage(60), // Left side: Neural position chart
                Constraint::Percentage(40), // Right side: Intercommunication
            ]
            .as_ref(),
        )
        .split(f.area());

    let left_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints(
            [
                Constraint::Percentage(70), // Neural chart
                Constraint::Percentage(20), // Status
                Constraint::Percentage(10), // Info
            ]
            .as_ref(),
        )
        .split(main_chunks[0]);

    let right_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints(
            [
                Constraint::Percentage(50), // Lissajous patterns
                Constraint::Percentage(50), // Signal flow matrix
            ]
            .as_ref(),
        )
        .split(main_chunks[1]);

    let chart_area = left_chunks[0];
    let status_area = left_chunks[1];
    let info_area = left_chunks[2];
    let lissajous_area = right_chunks[0];
    let signal_matrix_area = right_chunks[1];

    // Auto-scale bounds
    let x_bounds = if app.x_min == f64::MAX || app.x_min == app.x_max {
        CANVAS_X_BOUNDS
    } else {
        [app.x_min * 1.1, app.x_max * 1.1]
    };
    let y_bounds = if app.y_min == f64::MAX || app.y_min == app.y_max {
        CANVAS_Y_BOUNDS
    } else {
        [app.y_min * 1.1, app.y_max * 1.1]
    };

    // Collect trajectory data
    let trajectory_data: Vec<Vec<(f64, f64)>> = app
        .regions
        .iter()
        .map(|region| region.get_trajectory_data())
        .collect();

    // Create animated datasets with pulsing effects and activity-based markers
    let datasets: Vec<Dataset> = app
        .regions
        .iter()
        .enumerate()
        .map(|(i, region)| {
            // Choose marker based on activity level for visual variety
            let marker = match region.get_activity_magnitude() {
                x if x > 3.0 => symbols::Marker::Braille, // High activity = dense pattern
                x if x > 2.0 => symbols::Marker::Block,   // Medium activity = solid block
                _ => symbols::Marker::Dot,                // Low activity = simple dot
            };
            
            // Pulse the color intensity based on neural activity
            let base_color = region.color;
            let pulsed_color = match region.get_activity_magnitude() as u8 % 3 {
                0 => base_color,
                1 => match base_color {
                    Color::Cyan => Color::LightCyan,
                    Color::Yellow => Color::LightYellow,
                    Color::Magenta => Color::LightMagenta,
                    other => other,
                },
                _ => match base_color {
                    Color::Cyan => Color::Blue,
                    Color::Yellow => Color::Red,
                    Color::Magenta => Color::Green,
                    other => other,
                },
            };
            
            Dataset::default()
                .name(format!("🧠 {}", region.name))  // Add brain emoji for fun
                .marker(marker)  // Dynamic marker based on activity
                .style(Style::default().fg(pulsed_color))  // Pulsing colors
                .data(&trajectory_data[i])
        })
        .collect();

    let chart = Chart::new(datasets)
        .block(
            TuiBlock::default()
                .title(format!("NeuroBreeze v1.0 🧠✨ [{}] - Gentle Neural Wind Simulation ✨🌊", &app.session_id))
                .borders(Borders::ALL),
        )
        .x_axis(
            ratatui::widgets::Axis::default()
                .title("X (Neural Activity) 🌐")
                .style(Style::default().fg(Color::LightBlue))  // More vibrant colors
                .bounds(x_bounds),
        )
        .y_axis(
            ratatui::widgets::Axis::default()
                .title("Y (Neural Activity) ⚡")
                .style(Style::default().fg(Color::LightGreen))  // More vibrant colors
                .bounds(y_bounds),
        );
    f.render_widget(chart, chart_area);

    // Status area with animated indicators
    let status_block = TuiBlock::default()
        .borders(Borders::ALL)
        .title("🧠 NeuroBreeze System Status 🌊");
    let status_inner = status_block.inner(status_area);
    f.render_widget(status_block, status_area);
    let status_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints(
            [
                Constraint::Length(1), // Control status
                Constraint::Length(1), // Architecture info
                Constraint::Length(1), // Block details
                Constraint::Length(1), // Damping status
            ]
            .as_ref(),
        )
        .split(status_inner);

    let control_paragraph = Paragraph::new(app.status_message.as_str())
        .style(Style::default().fg(Color::LightYellow));
    f.render_widget(control_paragraph, status_chunks[0]);

    let arch_paragraph =
        Paragraph::new("Architecture: 3 dLinOSS Blocks (Fast⚡→Medium🌊→Slow🌱) + GLU/ReLU")
        .style(Style::default().fg(Color::LightCyan));
    f.render_widget(arch_paragraph, status_chunks[1]);

    let blocks_paragraph = Paragraph::new(
        "Blocks: Fast(dt=0.01)⚡ → GLU+ReLU → Medium(dt=0.05)🌊 → GLU+ReLU → Slow(dt=0.1)🌱".to_string()
    )
    .style(Style::default().fg(Color::LightGreen));
    f.render_widget(blocks_paragraph, status_chunks[2]);

    // Dynamic damping status with activity indicators
    let total_activity: f64 = app.regions.iter().map(|r| r.get_activity_magnitude()).sum();
    let activity_bar = match (total_activity / 3.0) as u8 % 4 {
        0 => "▁▁▁",
        1 => "▃▃▃", 
        2 => "▅▅▅",
        _ => "▇▇▇",
    };
    
    let damping_paragraph = Paragraph::new(format!(
        "Damping: {} | Coupling: {:.3} | Time: {:.2}s | Activity: {} {:.1}",
        if app.damping_enabled { "ENABLED ✅" } else { "DISABLED ❌" },
        app.coupling_strength,
        app.simulation_time,
        activity_bar,
        total_activity / 3.0
    ))
    .style(Style::default().fg(if app.damping_enabled { Color::Green } else { Color::Red }));
    f.render_widget(damping_paragraph, status_chunks[3]);

    // Brain regions info with dynamic styling
    let info_block = TuiBlock::default()
        .borders(Borders::ALL)
        .title("🧠 Neural Regions (NeuroBreeze dLinOSS) 🌊");
    let info_inner = info_block.inner(info_area);
    f.render_widget(info_block, info_area);
    let region_info: Vec<ListItem> = app
        .regions
        .iter()
        .map(|region| {
            // Add emojis and activity indicators
            let activity_emoji = match region.get_activity_magnitude() {
                x if x > 3.0 => "🔥",  // High activity
                x if x > 2.0 => "⚡",  // Medium activity  
                _ => "💤",             // Low activity
            };
            
            let region_emoji = match region.name.as_str() {
                "Prefrontal Cortex" => "🧠",
                "Default Mode Network" => "🌐", 
                "Thalamus" => "⚙️",
                _ => "🔮",
            };
            
            ListItem::new(format!(
                "{} {}: ({:.2}, {:.2}, {:.2}) |Activity|={:.3} {}",
                region_emoji,
                region.name,
                region.position.0,
                region.position.1,
                region.position.2,
                region.get_activity_magnitude(),
                activity_emoji
            ))
            .style(Style::default().fg(region.color))
        })
        .collect();

    let regions_list = List::new(region_info);
    f.render_widget(regions_list, info_inner);

    // ========= 3×16 SPINNING LISSAJOUS INTERCOMMUNICATION VISUALIZATION =========
    
    // Show all 16 individual intercommunication signals as spinning Lissajous patterns
    if !app.intercommunication_matrix.is_empty() {
        // Create a grid layout to display multiple small Lissajous patterns
        // We'll use a text-based representation since Chart widgets are too large for a grid
        
        let mut lissajous_text = Vec::new();
        let region_names = ["🧠PFC", "🌐DMN", "⚙️THL"];
        let region_colors = [Color::Cyan, Color::Yellow, Color::Magenta];
        
        // Add header
        lissajous_text.push(ListItem::new("🌀 3×16 Intercommunication Signal Matrix 🌊").style(Style::default().fg(Color::White)));
        lissajous_text.push(ListItem::new("".to_string())); // Spacer
        
        for region_idx in 0..3 {
            let region_signals = &app.intercommunication_matrix[region_idx];
            let region_velocities = &app.signal_velocities[region_idx];
            
            // Create visual representation of spinning signals
            let mut signal_row = String::new();
            signal_row.push_str(&format!("{} │", region_names[region_idx]));
            
            for signal_idx in 0..16 {
                if signal_idx < region_signals.len() && signal_idx < region_velocities.len() {
                    let signal_strength = region_signals[signal_idx].abs();
                    let velocity = region_velocities[signal_idx].abs();
                    
                    // Create spinning pattern based on phase, strength, and velocity
                    let phase = app.lissajous_phase + signal_idx as f64 * 0.2 + region_idx as f64;
                    let spin_phase = phase * (1.0 + velocity * 0.5);
                    
                    // Different spinning symbols based on signal strength and phase
                    let symbol = match ((spin_phase % (2.0 * std::f64::consts::PI)) / (std::f64::consts::PI / 8.0)) as usize {
                        0 => if signal_strength > 2.0 { "🔴" } else if signal_strength > 1.0 { "◐" } else { "·" },
                        1 => if signal_strength > 2.0 { "🟠" } else if signal_strength > 1.0 { "◓" } else { "·" },
                        2 => if signal_strength > 2.0 { "🟡" } else if signal_strength > 1.0 { "◑" } else { "·" },
                        3 => if signal_strength > 2.0 { "🟢" } else if signal_strength > 1.0 { "◒" } else { "·" },
                        4 => if signal_strength > 2.0 { "🔵" } else if signal_strength > 1.0 { "◐" } else { "·" },
                        5 => if signal_strength > 2.0 { "🟣" } else if signal_strength > 1.0 { "◓" } else { "·" },
                        6 => if signal_strength > 2.0 { "⚫" } else if signal_strength > 1.0 { "◑" } else { "·" },
                        7 => if signal_strength > 2.0 { "⚪" } else if signal_strength > 1.0 { "◒" } else { "·" },
                        8 => if signal_strength > 2.0 { "🔴" } else if signal_strength > 1.0 { "◐" } else { "·" },
                        9 => if signal_strength > 2.0 { "🟠" } else if signal_strength > 1.0 { "◓" } else { "·" },
                        10 => if signal_strength > 2.0 { "🟡" } else if signal_strength > 1.0 { "◑" } else { "·" },
                        11 => if signal_strength > 2.0 { "🟢" } else if signal_strength > 1.0 { "◒" } else { "·" },
                        12 => if signal_strength > 2.0 { "🔵" } else if signal_strength > 1.0 { "◐" } else { "·" },
                        13 => if signal_strength > 2.0 { "🟣" } else if signal_strength > 1.0 { "◓" } else { "·" },
                        14 => if signal_strength > 2.0 { "⚫" } else if signal_strength > 1.0 { "◑" } else { "·" },
                        15 => if signal_strength > 2.0 { "⚪" } else if signal_strength > 1.0 { "◒" } else { "·" },
                        _ => "○",
                    };
                    
                    signal_row.push_str(symbol);
                } else {
                    signal_row.push('·');
                }
            }
            
            lissajous_text.push(ListItem::new(signal_row).style(Style::default().fg(region_colors[region_idx])));
        }
        
        // Add velocity legend
        lissajous_text.push(ListItem::new("".to_string())); // Spacer
        lissajous_text.push(ListItem::new("🌊 Legend: 🔴🟠🟡🟢🔵🟣⚫⚪ = high velocity spinning signals").style(Style::default().fg(Color::Gray)));
        lissajous_text.push(ListItem::new("💨 Medium: ◐◓◑◒ = spinning signals, Low: · = inactive").style(Style::default().fg(Color::Gray)));
        lissajous_text.push(ListItem::new("🎯 Each column = 1 of 16 intercommunication signals per region").style(Style::default().fg(Color::Gray)));
        
        let lissajous_list = List::new(lissajous_text);
        f.render_widget(lissajous_list, lissajous_area);
    } else {
        let empty_lissajous_block = TuiBlock::default()
            .borders(Borders::ALL)
            .title("🌀 Lissajous Signal Patterns 🌊");
        f.render_widget(empty_lissajous_block, lissajous_area);
    }
    
    // ========= SIGNAL FLOW MATRIX =========
    
    let matrix_block = TuiBlock::default()
        .borders(Borders::ALL)
        .title("📡 Signal Flow Matrix (gri↔gro) 🔄");
    let matrix_inner = matrix_block.inner(signal_matrix_area);
    f.render_widget(matrix_block, signal_matrix_area);
    
    if !app.signal_history.is_empty() {
        let latest_signals = app.signal_history.back().unwrap();
        
        // Display signal strength matrix
        let mut matrix_lines = Vec::new();
        
        for region_idx in 0..3 {
            let signal_base = region_idx * 6;
            let region_names = ["🧠PFC", "🌐DMN", "⚙️THL"];
            
            if signal_base + 5 < latest_signals.len() {
                let gro1_strength = latest_signals[signal_base..signal_base + 3].iter().map(|x| x.abs()).sum::<f64>() / 3.0;
                let gro2_strength = latest_signals[signal_base + 3..signal_base + 6].iter().map(|x| x.abs()).sum::<f64>() / 3.0;
                
                let gro1_bar = match (gro1_strength * 10.0) as usize {
                    0..=1 => "▁",
                    2..=3 => "▃", 
                    4..=5 => "▅",
                    6..=7 => "▆",
                    _ => "▇",
                };
                
                let gro2_bar = match (gro2_strength * 10.0) as usize {
                    0..=1 => "▁",
                    2..=3 => "▃",
                    4..=5 => "▅", 
                    6..=7 => "▆",
                    _ => "▇",
                };
                
                matrix_lines.push(ListItem::new(format!(
                    "{} → gro1:{} gro2:{} │ {:.2}/{:.2}",
                    region_names[region_idx], gro1_bar, gro2_bar, gro1_strength, gro2_strength
                )).style(Style::default().fg([Color::Cyan, Color::Yellow, Color::Magenta][region_idx])));
            }
        }
        
        let matrix_list = List::new(matrix_lines);
        f.render_widget(matrix_list, matrix_inner);
    }
}

fn run_app(terminal: &mut Terminal<CrosstermBackend<Stdout>>) -> Result<(), Box<dyn Error>> {
    let mut app = PureDLinossBrainApp::new();
    let mut last_tick = Instant::now();

    loop {
        terminal.draw(|f| draw_ui(f, &app))?;

        let timeout = Duration::from_millis(SIMULATION_SPEED)
            .checked_sub(last_tick.elapsed())
            .unwrap_or_else(|| Duration::from_secs(0));

        if crossterm::event::poll(timeout)? {
            if let event::Event::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Char('q') => return Ok(()),
                    KeyCode::Char('p') => app.toggle_pause(),
                    KeyCode::Char('d') => app.toggle_damping(),
                    KeyCode::Char('+') => {
                        app.coupling_strength = (app.coupling_strength + 0.01).min(1.0);
                    }
                    KeyCode::Char('-') => {
                        app.coupling_strength = (app.coupling_strength - 0.01).max(0.0);
                    }
                    _ => {}
                }
            }
        }

        if last_tick.elapsed() >= Duration::from_millis(SIMULATION_SPEED) {
            app.on_tick();
            last_tick = Instant::now();
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Run the pure dLinOSS app
    let res = run_app(&mut terminal);

    // Restore terminal
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen,)?;
    terminal.show_cursor()?;

    if let Err(err) = res {
        println!("{:?}", err);
    }

    Ok(())
}
