//! LinOSS 3D Neural Visualizer with WGPU Backend - Simplified Implementation
//! 
//! This demonstrates the architectural benefits of using WGPU for both:
//! - egui rendering backend
//! - Burn neural computation backend
//! 
//! Key improvements over OpenGL:
//! 1. Unified WGPU backend for both UI and ML computation
//! 2. Modern, safe, cross-platform graphics API
//! 3. Better debugging and validation
//! 4. Potential for GPU memory sharing (future enhancement)
//! 5. Native support in both egui and Burn frameworks

use eframe::{egui, NativeOptions};
use egui_plot::{Line, Plot, PlotPoints};
use std::collections::VecDeque;

// LinOSS imports with WGPU backend potential
use linoss_rust::Vector;
use nalgebra::DVector;

// For future WGPU + Burn integration
// use burn::backend::Wgpu;
// type BurnWgpuBackend = Wgpu<f32, i32>;

// Neural state with WGPU architecture ready
pub struct LinossNeuralState {
    // Current state vectors
    pub positions: Vec<[f32; 3]>,      // 3D positions for visualization
    pub velocities: Vec<[f32; 3]>,     // Velocities for dynamics
    pub neural_outputs: Vec<f32>,      // Neural layer outputs
    
    // Input signals
    pub input_signal: Vector,
    
    // Simulation parameters
    pub time: f32,
    pub dt: f32,
    pub oscillator_count: usize,
    
    // LinOSS configuration
    pub d_input: usize,
    pub d_model: usize,
    pub d_output: usize,
    pub delta_t: f64,
    pub damping_enabled: bool,
    pub damping_strength: f64,
    
    // Future: WGPU device for GPU-GPU communication
    // pub wgpu_device: Option<Arc<wgpu::Device>>,
    // pub neural_compute_buffer: Option<wgpu::Buffer>,
    // pub position_buffer: Option<wgpu::Buffer>,
}

impl LinossNeuralState {
    pub fn new(oscillator_count: usize) -> Self {
        let d_input = 3;           // 3D input (x, y, z coordinates)
        let d_model = oscillator_count.max(8); // Hidden dimension 
        let d_output = oscillator_count; // Output for each oscillator
        
        let positions = vec![[0.0; 3]; oscillator_count];
        let velocities = vec![[0.0; 3]; oscillator_count]; 
        let neural_outputs = vec![0.0; oscillator_count];
        let input_signal = DVector::zeros(d_input);
        
        Self {
            positions,
            velocities,
            neural_outputs,
            input_signal,
            time: 0.0,
            dt: 0.016, // ~60 FPS
            oscillator_count,
            d_input,
            d_model,
            d_output,
            delta_t: 0.1,
            damping_enabled: true,
            damping_strength: 0.1,
        }
    }
    
    pub fn update(&mut self, params: &LinossParams) {
        self.time += self.dt;
        
        // Generate input signal (could be from Burn tensors in WGPU backend)
        let t = self.time as f64;
        let frequency = params.frequency as f64;
        self.input_signal[0] = (t * frequency).sin() * params.amplitude as f64;
        self.input_signal[1] = (t * frequency * 1.1).cos() * params.amplitude as f64;
        self.input_signal[2] = (t * frequency * 0.8).sin() * params.amplitude as f64 * 0.5;
        
        // LinOSS-inspired dynamics (future: move to WGPU compute shaders)
        for i in 0..self.oscillator_count {
            let phase = i as f32 * std::f32::consts::TAU / self.oscillator_count as f32;
            let t = self.time;
            
            // D-LinOSS inspired dynamics with oscillatory behavior
            let alpha = params.alpha;
            let beta = params.beta;
            let gamma = params.gamma;
            
            // Primary oscillations with individual frequency shifts
            let freq_x = params.frequency * (1.0 + 0.1 * (i as f32 / self.oscillator_count as f32));
            let freq_y = params.frequency * (1.1 + 0.05 * (i as f32 / self.oscillator_count as f32));
            let freq_z = params.frequency * (0.8 + 0.15 * (i as f32 / self.oscillator_count as f32));
            
            // Oscillatory dynamics with damping
            let damping = if self.damping_enabled { 
                (-gamma * t).exp() 
            } else { 
                1.0 
            };
            
            // Position updates with coupling
            let base_x = alpha * (t * freq_x + phase).sin() * damping;
            let base_y = beta * (t * freq_y + phase + std::f32::consts::FRAC_PI_2).cos() * damping;
            let base_z = gamma * (t * freq_z * 0.5 + phase).sin() * (t * 0.3 + phase).cos() * damping;
            
            // Add coupling between oscillators (LinOSS-like coupling)
            let mut coupled_x = base_x;
            let mut coupled_y = base_y;
            let mut coupled_z = base_z;
            
            if i > 0 {
                let prev_pos = self.positions[i - 1];
                let coupling_strength = params.coupling * 0.1;
                coupled_x += prev_pos[0] * coupling_strength;
                coupled_y += prev_pos[1] * coupling_strength;
                coupled_z += prev_pos[2] * coupling_strength;
            }
            
            // Update positions with momentum
            let momentum = 0.1;
            self.velocities[i][0] = momentum * self.velocities[i][0] + (1.0 - momentum) * (coupled_x - self.positions[i][0]);
            self.velocities[i][1] = momentum * self.velocities[i][1] + (1.0 - momentum) * (coupled_y - self.positions[i][1]);
            self.velocities[i][2] = momentum * self.velocities[i][2] + (1.0 - momentum) * (coupled_z - self.positions[i][2]);
            
            self.positions[i][0] = coupled_x;
            self.positions[i][1] = coupled_y;
            self.positions[i][2] = coupled_z;
            
            // Neural output (combination of position components)
            self.neural_outputs[i] = (coupled_x + coupled_y * 0.7 + coupled_z * 0.3) * params.amplitude;
        }
    }
    
    pub fn get_signal_values(&self) -> Vec<f32> {
        self.neural_outputs.clone()
    }
}

// Parameters for LinOSS dynamics
#[derive(Clone, Debug)]
pub struct LinossParams {
    pub alpha: f32,           // X oscillation strength
    pub beta: f32,            // Y oscillation strength
    pub gamma: f32,           // Z oscillation strength / damping
    pub frequency: f32,       // Base frequency
    pub amplitude: f32,       // Output amplitude
    pub coupling: f32,        // Inter-oscillator coupling
    pub oscillator_count: usize,
}

impl Default for LinossParams {
    fn default() -> Self {
        Self {
            alpha: 1.2,
            beta: 0.8,
            gamma: 0.5,
            frequency: 1.0,
            amplitude: 1.5,
            coupling: 0.3,
            oscillator_count: 32,
        }
    }
}

// Plot data for 2D visualization
#[derive(Default)]
pub struct PlotData {
    pub time_series: VecDeque<f32>,
    pub oscillator_signals: Vec<VecDeque<f32>>,
    pub max_points: usize,
}

impl PlotData {
    pub fn new(oscillator_count: usize) -> Self {
        Self {
            time_series: VecDeque::new(),
            oscillator_signals: vec![VecDeque::new(); oscillator_count],
            max_points: 200,
        }
    }
    
    pub fn update(&mut self, time: f32, signals: &[f32]) {
        self.time_series.push_back(time);
        if self.time_series.len() > self.max_points {
            self.time_series.pop_front();
        }
        
        for (i, &signal) in signals.iter().enumerate() {
            if i < self.oscillator_signals.len() {
                self.oscillator_signals[i].push_back(signal);
                if self.oscillator_signals[i].len() > self.max_points {
                    self.oscillator_signals[i].pop_front();
                }
            }
        }
    }
}

// Main application with WGPU backend awareness
#[derive(Default)]
struct LinossWgpuDemoApp {
    neural_state: Option<LinossNeuralState>,
    linoss_params: LinossParams,
    plot_data: Option<PlotData>,
    simulation_running: bool,
    show_architecture_info: bool,
}

impl LinossWgpuDemoApp {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let params = LinossParams::default();
        let neural_state = LinossNeuralState::new(params.oscillator_count);
        let plot_data = PlotData::new(params.oscillator_count);
        
        Self {
            neural_state: Some(neural_state),
            linoss_params: params,
            plot_data: Some(plot_data),
            simulation_running: true,
            show_architecture_info: true,
        }
    }
}

impl eframe::App for LinossWgpuDemoApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Update LinOSS neural simulation
        if self.simulation_running {
            if let Some(ref mut neural_state) = self.neural_state {
                neural_state.update(&self.linoss_params);
                
                // Update plot data
                let signals = neural_state.get_signal_values();
                if let Some(ref mut plot_data) = self.plot_data {
                    plot_data.update(neural_state.time, &signals);
                }
            }
        }
        
        // Top panel
        egui::TopBottomPanel::top("title").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("🚀 LinOSS WGPU Architecture Demo");
                ui.separator();
                if ui.button(if self.simulation_running { "⏸ Pause" } else { "▶ Play" }).clicked() {
                    self.simulation_running = !self.simulation_running;
                }
                ui.separator();
                ui.checkbox(&mut self.show_architecture_info, "Architecture Info");
                ui.separator();
                ui.label("🎮 Backend: WGPU");
            });
        });
        
        // Architecture information panel
        if self.show_architecture_info {
            egui::SidePanel::left("architecture").show(ctx, |ui| {
                ui.heading("🏗️ WGPU Architecture Benefits");
                
                ui.group(|ui| {
                    ui.label("🎯 Current Implementation:");
                    ui.label("✅ egui with WGPU backend");
                    ui.label("✅ LinOSS neural dynamics");
                    ui.label("✅ Real-time visualization");
                    ui.label("✅ Cross-platform support");
                });
                
                ui.separator();
                
                ui.group(|ui| {
                    ui.label("🚀 WGPU Advantages over OpenGL:");
                    ui.label("• Modern, safe graphics API");
                    ui.label("• Better validation & debugging");
                    ui.label("• Cross-platform (Vulkan/Metal/DX12)");
                    ui.label("• WebGPU compatibility");
                    ui.label("• Memory safety guarantees");
                });
                
                ui.separator();
                
                ui.group(|ui| {
                    ui.label("🔥 Burn + WGPU Integration:");
                    ui.label("• Unified GPU backend");
                    ui.label("• Shared memory pools");
                    ui.label("• Zero-copy tensor operations");
                    ui.label("• GPU-GPU communication");
                    ui.label("• Compute shader neural nets");
                });
                
                ui.separator();
                
                ui.group(|ui| {
                    ui.label("⚡ Future Optimizations:");
                    ui.label("• Neural dynamics in compute shaders");
                    ui.label("• Direct GPU buffer rendering");
                    ui.label("• Real-time GPU feedback loops");
                    ui.label("• Advanced neural visualizations");
                    ui.label("• Multi-GPU scaling");
                });
                
                ui.separator();
                
                ui.group(|ui| {
                    ui.label("📊 Performance Benefits:");
                    ui.label("• Reduced CPU-GPU transfers");
                    ui.label("• Lower memory bandwidth usage");
                    ui.label("• Better GPU utilization");
                    ui.label("• Async computation pipelines");
                    ui.label("• Hardware-optimized kernels");
                });
                
                ui.separator();
                
                ui.heading("🎮 Implementation Status");
                ui.label("🟢 WGPU Backend: Active");
                ui.label("🟡 Burn Integration: Planned");
                ui.label("🔵 GPU Compute: Future");
                ui.label("🟣 Zero-Copy: Future");
            });
        }
        
        // Central panel with isometric 3D view and signals
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                // 3D isometric view
                ui.group(|ui| {
                    ui.vertical(|ui| {
                        ui.heading("🧠 LinOSS Neural Oscillators (3D Isometric)");
                        
                        if let Some(ref neural_state) = self.neural_state {
                            Plot::new("3d_neural_view")
                                .height(400.0)
                                .view_aspect(1.0)
                                .show(ui, |plot_ui| {
                                    // Convert 3D positions to 2D isometric projection
                                    let projected_points: Vec<[f64; 2]> = neural_state.positions
                                        .iter()
                                        .map(|pos| {
                                            // Isometric projection: 3D -> 2D
                                            let x = pos[0] - pos[1] * 0.5;
                                            let y = pos[2] + (pos[0] + pos[1]) * 0.3;
                                            [x as f64, y as f64]
                                        })
                                        .collect();
                                    
                                    // Plot oscillators as points with WGPU-inspired colors
                                    plot_ui.points(
                                        egui_plot::Points::new(PlotPoints::from(projected_points.clone()))
                                            .name("Neural Oscillators")
                                            .radius(6.0)
                                            .color(egui::Color32::from_rgb(100, 200, 255))
                                    );
                                    
                                    // Connect with lines
                                    if projected_points.len() > 1 {
                                        plot_ui.line(
                                            Line::new(PlotPoints::from(projected_points))
                                                .name("Neural Network")
                                                .width(1.5)
                                                .color(egui::Color32::from_rgba_unmultiplied(150, 255, 150, 120))
                                        );
                                    }
                                });
                        }
                    });
                });
                
                // Signal plots
                ui.group(|ui| {
                    ui.vertical(|ui| {
                        ui.heading("📈 Neural Signal Time Series");
                        
                        if let (Some(plot_data), Some(_neural_state)) = (&self.plot_data, &self.neural_state) {
                            if !plot_data.time_series.is_empty() {
                                Plot::new("neural_signals")
                                    .height(400.0)
                                    .legend(egui_plot::Legend::default())
                                    .show(ui, |plot_ui| {
                                        let colors = [
                                            egui::Color32::from_rgb(255, 100, 100),
                                            egui::Color32::from_rgb(100, 255, 100),
                                            egui::Color32::from_rgb(100, 100, 255),
                                            egui::Color32::from_rgb(255, 255, 100),
                                            egui::Color32::from_rgb(255, 100, 255),
                                            egui::Color32::from_rgb(100, 255, 255),
                                        ];
                                        
                                        // Show first 6 oscillators
                                        for (i, signals) in plot_data.oscillator_signals.iter().enumerate().take(6) {
                                            if !signals.is_empty() && plot_data.time_series.len() == signals.len() {
                                                let points: PlotPoints = plot_data.time_series.iter()
                                                    .zip(signals.iter())
                                                    .map(|(&time, &value)| [time as f64, value as f64])
                                                    .collect();
                                                
                                                let line = Line::new(points)
                                                    .color(colors[i % colors.len()])
                                                    .width(2.0)
                                                    .name(format!("Oscillator-{}", i));
                                                
                                                plot_ui.line(line);
                                            }
                                        }
                                    });
                            }
                        }
                    });
                });
            });
            
            ui.separator();
            
            // Parameters
            ui.horizontal(|ui| {
                ui.group(|ui| {
                    ui.label("LinOSS Parameters:");
                    ui.add(egui::Slider::new(&mut self.linoss_params.alpha, 0.1..=3.0).text("Alpha"));
                    ui.add(egui::Slider::new(&mut self.linoss_params.beta, 0.1..=3.0).text("Beta"));
                    ui.add(egui::Slider::new(&mut self.linoss_params.gamma, 0.1..=2.0).text("Gamma"));
                });
                
                ui.group(|ui| {
                    ui.label("Dynamics:");
                    ui.add(egui::Slider::new(&mut self.linoss_params.frequency, 0.1..=3.0).text("Frequency"));
                    ui.add(egui::Slider::new(&mut self.linoss_params.amplitude, 0.1..=3.0).text("Amplitude"));
                    ui.add(egui::Slider::new(&mut self.linoss_params.coupling, 0.0..=1.0).text("Coupling"));
                });
                
                ui.group(|ui| {
                    ui.label("Network:");
                    ui.add(egui::Slider::new(&mut self.linoss_params.oscillator_count, 8..=64).text("Oscillators"));
                    
                    if ui.button("Update Network").clicked() {
                        self.neural_state = Some(LinossNeuralState::new(self.linoss_params.oscillator_count));
                        self.plot_data = Some(PlotData::new(self.linoss_params.oscillator_count));
                    }
                });
            });
            
            ui.separator();
            
            // Status
            ui.horizontal(|ui| {
                if let Some(ref neural_state) = self.neural_state {
                    ui.label(format!("⏰ Time: {:.2}s", neural_state.time));
                    ui.separator();
                    ui.label(format!("🔄 Oscillators: {}", neural_state.oscillator_count));
                    ui.separator();
                    ui.label("🧠 Neural Dynamics: LinOSS");
                    ui.separator();
                    ui.label("🎮 Rendering: WGPU Backend");
                }
            });
        });
        
        ctx.request_repaint();
    }
}

fn main() -> eframe::Result {
    env_logger::init();
    
    let options = NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 800.0])
            .with_title("LinOSS Neural Visualizer - WGPU Architecture Demo"),
        renderer: eframe::Renderer::Wgpu, // Force WGPU backend
        ..Default::default()
    };
    
    eframe::run_native(
        "LinOSS WGPU Demo",
        options,
        Box::new(|cc| Ok(Box::new(LinossWgpuDemoApp::new(cc)))),
    )
}
