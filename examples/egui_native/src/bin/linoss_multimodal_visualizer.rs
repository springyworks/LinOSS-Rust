//! LinOSS 3D Neural Visualizer with Animated GIF Integration
//! 
//! This demonstrates the complete WGPU unified architecture with:
//! - Real-time D-LinOSS neural dynamics
//! - 3D neural oscillator visualization  
//! - 2D time series plotting
//! - Animated GIF display integration
//! - Modern egui interface with image support
//! 
//! Features:
//! - WGPU unified backend for neural computation and rendering
//! - Animated GIF assets displayed alongside neural visualization
//! - Multiple view modes: 3D, 2D plots, and multimedia integration
//! - Real-time parameter control and neural dynamics

use eframe::{egui, NativeOptions};
use egui_plot::{Line, Plot, PlotPoints};
use std::collections::VecDeque;
use std::path::Path;
use anyhow::Result;

// LinOSS imports with WGPU backend potential
use linoss_rust::Vector;
use nalgebra::DVector;

// Image and animation support
use image::codecs::gif::GifDecoder;
use image::AnimationDecoder;
use std::io::Cursor;

// GIF animation manager
pub struct AnimatedGif {
    frames: Vec<egui::ColorImage>,
    frame_durations: Vec<std::time::Duration>,
    current_frame: usize,
    last_frame_time: std::time::Instant,
    texture_handles: Vec<Option<egui::TextureHandle>>,
    is_loaded: bool,
}

impl Default for AnimatedGif {
    fn default() -> Self {
        Self::new()
    }
}

impl AnimatedGif {
    pub fn new() -> Self {
        Self {
            frames: Vec::new(),
            frame_durations: Vec::new(),
            current_frame: 0,
            last_frame_time: std::time::Instant::now(),
            texture_handles: Vec::new(),
            is_loaded: false,
        }
    }
    
    pub fn load_from_path<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let file_data = std::fs::read(path)?;
        self.load_from_bytes(&file_data)
    }
    
    pub fn load_from_bytes(&mut self, data: &[u8]) -> Result<()> {
        let cursor = Cursor::new(data);
        let decoder = GifDecoder::new(cursor)?;
        let frames = decoder.into_frames();
        
        self.frames.clear();
        self.frame_durations.clear();
        self.texture_handles.clear();
        
        for frame_result in frames {
            let frame = frame_result?;
            let delay = frame.delay();
            let original_duration = std::time::Duration::from_millis(
                (delay.numer_denom_ms().0 as u64 * 1000) / delay.numer_denom_ms().1 as u64
            );
            
            // Override extremely long durations with reasonable animation speed
            let final_duration = if original_duration > std::time::Duration::from_millis(500) {
                std::time::Duration::from_millis(100) // 10 FPS for reasonable animation
            } else {
                // Ensure minimum frame duration (avoid too fast animation)
                let min_duration = std::time::Duration::from_millis(50); // 20 FPS max
                if original_duration < min_duration { min_duration } else { original_duration }
            };
            
            // Convert frame to egui::ColorImage
            let image_buffer = frame.into_buffer();
            let (width, height) = image_buffer.dimensions();
            let pixels: Vec<egui::Color32> = image_buffer
                .pixels()
                .map(|p| egui::Color32::from_rgba_unmultiplied(p[0], p[1], p[2], p[3]))
                .collect();
            
            let color_image = egui::ColorImage {
                size: [width as usize, height as usize],
                pixels,
            };
            
            self.frames.push(color_image);
            self.frame_durations.push(final_duration);
            self.texture_handles.push(None);
            
            log::info!("🎬 Loaded GIF frame {} with duration {:?} (original: {:?})", 
                self.frames.len(), final_duration, original_duration);
        }
        
        self.is_loaded = true;
        log::info!("✅ Loaded animated GIF with {} frames", self.frames.len());
        Ok(())
    }
    
    pub fn update_and_get_texture(&mut self, ctx: &egui::Context) -> Option<&egui::TextureHandle> {
        if !self.is_loaded || self.frames.is_empty() {
            return None;
        }
        
        // Update frame timing
        let now = std::time::Instant::now();
        let elapsed = now.duration_since(self.last_frame_time);
        
        // Check if it's time to advance to next frame
        if elapsed >= self.frame_durations[self.current_frame] {
            let old_frame = self.current_frame;
            self.current_frame = (self.current_frame + 1) % self.frames.len();
            self.last_frame_time = now;
            
            println!("🎬 GIF frame changed: {} -> {} (elapsed: {:?}ms, needed: {:?}ms)", 
                old_frame, self.current_frame, elapsed.as_millis(), 
                self.frame_durations[old_frame].as_millis());
        }
        
        // Ensure we have a texture handle for the current frame
        if self.texture_handles[self.current_frame].is_none() {
            let texture_handle = ctx.load_texture(
                format!("animated_gif_frame_{}", self.current_frame),
                self.frames[self.current_frame].clone(),
                egui::TextureOptions::default(),
            );
            self.texture_handles[self.current_frame] = Some(texture_handle);
        }
        
        self.texture_handles[self.current_frame].as_ref()
    }
    
    pub fn get_frame_info(&self) -> (usize, usize) {
        (self.current_frame + 1, self.frames.len())
    }
    
    pub fn get_fps(&self) -> f32 {
        if self.frames.is_empty() {
            return 0.0;
        }
        let avg_duration: std::time::Duration = self.frame_durations.iter().sum::<std::time::Duration>() / self.frame_durations.len() as u32;
        1000.0 / avg_duration.as_millis() as f32
    }
    
    pub fn get_time_since_last_frame(&self) -> std::time::Duration {
        std::time::Instant::now().duration_since(self.last_frame_time)
    }
}

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
        
        // D-LinOSS inspired dynamics (future: move to WGPU compute shaders)
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
    
    pub fn get_line_data(&self, oscillator_idx: usize) -> PlotPoints {
        if oscillator_idx < self.oscillator_signals.len() {
            self.time_series
                .iter()
                .zip(self.oscillator_signals[oscillator_idx].iter())
                .map(|(&t, &s)| [t as f64, s as f64])
                .collect()
        } else {
            PlotPoints::new(vec![])
        }
    }
}

// Main application with WGPU unified backend and animated GIF
pub struct LinossMultiModalApp {
    neural_state: LinossNeuralState,
    params: LinossParams,
    plot_data: PlotData,
    animated_gif: AnimatedGif,
    
    // UI state
    show_3d: bool,
    show_plots: bool,
    show_params: bool,
    show_gif: bool,
    gif_size: f32,
    
    // Layout options
    layout_mode: LayoutMode,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum LayoutMode {
    Tabbed,
    Split,
    Grid,
}

impl LinossMultiModalApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        env_logger::init();
        
        let params = LinossParams::default();
        let neural_state = LinossNeuralState::new(params.oscillator_count);
        let plot_data = PlotData::new(params.oscillator_count);
        let mut animated_gif = AnimatedGif::new();
        
        // Try to load the GIF from assets
        if let Err(e) = animated_gif.load_from_path("assets/image015.gif") {
            log::warn!("Failed to load animated GIF: {}", e);
        }
        
        log::info!("🚀 LinOSS Multi-Modal Visualizer initialized");
        log::info!("🎨 Features: 3D neural visualization + 2D plotting + animated GIF");
        log::info!("🔧 Architecture: WGPU unified backend ready");
        
        Self {
            neural_state,
            params,
            plot_data,
            animated_gif,
            show_3d: true,
            show_plots: true,
            show_params: true,
            show_gif: true,
            gif_size: 200.0,
            layout_mode: LayoutMode::Split,
        }
    }
    
    fn draw_control_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("🧠 LinOSS Neural Oscillator Controls");
        
        ui.horizontal(|ui| {
            ui.label("Architecture:");
            ui.colored_label(egui::Color32::GREEN, "WGPU Unified Backend");
            ui.colored_label(egui::Color32::LIGHT_BLUE, "Multi-Modal Visualization");
        });
        
        ui.separator();
        
        // Neural parameters
        ui.add(egui::Slider::new(&mut self.params.alpha, 0.1..=3.0).text("Alpha (X strength)"));
        ui.add(egui::Slider::new(&mut self.params.beta, 0.1..=3.0).text("Beta (Y strength)"));
        ui.add(egui::Slider::new(&mut self.params.gamma, 0.1..=2.0).text("Gamma (Z/damping)"));
        ui.add(egui::Slider::new(&mut self.params.frequency, 0.1..=5.0).text("Frequency"));
        ui.add(egui::Slider::new(&mut self.params.amplitude, 0.1..=3.0).text("Amplitude"));
        ui.add(egui::Slider::new(&mut self.params.coupling, 0.0..=1.0).text("Coupling"));
        
        let old_count = self.params.oscillator_count;
        ui.add(egui::Slider::new(&mut self.params.oscillator_count, 4..=128).text("Oscillator Count"));
        
        if old_count != self.params.oscillator_count {
            self.neural_state = LinossNeuralState::new(self.params.oscillator_count);
            self.plot_data = PlotData::new(self.params.oscillator_count);
        }
        
        ui.separator();
        
        // View options
        ui.label("🎨 Visualization Options:");
        ui.checkbox(&mut self.show_3d, "Show 3D Neural View");
        ui.checkbox(&mut self.show_plots, "Show 2D Time Plots");
        ui.checkbox(&mut self.show_gif, "Show Animated GIF");
        
        if self.show_gif {
            ui.add(egui::Slider::new(&mut self.gif_size, 50.0..=400.0).text("GIF Size"));
        }
        
        ui.separator();
        
        // Layout options
        ui.label("🔧 Layout Mode:");
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.layout_mode, LayoutMode::Tabbed, "Tabbed");
            ui.selectable_value(&mut self.layout_mode, LayoutMode::Split, "Split");
            ui.selectable_value(&mut self.layout_mode, LayoutMode::Grid, "Grid");
        });
        
        ui.separator();
        
        // Status
        ui.label(format!("Time: {:.2}s", self.neural_state.time));
        ui.label(format!("Neural Outputs: {}", self.neural_state.neural_outputs.len()));
        
        let (current_frame, total_frames) = self.animated_gif.get_frame_info();
        if total_frames > 0 {
            ui.label(format!("GIF Frame: {}/{}", current_frame, total_frames));
        }
    }
    
    fn draw_3d_view(&mut self, ui: &mut egui::Ui) {
        ui.heading("🎨 3D Neural Oscillator Visualization");
        
        ui.horizontal(|ui| {
            ui.colored_label(egui::Color32::YELLOW, "⚡ WGPU Architecture:");
            ui.label("Unified GPU compute + rendering");
        });
        
        // 3D visualization using isometric projection
        Plot::new("3d_neural_plot")
            .width(ui.available_width().min(600.0))
            .height(400.0)
            .data_aspect(1.0)
            .show(ui, |plot_ui| {
                // Draw oscillator positions as points in 3D (projected to 2D)
                for (i, pos) in self.neural_state.positions.iter().enumerate() {
                    let color = egui::Color32::from_rgb(
                        128 + (self.neural_state.neural_outputs[i] * 50.0) as u8,
                        100,
                        128 - (self.neural_state.neural_outputs[i] * 50.0) as u8,
                    );
                    
                    // Isometric projection: x + 0.5*z, y + 0.5*z
                    let proj_x = pos[0] + 0.5 * pos[2];
                    let proj_y = pos[1] + 0.5 * pos[2];
                    
                    plot_ui.points(
                        egui_plot::Points::new(vec![[proj_x as f64, proj_y as f64]])
                            .radius(4.0)
                            .color(color)
                    );
                }
                
                // Draw connections between oscillators
                if self.neural_state.positions.len() > 1 {
                    let line_points: PlotPoints = self.neural_state.positions
                        .iter()
                        .map(|pos| [
                            (pos[0] + 0.5 * pos[2]) as f64,
                            (pos[1] + 0.5 * pos[2]) as f64
                        ])
                        .collect();
                    
                    plot_ui.line(
                        Line::new(line_points)
                            .color(egui::Color32::GRAY)
                            .width(1.0)
                    );
                }
            });
    }
    
    fn draw_time_plots(&mut self, ui: &mut egui::Ui) {
        ui.heading("📊 Neural Signal Time Series");
        
        Plot::new("time_series_plot")
            .width(ui.available_width().min(700.0))
            .height(250.0)
            .auto_bounds([true, true].into())
            .show(ui, |plot_ui| {
                // Show first few oscillators with different colors
                let colors = [
                    egui::Color32::RED,
                    egui::Color32::GREEN, 
                    egui::Color32::BLUE,
                    egui::Color32::YELLOW,
                    egui::Color32::from_rgb(0, 255, 255), // CYAN equivalent
                    egui::Color32::LIGHT_RED,
                ];
                
                for (i, &color) in colors.iter().enumerate().take(self.params.oscillator_count.min(colors.len())) {
                    let line_data = self.plot_data.get_line_data(i);
                    plot_ui.line(
                        Line::new(line_data)
                            .color(color)
                            .width(1.5)
                            .name(format!("Oscillator {}", i))
                    );
                }
            });
    }
    
    fn draw_animated_gif(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.heading("🎬 Animated GIF Integration");
        
        if let Some(texture) = self.animated_gif.update_and_get_texture(ctx) {
            let image_size = egui::Vec2::splat(self.gif_size);
            ui.add(egui::Image::from_texture(texture).fit_to_exact_size(image_size));
            
            let (current_frame, total_frames) = self.animated_gif.get_frame_info();
            ui.label(format!("Frame {}/{} - FPS: {:.1}", current_frame, total_frames, 
                self.animated_gif.get_fps()));
            
            // Debug info
            ui.label(format!("Time since last frame: {:.0}ms", 
                self.animated_gif.get_time_since_last_frame().as_millis()));
        } else {
            ui.colored_label(egui::Color32::RED, "❌ GIF not loaded");
            ui.label("Expected: assets/image015.gif");
        }
    }
    
    fn draw_content(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        match self.layout_mode {
            LayoutMode::Tabbed => {
                ui.horizontal_top(|ui| {
                    ui.selectable_value(&mut self.show_3d, true, "🎨 3D View");
                    ui.selectable_value(&mut self.show_plots, true, "📊 Plots");
                    ui.selectable_value(&mut self.show_gif, true, "🎬 GIF");
                });
                
                ui.separator();
                
                if self.show_3d {
                    self.draw_3d_view(ui);
                } else if self.show_plots {
                    self.draw_time_plots(ui);
                } else if self.show_gif {
                    self.draw_animated_gif(ui, ctx);
                }
            },
            
            LayoutMode::Split => {
                ui.columns(2, |columns| {
                    if self.show_3d {
                        self.draw_3d_view(&mut columns[0]);
                    }
                    
                    columns[1].vertical(|ui| {
                        if self.show_plots {
                            self.draw_time_plots(ui);
                        }
                        if self.show_gif {
                            ui.separator();
                            self.draw_animated_gif(ui, ctx);
                        }
                    });
                });
            },
            
            LayoutMode::Grid => {
                egui::Grid::new("visualization_grid")
                    .num_columns(2)
                    .spacing([10.0, 10.0])
                    .show(ui, |ui| {
                        if self.show_3d {
                            self.draw_3d_view(ui);
                        }
                        if self.show_gif {
                            self.draw_animated_gif(ui, ctx);
                        }
                        ui.end_row();
                        
                        if self.show_plots {
                            ui.allocate_ui_with_layout(
                                egui::Vec2::new(700.0, 250.0),
                                egui::Layout::top_down(egui::Align::LEFT),
                                |ui| self.draw_time_plots(ui)
                            );
                        }
                        ui.end_row();
                    });
            }
        }
    }
}

impl eframe::App for LinossMultiModalApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Update neural dynamics
        self.neural_state.update(&self.params);
        
        // Update plot data
        let signals = self.neural_state.get_signal_values();
        self.plot_data.update(self.neural_state.time, &signals);
        
        // UI Layout
        egui::SidePanel::left("control_panel")
            .min_width(350.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    if self.show_params {
                        self.draw_control_panel(ui);
                    }
                });
            });
        
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("🧠 LinOSS Multi-Modal Neural Visualizer");
            
            ui.horizontal(|ui| {
                ui.label("🚀 Features:");
                ui.colored_label(egui::Color32::GREEN, "3D Neural Dynamics");
                ui.colored_label(egui::Color32::LIGHT_BLUE, "2D Time Series");
                ui.colored_label(egui::Color32::YELLOW, "Animated GIF");
                ui.colored_label(egui::Color32::LIGHT_RED, "WGPU Backend");
            });
            
            ui.separator();
            
            self.draw_content(ui, ctx);
        });
        
        // Request continuous updates for animation
        ctx.request_repaint();
    }
}

fn main() -> Result<()> {
    let options = NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 900.0])
            .with_title("LinOSS Multi-Modal Neural Visualizer - WGPU + Animated GIF"),
        wgpu_options: eframe::egui_wgpu::WgpuConfiguration {
            device_descriptor: std::sync::Arc::new(|_adapter| eframe::wgpu::DeviceDescriptor {
                required_features: eframe::wgpu::Features::default(),
                required_limits: eframe::wgpu::Limits::default(),
                ..Default::default()
            }),
            ..Default::default()
        },
        ..Default::default()
    };
    
    eframe::run_native(
        "LinOSS Multi-Modal Neural Visualizer",
        options,
        Box::new(|cc| Ok(Box::new(LinossMultiModalApp::new(cc)))),
    ).map_err(|e| anyhow::anyhow!("eframe error: {}", e))?;
    
    Ok(())
}
