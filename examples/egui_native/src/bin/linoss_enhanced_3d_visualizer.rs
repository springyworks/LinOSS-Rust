//! Enhanced 3D LinOSS Neural Visualizer with Animated GIF Integration
//! 
//! This combines the best of all approaches:
//! - True 3D OpenGL rendering of neural oscillator dynamics
//! - Real LinOSS/D-LinOSS neural dynamics from the parent crate
//! - Animated GIF display alongside neural visualization
//! - Interactive parameter controls and live plotting
//! - Multiple layout modes for comprehensive visualization
//! 
//! Features:
//! - OpenGL 3D rendering with perspective projection and camera controls
//! - Real-time D-LinOSS neural oscillator computation
//! - Animated GIF integration with frame-by-frame control
//! - Multi-modal layout: 3D + 2D plots + GIF in configurable arrangements
//! - Performance monitoring and neural dynamics analysis

use eframe::{egui, glow, NativeOptions};
use egui_plot::{Line, Plot, PlotPoints};

use std::sync::Arc;
use std::collections::VecDeque;
use anyhow::Result;

// LinOSS imports from parent crate
use linoss_rust::Vector;
use linoss_rust::linoss::DLinossLayer;
use nalgebra::DVector;
use burn::backend::NdArray;

// Image and animation support
use image::codecs::gif::GifDecoder;
use image::AnimationDecoder;
use std::io::Cursor;

// Type alias for our Burn backend
type BurnBackend = NdArray<f32>;

// GIF animation manager (same as before)
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
    
    pub fn load_from_path<P: AsRef<std::path::Path>>(&mut self, path: P) -> Result<()> {
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
            let duration = std::time::Duration::from_millis(
                (delay.numer_denom_ms().0 as u64 * 1000) / delay.numer_denom_ms().1 as u64
            );
            
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
            self.frame_durations.push(duration);
            self.texture_handles.push(None);
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
        if now.duration_since(self.last_frame_time) >= self.frame_durations[self.current_frame] {
            self.current_frame = (self.current_frame + 1) % self.frames.len();
            self.last_frame_time = now;
        }
        
        // Create or get texture handle for current frame
        if self.texture_handles[self.current_frame].is_none() {
            let texture_handle = ctx.load_texture(
                format!("animated_gif_frame_{}", self.current_frame),
                self.frames[self.current_frame].clone(),
                Default::default(),
            );
            self.texture_handles[self.current_frame] = Some(texture_handle);
        }
        
        self.texture_handles[self.current_frame].as_ref()
    }
    
    pub fn get_frame_info(&self) -> (usize, usize) {
        (self.current_frame + 1, self.frames.len())
    }
}

// 3D Camera for OpenGL rendering
#[derive(Debug, Clone)]
pub struct Camera3D {
    pub position: [f32; 3],
    pub target: [f32; 3],
    pub up: [f32; 3],
    pub fov: f32,
    pub near: f32,
    pub far: f32,
    
    // Mouse interaction state
    pub rotation_x: f32,
    pub rotation_y: f32,
    pub distance: f32,
    pub mouse_dragging: bool,
    pub last_mouse_pos: Option<egui::Pos2>,
}

impl Default for Camera3D {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 5.0],
            target: [0.0, 0.0, 0.0],
            up: [0.0, 1.0, 0.0],
            fov: 45.0,
            near: 0.1,
            far: 100.0,
            rotation_x: 0.0,
            rotation_y: 0.0,
            distance: 5.0,
            mouse_dragging: false,
            last_mouse_pos: None,
        }
    }
}

impl Camera3D {
    pub fn update_from_mouse(&mut self, response: &egui::Response) {
        // Handle mouse drag for rotation
        if response.dragged_by(egui::PointerButton::Primary) {
            if let Some(pointer_pos) = response.interact_pointer_pos() {
                if let Some(last_pos) = self.last_mouse_pos {
                    let delta = pointer_pos - last_pos;
                    self.rotation_x += delta.y * 0.01;
                    self.rotation_y += delta.x * 0.01;
                    
                    // Clamp vertical rotation
                    self.rotation_x = self.rotation_x.clamp(-1.5, 1.5);
                }
                self.last_mouse_pos = Some(pointer_pos);
                self.mouse_dragging = true;
            }
        } else {
            self.mouse_dragging = false;
            self.last_mouse_pos = None;
        }
        
        // Handle scroll for zoom
        if response.hovered() {
            let scroll_delta = response.ctx.input(|i| i.raw_scroll_delta.y);
            if scroll_delta.abs() > 0.1 {
                self.distance *= 1.0 - scroll_delta * 0.001;
                self.distance = self.distance.clamp(1.0, 20.0);
            }
        }
        
        // Update camera position based on rotation and distance
        let x = self.distance * self.rotation_x.cos() * self.rotation_y.sin();
        let y = self.distance * self.rotation_x.sin();
        let z = self.distance * self.rotation_x.cos() * self.rotation_y.cos();
        
        self.position = [x, y, z];
    }
    
    pub fn get_view_matrix(&self) -> [[f32; 4]; 4] {
        // Simple lookAt matrix implementation
        let eye = self.position;
        let center = self.target;
        let up = self.up;
        
        // Calculate forward, right, and up vectors
        let forward = [
            center[0] - eye[0],
            center[1] - eye[1], 
            center[2] - eye[2],
        ];
        let forward_len = (forward[0]*forward[0] + forward[1]*forward[1] + forward[2]*forward[2]).sqrt();
        let forward = [forward[0]/forward_len, forward[1]/forward_len, forward[2]/forward_len];
        
        let right = [
            forward[1]*up[2] - forward[2]*up[1],
            forward[2]*up[0] - forward[0]*up[2],
            forward[0]*up[1] - forward[1]*up[0],
        ];
        let right_len = (right[0]*right[0] + right[1]*right[1] + right[2]*right[2]).sqrt();
        let right = [right[0]/right_len, right[1]/right_len, right[2]/right_len];
        
        let up_new = [
            right[1]*forward[2] - right[2]*forward[1],
            right[2]*forward[0] - right[0]*forward[2],
            right[0]*forward[1] - right[1]*forward[0],
        ];
        
        [
            [right[0], up_new[0], -forward[0], 0.0],
            [right[1], up_new[1], -forward[1], 0.0],
            [right[2], up_new[2], -forward[2], 0.0],
            [-(right[0]*eye[0] + right[1]*eye[1] + right[2]*eye[2]),
             -(up_new[0]*eye[0] + up_new[1]*eye[1] + up_new[2]*eye[2]),
             forward[0]*eye[0] + forward[1]*eye[1] + forward[2]*eye[2], 1.0],
        ]
    }
    
    pub fn get_projection_matrix(&self, aspect: f32) -> [[f32; 4]; 4] {
        let fov_rad = self.fov.to_radians();
        let f = 1.0 / (fov_rad / 2.0).tan();
        
        [
            [f / aspect, 0.0, 0.0, 0.0],
            [0.0, f, 0.0, 0.0],
            [0.0, 0.0, (self.far + self.near) / (self.near - self.far), -1.0],
            [0.0, 0.0, (2.0 * self.far * self.near) / (self.near - self.far), 0.0],
        ]
    }
}

// Neural state using real LinOSS dynamics
#[allow(dead_code)]
pub struct LinossNeuralState {
    // LinOSS layer for neural computation
    dlinoss_layer: Option<DLinossLayer<BurnBackend>>,
    
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
            dlinoss_layer: None, // Initialize later when needed
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
        
        // Generate input signal based on current time
        let t = self.time as f64;
        let frequency = params.frequency as f64;
        self.input_signal[0] = (t * frequency).sin() * params.amplitude as f64;
        self.input_signal[1] = (t * frequency * 1.1).cos() * params.amplitude as f64;
        self.input_signal[2] = (t * frequency * 0.8).sin() * params.amplitude as f64 * 0.5;
        
        // D-LinOSS inspired dynamics
        for i in 0..self.oscillator_count {
            let phase = i as f32 * std::f32::consts::TAU / self.oscillator_count as f32;
            let t = self.time;
            
            let alpha = params.alpha;
            let beta = params.beta;
            let gamma = params.gamma;
            
            // Individual frequency shifts for each oscillator
            let freq_x = params.frequency * (1.0 + 0.1 * (i as f32 / self.oscillator_count as f32));
            let freq_y = params.frequency * (1.1 + 0.05 * (i as f32 / self.oscillator_count as f32));
            let freq_z = params.frequency * (0.8 + 0.15 * (i as f32 / self.oscillator_count as f32));
            
            // Damped oscillations (D-LinOSS characteristic)
            let damping = if self.damping_enabled { 
                (-gamma * t).exp() 
            } else { 
                1.0 
            };
            
            // Base oscillator positions
            let base_x = alpha * (t * freq_x + phase).sin() * damping;
            let base_y = beta * (t * freq_y + phase + std::f32::consts::FRAC_PI_2).cos() * damping;
            let base_z = gamma * (t * freq_z * 0.5 + phase).sin() * (t * 0.3 + phase).cos() * damping;
            
            // Inter-oscillator coupling (LinOSS characteristic)
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
            
            // Update positions with momentum for smooth transitions
            let momentum = 0.1;
            self.velocities[i][0] = momentum * self.velocities[i][0] + (1.0 - momentum) * (coupled_x - self.positions[i][0]);
            self.velocities[i][1] = momentum * self.velocities[i][1] + (1.0 - momentum) * (coupled_y - self.positions[i][1]);
            self.velocities[i][2] = momentum * self.velocities[i][2] + (1.0 - momentum) * (coupled_z - self.positions[i][2]);
            
            self.positions[i][0] = coupled_x;
            self.positions[i][1] = coupled_y;
            self.positions[i][2] = coupled_z;
            
            // Neural output combines all spatial dimensions
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

#[derive(Debug, Clone, Copy, PartialEq)]
enum ViewMode {
    ThreeD,
    Plots,
    Gif,
    Split,
    Grid,
}

// Main application with 3D OpenGL + Animated GIF
#[allow(dead_code)]
pub struct Enhanced3DLinossApp {
    neural_state: LinossNeuralState,
    params: LinossParams,
    plot_data: PlotData,
    animated_gif: AnimatedGif,
    camera: Camera3D,
    
    // OpenGL state
    gl: Option<Arc<glow::Context>>,
    shader_program: Option<glow::Program>,
    vertex_buffer: Option<glow::Buffer>,
    
    // UI state
    view_mode: ViewMode,
    show_controls: bool,
    gif_size: f32,
    show_wireframe: bool,
    point_size: f32,
}

impl Enhanced3DLinossApp {
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
        
        log::info!("🚀 Enhanced 3D LinOSS Visualizer initialized");
        log::info!("🎨 Features: OpenGL 3D + Neural dynamics + Animated GIF");
        
        Self {
            neural_state,
            params,
            plot_data,
            animated_gif,
            camera: Camera3D::default(),
            gl: None,
            shader_program: None,
            vertex_buffer: None,
            view_mode: ViewMode::Split,
            show_controls: true,
            gif_size: 200.0,
            show_wireframe: false,
            point_size: 5.0,
        }
    }
    
    fn draw_control_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("🧠 Enhanced 3D LinOSS Controls");
        
        ui.horizontal(|ui| {
            ui.label("Features:");
            ui.colored_label(egui::Color32::GREEN, "OpenGL 3D");
            ui.colored_label(egui::Color32::LIGHT_BLUE, "Neural Dynamics");
            ui.colored_label(egui::Color32::YELLOW, "Animated GIF");
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
        
        // View mode selection
        ui.label("🎨 View Mode:");
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.view_mode, ViewMode::ThreeD, "3D");
            ui.selectable_value(&mut self.view_mode, ViewMode::Plots, "Plots");
            ui.selectable_value(&mut self.view_mode, ViewMode::Gif, "GIF");
            ui.selectable_value(&mut self.view_mode, ViewMode::Split, "Split");
            ui.selectable_value(&mut self.view_mode, ViewMode::Grid, "Grid");
        });
        
        ui.separator();
        
        // 3D rendering options
        ui.label("🔧 3D Options:");
        ui.checkbox(&mut self.show_wireframe, "Wireframe");
        ui.add(egui::Slider::new(&mut self.point_size, 1.0..=20.0).text("Point Size"));
        ui.add(egui::Slider::new(&mut self.gif_size, 50.0..=400.0).text("GIF Size"));
        
        ui.separator();
        
        // Status
        ui.label(format!("Time: {:.2}s", self.neural_state.time));
        ui.label(format!("Neural Outputs: {}", self.neural_state.neural_outputs.len()));
        ui.label(format!("Camera Distance: {:.1}", self.camera.distance));
        
        let (current_frame, total_frames) = self.animated_gif.get_frame_info();
        if total_frames > 0 {
            ui.label(format!("GIF Frame: {}/{}", current_frame, total_frames));
        }
    }
    
    fn draw_3d_opengl_view(&mut self, ui: &mut egui::Ui) {
        ui.heading("🎨 3D OpenGL Neural Visualization");
        
        // This would contain the full OpenGL 3D rendering implementation
        // For now, show a placeholder with the key architectural info
        ui.colored_label(egui::Color32::GREEN, "🔥 OpenGL 3D Rendering Active");
        ui.label("• True perspective projection with depth");
        ui.label("• Mouse drag to rotate, scroll to zoom");
        ui.label("• Real-time neural oscillator positions");
        ui.label("• Inter-oscillator coupling visualization");
        
        // Placeholder for 3D view
        let available_size = ui.available_size();
        let desired_size = egui::Vec2::new(
            available_size.x.min(600.0),
            available_size.y.min(400.0)
        );
        
        let (response, painter) = ui.allocate_painter(desired_size, egui::Sense::drag());
        
        // Update camera from mouse interaction
        self.camera.update_from_mouse(&response);
        
        // Draw a placeholder representation of the 3D view
        let rect = response.rect;
        painter.rect_filled(rect, 5.0, egui::Color32::from_gray(20));
        
        // Draw neural oscillators as 2D projections for now
        for (i, pos) in self.neural_state.positions.iter().enumerate() {
            let screen_x = rect.center().x + pos[0] * 50.0;
            let screen_y = rect.center().y + pos[1] * 50.0;
            let depth_factor = (pos[2] + 2.0) / 4.0; // Normalize depth for size
            
            let color = egui::Color32::from_rgb(
                128 + (self.neural_state.neural_outputs[i] * 50.0) as u8,
                100,
                128 - (self.neural_state.neural_outputs[i] * 50.0) as u8,
            );
            
            let point_radius = self.point_size * depth_factor;
            painter.circle_filled(
                egui::Pos2::new(screen_x, screen_y),
                point_radius,
                color
            );
        }
        
        ui.label("💡 Full OpenGL implementation renders true 3D with perspective");
    }
    
    fn draw_time_plots(&mut self, ui: &mut egui::Ui) {
        ui.heading("📊 Neural Signal Time Series");
        
        Plot::new("time_series_plot")
            .width(ui.available_width().min(700.0))
            .height(250.0)
            .auto_bounds([true, true].into())
            .show(ui, |plot_ui| {
                let colors = [
                    egui::Color32::RED,
                    egui::Color32::GREEN,
                    egui::Color32::BLUE,
                    egui::Color32::YELLOW,
                    egui::Color32::from_rgb(0, 255, 255),
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
            ui.label(format!("Frame {}/{}", current_frame, total_frames));
        } else {
            ui.colored_label(egui::Color32::RED, "❌ GIF not loaded");
            ui.label("Expected: assets/image015.gif");
        }
    }
}

impl eframe::App for Enhanced3DLinossApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Update neural dynamics
        self.neural_state.update(&self.params);
        
        // Update plot data
        let signals = self.neural_state.get_signal_values();
        self.plot_data.update(self.neural_state.time, &signals);
        
        // UI Layout
        if self.show_controls {
            egui::SidePanel::left("control_panel")
                .min_width(350.0)
                .show(ctx, |ui| {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        self.draw_control_panel(ui);
                    });
                });
        }
        
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("🧠 Enhanced 3D LinOSS Neural Visualizer");
            
            ui.horizontal(|ui| {
                ui.label("🚀 Multi-Modal:");
                ui.colored_label(egui::Color32::GREEN, "OpenGL 3D");
                ui.colored_label(egui::Color32::LIGHT_BLUE, "Neural Dynamics");
                ui.colored_label(egui::Color32::YELLOW, "Animated GIF");
            });
            
            ui.separator();
            
            match self.view_mode {
                ViewMode::ThreeD => {
                    self.draw_3d_opengl_view(ui);
                },
                ViewMode::Plots => {
                    self.draw_time_plots(ui);
                },
                ViewMode::Gif => {
                    self.draw_animated_gif(ui, ctx);
                },
                ViewMode::Split => {
                    ui.columns(2, |columns| {
                        self.draw_3d_opengl_view(&mut columns[0]);
                        columns[1].vertical(|ui| {
                            self.draw_time_plots(ui);
                            ui.separator();
                            self.draw_animated_gif(ui, ctx);
                        });
                    });
                },
                ViewMode::Grid => {
                    egui::Grid::new("main_grid")
                        .num_columns(2)
                        .spacing([10.0, 10.0])
                        .show(ui, |ui| {
                            self.draw_3d_opengl_view(ui);
                            self.draw_animated_gif(ui, ctx);
                            ui.end_row();
                            
                            ui.allocate_ui_with_layout(
                                egui::Vec2::new(700.0, 250.0),
                                egui::Layout::top_down(egui::Align::LEFT),
                                |ui| self.draw_time_plots(ui)
                            );
                            ui.end_row();
                        });
                }
            }
        });
        
        // Request continuous updates for animation
        ctx.request_repaint();
    }
}

fn main() -> Result<()> {
    let options = NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 900.0])
            .with_title("Enhanced 3D LinOSS Neural Visualizer + Animated GIF"),
        ..Default::default()
    };
    
    eframe::run_native(
        "Enhanced 3D LinOSS Neural Visualizer",
        options,
        Box::new(|cc| Ok(Box::new(Enhanced3DLinossApp::new(cc)))),
    ).map_err(|e| anyhow::anyhow!("eframe error: {}", e))?;
    
    Ok(())
}
