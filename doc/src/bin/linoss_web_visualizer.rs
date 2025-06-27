// LinOSS Web Visualizer - Combining egui and simple D-LinOSS for web deployment
// This is a comprehensive neural oscillator visualization designed for GitHub Pages

use eframe::egui::{self, Color32, Context, RichText, Ui};
use eframe::egui_plot::{Line, Plot, PlotPoints};
use eframe::{App, Frame};
use nalgebra::Vector3;
use std::collections::VecDeque;
use std::time::Instant;

// Import our simple D-LinOSS implementation
use linoss_web_demo::{SimpleLinOSS, LinOSSParams, SimpleRng};

const MAX_PLOT_POINTS: usize = 1000;
const UPDATE_FREQUENCY: f64 = 60.0; // Hz

#[derive(Debug, Clone)]
pub struct VisualizationParams {
    pub frequency: f32,
    pub damping: f32,
    pub coupling: f32,
    pub nonlinearity: f32,
    pub noise_level: f32,
    pub time_step: f32,
    pub enable_3d: bool,
}

impl Default for LinossParams {
    fn default() -> Self {
        Self {
            oscillator_count: OSCILLATOR_COUNT,
            frequency: 10.0,
            damping: 0.1,
            coupling: 0.05,
            noise_level: 0.01,
            time_step: 1.0 / UPDATE_FREQUENCY,
            enable_3d: true,
            enable_gpu: false, // Start with CPU for web compatibility
        }
    }
}

#[derive(Clone)]
pub struct PlotData {
    pub time_series: Vec<VecDeque<[f64; 2]>>,
    pub time: f64,
}

impl PlotData {
    pub fn new(oscillator_count: usize) -> Self {
        Self {
            time_series: vec![VecDeque::with_capacity(MAX_PLOT_POINTS); oscillator_count],
            time: 0.0,
        }
    }

    pub fn add_point(&mut self, oscillator_idx: usize, value: f64) {
        if oscillator_idx < self.time_series.len() {
            let series = &mut self.time_series[oscillator_idx];
            series.push_back([self.time, value]);
            if series.len() > MAX_PLOT_POINTS {
                series.pop_front();
            }
        }
    }

    pub fn update_time(&mut self, dt: f64) {
        self.time += dt;
    }

    pub fn get_line_data(&self, oscillator_idx: usize) -> PlotPoints {
        if oscillator_idx < self.time_series.len() {
            PlotPoints::from_iter(self.time_series[oscillator_idx].iter().copied())
        } else {
            PlotPoints::new(Vec::new())
        }
    }

    pub fn clear(&mut self) {
        for series in &mut self.time_series {
            series.clear();
        }
        self.time = 0.0;
    }
}

pub struct LinossWebVisualizerApp {
    params: LinossParams,
    plot_data: PlotData,
    neural_state: Option<DLinossLayer<BurnBackend>>,
    positions: Vec<Vector3<f32>>,
    velocities: Vec<Vector3<f32>>,
    rng: SimpleRng,  // Simple RNG instead of rand
    last_update: Instant,
    is_running: bool,
    show_controls: bool,
    show_3d: bool,
    show_neural: bool,
    performance_fps: f32,
    frame_count: u64,
    last_fps_update: Instant,
}

impl Default for LinossWebVisualizerApp {
    fn default() -> Self {
        let params = LinossParams::default();
        let plot_data = PlotData::new(params.oscillator_count);
        
        // Initialize positions in a circular arrangement
        let mut positions = Vec::new();
        let mut velocities = Vec::new();
        
        for i in 0..params.oscillator_count {
            let angle = 2.0 * std::f32::consts::PI * i as f32 / params.oscillator_count as f32;
            positions.push(Vector3::new(
                angle.cos() * 2.0,
                angle.sin() * 2.0,
                0.0
            ));
            velocities.push(Vector3::new(0.0, 0.0, 0.0));
        }

        Self {
            params,
            plot_data,
            neural_state: None,
            positions,
            velocities,
            rng: SimpleRng::new(),
            last_update: Instant::now(),
            is_running: true,
            show_controls: true,
            show_3d: true,
            show_neural: true,
            performance_fps: 0.0,
            frame_count: 0,
            last_fps_update: Instant::now(),
        }
    }
}

impl LinossWebVisualizerApp {
    pub fn new() -> Self {
        Self::default()
    }

    fn initialize_neural_layer(&mut self) {
        if self.params.enable_gpu {
            // Try to initialize DLinOSS layer
            let config = DLinossLayerConfig {
                d_input: self.params.oscillator_count * 2, // position + velocity
                d_model: self.params.oscillator_count * 2,
                d_output: self.params.oscillator_count,
                delta_t: self.params.time_step,
                init_std: 0.1,
                enable_layer_norm: true,
                enable_damping: true,
                init_damping: self.params.damping,
                num_damping_scales: 4,
                a_parameterization: AParameterization::Diagonal,
            };
            
            match DLinossLayer::new(config, &Default::default()) {
                Ok(layer) => {
                    self.neural_state = Some(layer);
                    #[cfg(target_arch = "wasm32")]
                    web_sys::console::log_1(&"✅ Neural layer initialized successfully".into());
                }
                Err(e) => {
                    #[cfg(target_arch = "wasm32")]
                    web_sys::console::log_1(&format!("❌ Failed to initialize neural layer: {}", e).into());
                }
            }
        }
    }

    fn update_simulation(&mut self) {
        let now = Instant::now();
        let dt = now.duration_since(self.last_update).as_secs_f64();
        self.last_update = now;

        if !self.is_running || dt > 0.1 { // Skip if paused or too much time passed
            return;
        }

        // Update oscillator dynamics
        for i in 0..self.params.oscillator_count {
            // Simple coupled oscillator dynamics
            let mut force = Vector3::new(0.0, 0.0, 0.0);
            
            // Spring force (harmonic oscillator)
            let omega = 2.0 * std::f32::consts::PI * self.params.frequency as f32;
            force -= self.positions[i] * omega * omega;
            
            // Damping
            force -= self.velocities[i] * self.params.damping as f32;
            
            // Coupling with neighbors
            for j in 0..self.params.oscillator_count {
                if i != j {
                    let diff = self.positions[j] - self.positions[i];
                    force += diff * self.params.coupling as f32;
                }
            }
            
            // Add noise
            if self.params.noise_level > 0.0 {
                force += Vector3::new(
                    (self.simple_rng.next_f32() - 0.5) * self.params.noise_level as f32,
                    (self.simple_rng.next_f32() - 0.5) * self.params.noise_level as f32,
                    (self.simple_rng.next_f32() - 0.5) * self.params.noise_level as f32,
                );
            }
            
            // Update velocity and position
            self.velocities[i] += force * dt as f32;
            self.positions[i] += self.velocities[i] * dt as f32;
            
            // Add to plot data
            let amplitude = self.positions[i].magnitude();
            self.plot_data.add_point(i, amplitude as f64);
        }
        
        self.plot_data.update_time(dt);
        
        // Update FPS counter
        self.frame_count += 1;
        if now.duration_since(self.last_fps_update).as_secs_f64() >= 1.0 {
            self.performance_fps = self.frame_count as f32;
            self.frame_count = 0;
            self.last_fps_update = now;
        }
    }

    fn draw_controls(&mut self, ui: &mut Ui) {
        ui.collapsing("🎛️ Simulation Controls", |ui| {
            ui.horizontal(|ui| {
                if ui.button(if self.is_running { "⏸️ Pause" } else { "▶️ Play" }).clicked() {
                    self.is_running = !self.is_running;
                }
                if ui.button("🔄 Reset").clicked() {
                    self.plot_data.clear();
                    // Reset positions
                    for i in 0..self.params.oscillator_count {
                        let angle = 2.0 * std::f32::consts::PI * i as f32 / self.params.oscillator_count as f32;
                        self.positions[i] = Vector3::new(
                            angle.cos() * 2.0,
                            angle.sin() * 2.0,
                            0.0
                        );
                        self.velocities[i] = Vector3::new(0.0, 0.0, 0.0);
                    }
                }
            });
            
            ui.separator();
            
            ui.add(egui::Slider::new(&mut self.params.frequency, 0.1..=50.0)
                .text("Frequency (Hz)")
                .logarithmic(true));
            
            ui.add(egui::Slider::new(&mut self.params.damping, 0.0..=1.0)
                .text("Damping"));
            
            ui.add(egui::Slider::new(&mut self.params.coupling, 0.0..=0.5)
                .text("Coupling"));
            
            ui.add(egui::Slider::new(&mut self.params.noise_level, 0.0..=0.1)
                .text("Noise Level"));
            
            ui.separator();
            
            ui.checkbox(&mut self.params.enable_gpu, "🚀 Enable Neural Layer (GPU)");
            if self.params.enable_gpu && self.neural_state.is_none() {
                if ui.button("Initialize Neural Layer").clicked() {
                    self.initialize_neural_layer();
                }
            }
        });
    }

    fn draw_visualization_panels(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.show_3d, "📊 3D View");
            ui.checkbox(&mut self.show_neural, "🧠 Neural Signals");
            ui.label(format!("FPS: {:.1}", self.performance_fps));
        });
        
        if self.show_3d {
            self.draw_3d_visualization(ui);
        }
        
        if self.show_neural {
            self.draw_neural_signals(ui);
        }
    }

    fn draw_3d_visualization(&mut self, ui: &mut Ui) {
        ui.collapsing("🌐 3D Oscillator Visualization", |ui| {
            Plot::new("3d_oscillators")
                .height(300.0)
                .data_aspect(1.0)
                .show(ui, |plot_ui| {
                    // Convert 3D positions to 2D isometric projection
                    let projected_points: Vec<[f64; 2]> = self.positions
                        .iter()
                        .map(|pos| {
                            // Isometric projection
                            let x = pos.x - pos.y * 0.5;
                            let y = pos.z + (pos.x + pos.y) * 0.3;
                            [x as f64, y as f64]
                        })
                        .collect();
                    
                    // Draw oscillator positions
                    for (i, &point) in projected_points.iter().enumerate() {
                        let color = match i % 6 {
                            0 => Color32::RED,
                            1 => Color32::GREEN,
                            2 => Color32::BLUE,
                            3 => Color32::YELLOW,
                            4 => Color32::from_rgb(255, 0, 255), // MAGENTA
                            _ => Color32::from_rgb(0, 255, 255), // CYAN
                        };
                        
                        plot_ui.points(
                            egui_plot::Points::new(vec![point])
                                .color(color)
                                .radius(8.0)
                                .name(format!("Oscillator {}", i))
                        );
                    }
                    
                    // Draw connections between oscillators
                    if projected_points.len() > 1 {
                        plot_ui.line(
                            Line::new(PlotPoints::from_iter(projected_points.iter().copied()))
                                .color(Color32::GRAY)
                                .width(1.0)
                                .style(egui_plot::LineStyle::Dashed { length: 5.0 })
                        );
                    }
                });
        });
    }

    fn draw_neural_signals(&mut self, ui: &mut Ui) {
        ui.collapsing("📈 Neural Signal Time Series", |ui| {
            Plot::new("neural_signals")
                .height(400.0)
                .auto_bounds([true, true].into())
                .show(ui, |plot_ui| {
                    let colors = [
                        Color32::RED,
                        Color32::GREEN,
                        Color32::BLUE,
                        Color32::YELLOW,
                        Color32::from_rgb(255, 0, 255), // MAGENTA
                        Color32::from_rgb(0, 255, 255), // CYAN
                    ];
                    
                    for (i, &color) in colors.iter().enumerate().take(self.params.oscillator_count) {
                        let line_data = self.plot_data.get_line_data(i);
                        plot_ui.line(
                            Line::new(line_data)
                                .color(color)
                                .width(1.5)
                                .name(format!("Oscillator {}", i))
                        );
                    }
                });
        });
    }
}

impl App for LinossWebVisualizerApp {
    fn update(&mut self, ctx: &Context, _frame: &mut Frame) {
        // Update simulation
        self.update_simulation();
        
        // Request repaint for animation
        ctx.request_repaint();
        
        // Main UI
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("🧠 LinOSS Web Visualizer");
            ui.label("Interactive Neural Oscillator Dynamics");
            ui.separator();
            
            // Show controls
            if self.show_controls {
                self.draw_controls(ui);
                ui.separator();
            }
            
            // Show visualization panels
            self.draw_visualization_panels(ui);
            
            ui.separator();
            
            // Footer with info
            ui.horizontal(|ui| {
                ui.label("🔬 LinOSS Neural Dynamics");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("ℹ️").clicked() {
                        self.show_controls = !self.show_controls;
                    }
                    ui.label(RichText::new("Real-time neural oscillator simulation").small());
                });
            });
        });
    }
}

// Entry point for web
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn start_web_app() {
    console_error_panic_hook::set_once();
    
    let web_options = eframe::WebOptions::default();
    
    wasm_bindgen_futures::spawn_local(async {
        eframe::WebRunner::new()
            .start(
                "linoss_canvas",
                web_options,
                Box::new(|_cc| Ok(Box::new(LinossWebVisualizerApp::new()))),
            )
            .await
            .expect("Failed to start eframe web app");
    });
}

// Entry point for native
#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result<()> {
    env_logger::init();
    
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 800.0])
            .with_title("LinOSS Web Visualizer"),
        ..Default::default()
    };
    
    eframe::run_native(
        "LinOSS Web Visualizer",
        options,
        Box::new(|_cc| Ok(Box::new(LinossWebVisualizerApp::new()))),
    )
}
