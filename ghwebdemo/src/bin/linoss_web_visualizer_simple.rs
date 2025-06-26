// LinOSS Web Visualizer - Simple WASM-compatible version
// Combining egui and simple D-LinOSS for web deployment

use eframe::egui::{self, Color32, Context, RichText, Ui};
use egui_plot::{Line, Plot, PlotPoints};
use eframe::{App, Frame};
use nalgebra::Vector3;
use std::collections::VecDeque;
use std::time::Instant;

// Inline simple D-LinOSS implementation (WASM-compatible)
/// Simple RNG for generating noise (WASM-compatible)
#[derive(Clone)]
pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    
    pub fn next_f32(&mut self) -> f32 {
        // Linear congruential generator
        self.state = self.state.wrapping_mul(1103515245).wrapping_add(12345);
        ((self.state >> 16) & 0x7fff) as f32 / 32768.0
    }
    
    pub fn next_f32_range(&mut self, min: f32, max: f32) -> f32 {
        min + (max - min) * self.next_f32()
    }
}

/// Simple D-LinOSS oscillator for web demo
#[derive(Clone)]
pub struct SimpleLinOSS {
    pub state: Vector3<f32>,
    pub velocity: Vector3<f32>,
    pub params: LinOSSParams,
    pub time: f32,
    pub dt: f32,
    pub rng: SimpleRng,
}

#[derive(Clone)]
pub struct LinOSSParams {
    pub frequency: f32,
    pub damping: f32,
    pub coupling: f32,
    pub nonlinearity: f32,
    pub noise_level: f32,
}

impl Default for LinOSSParams {
    fn default() -> Self {
        Self {
            frequency: 10.0,
            damping: 0.1,
            coupling: 0.5,
            nonlinearity: 0.8,
            noise_level: 0.01,
        }
    }
}

impl SimpleLinOSS {
    pub fn new(params: LinOSSParams) -> Self {
        Self {
            state: Vector3::new(0.1, 0.0, 0.0),
            velocity: Vector3::zeros(),
            params,
            time: 0.0,
            dt: 0.001,
            rng: SimpleRng::new(42),
        }
    }
    
    pub fn step(&mut self) {
        // D-LinOSS dynamics: simplified version
        let omega = 2.0 * std::f32::consts::PI * self.params.frequency;
        let gamma = self.params.damping;
        let beta = self.params.nonlinearity;
        let coupling = self.params.coupling;
        
        // Coupling matrix (simplified)
        let coupling_matrix = nalgebra::Matrix3::new(
            0.0, coupling, -coupling,
            -coupling, 0.0, coupling,
            coupling, -coupling, 0.0
        );
        
        // Nonlinear term
        let nonlinear = Vector3::new(
            beta * self.state.x * (self.state.x * self.state.x - 1.0),
            beta * self.state.y * (self.state.y * self.state.y - 1.0),
            beta * self.state.z * (self.state.z * self.state.z - 1.0),
        );
        
        // Noise
        let noise = Vector3::new(
            self.rng.next_f32_range(-1.0, 1.0) * self.params.noise_level,
            self.rng.next_f32_range(-1.0, 1.0) * self.params.noise_level,
            self.rng.next_f32_range(-1.0, 1.0) * self.params.noise_level,
        );
        
        // Update equations (simplified Euler integration)
        let acceleration = -omega * omega * self.state 
            - 2.0 * gamma * omega * self.velocity
            + coupling_matrix * self.state
            + nonlinear
            + noise;
        
        self.velocity += acceleration * self.dt;
        self.state += self.velocity * self.dt;
        self.time += self.dt;
        
        // Apply some bounds to prevent explosion
        for i in 0..3 {
            if self.state[i].abs() > 10.0 {
                self.state[i] = self.state[i].signum() * 10.0;
                self.velocity[i] *= 0.5;
            }
        }
    }
    
    pub fn reset(&mut self) {
        self.state = Vector3::new(0.1, 0.0, 0.0);
        self.velocity = Vector3::zeros();
        self.time = 0.0;
        self.rng = SimpleRng::new(42);
    }
}

const MAX_PLOT_POINTS: usize = 1000;

#[derive(Debug)]
pub struct PlotData {
    pub x_series: VecDeque<[f64; 2]>, // [time, value] pairs
    pub y_series: VecDeque<[f64; 2]>,
    pub z_series: VecDeque<[f64; 2]>,
    pub time: f64,
}

impl PlotData {
    pub fn new() -> Self {
        Self {
            x_series: VecDeque::with_capacity(MAX_PLOT_POINTS),
            y_series: VecDeque::with_capacity(MAX_PLOT_POINTS),
            z_series: VecDeque::with_capacity(MAX_PLOT_POINTS),
            time: 0.0,
        }
    }

    pub fn add_point(&mut self, state: Vector3<f32>) {
        let time = self.time;
        
        self.x_series.push_back([time, state.x as f64]);
        self.y_series.push_back([time, state.y as f64]);
        self.z_series.push_back([time, state.z as f64]);
        
        if self.x_series.len() > MAX_PLOT_POINTS {
            self.x_series.pop_front();
            self.y_series.pop_front();
            self.z_series.pop_front();
        }
    }

    pub fn update_time(&mut self, dt: f64) {
        self.time += dt;
    }

    pub fn get_x_line(&self) -> PlotPoints {
        PlotPoints::from_iter(self.x_series.iter().copied())
    }

    pub fn get_y_line(&self) -> PlotPoints {
        PlotPoints::from_iter(self.y_series.iter().copied())
    }

    pub fn get_z_line(&self) -> PlotPoints {
        PlotPoints::from_iter(self.z_series.iter().copied())
    }

    pub fn clear(&mut self) {
        self.x_series.clear();
        self.y_series.clear();
        self.z_series.clear();
        self.time = 0.0;
    }
}

pub struct LinossWebVisualizerApp {
    oscillator: SimpleLinOSS,
    plot_data: PlotData,
    history: VecDeque<Vector3<f32>>,
    last_update: Instant,
    is_running: bool,
    show_controls: bool,
    show_3d: bool,
    performance_fps: f32,
    frame_count: u64,
    last_fps_update: Instant,
}

impl Default for LinossWebVisualizerApp {
    fn default() -> Self {
        Self {
            oscillator: SimpleLinOSS::new(LinOSSParams::default()),
            plot_data: PlotData::new(),
            history: VecDeque::with_capacity(MAX_PLOT_POINTS),
            last_update: Instant::now(),
            is_running: true,
            show_controls: true,
            show_3d: true,
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

    fn update_simulation(&mut self) {
        if !self.is_running {
            return;
        }

        let now = Instant::now();
        let dt = now.duration_since(self.last_update).as_secs_f32();
        
        if dt > 0.016 { // ~60 FPS
            // Step the oscillator
            for _ in 0..10 { // Multiple steps for smoother visualization
                self.oscillator.step();
            }
            
            // Update plot data
            self.plot_data.add_point(self.oscillator.state);
            self.plot_data.update_time(dt as f64);
            
            // Update history for 3D visualization
            self.history.push_back(self.oscillator.state);
            if self.history.len() > MAX_PLOT_POINTS {
                self.history.pop_front();
            }
            
            self.last_update = now;
            self.frame_count += 1;
            
            // Update FPS
            if now.duration_since(self.last_fps_update).as_secs() >= 1 {
                self.performance_fps = self.frame_count as f32;
                self.frame_count = 0;
                self.last_fps_update = now;
            }
        }
    }

    fn show_controls(&mut self, ui: &mut Ui) {
        ui.collapsing("🎛️ Controls", |ui| {
            ui.horizontal(|ui| {
                if ui.button(if self.is_running { "⏸ Pause" } else { "▶ Play" }).clicked() {
                    self.is_running = !self.is_running;
                }
                if ui.button("🔄 Reset").clicked() {
                    self.oscillator.reset();
                    self.plot_data.clear();
                    self.history.clear();
                }
            });
            
            ui.separator();
            
            ui.label("🎚️ Oscillator Parameters:");
            ui.add(egui::Slider::new(&mut self.oscillator.params.frequency, 1.0..=50.0).text("Frequency"));
            ui.add(egui::Slider::new(&mut self.oscillator.params.damping, 0.0..=1.0).text("Damping"));
            ui.add(egui::Slider::new(&mut self.oscillator.params.coupling, -2.0..=2.0).text("Coupling"));
            ui.add(egui::Slider::new(&mut self.oscillator.params.nonlinearity, 0.0..=2.0).text("Nonlinearity"));
            ui.add(egui::Slider::new(&mut self.oscillator.params.noise_level, 0.0..=0.1).text("Noise Level"));
            
            ui.separator();
            
            ui.label("📊 Visualization:");
            ui.checkbox(&mut self.show_3d, "Show 3D Phase Space");
            
            ui.separator();
            
            ui.label(format!("⚡ Performance: {:.1} FPS", self.performance_fps));
            ui.label(format!("🕐 Time: {:.2}s", self.oscillator.time));
            ui.label(format!("📍 State: [{:.3}, {:.3}, {:.3}]", 
                self.oscillator.state.x, self.oscillator.state.y, self.oscillator.state.z));
        });
    }

    fn show_time_series(&mut self, ui: &mut Ui) {
        ui.collapsing("📈 Time Series", |ui| {
            Plot::new("time_series")
                .height(200.0)
                .show(ui, |plot_ui| {
                    plot_ui.line(
                        Line::new(self.plot_data.get_x_line())
                            .color(Color32::RED)
                            .name("X")
                    );
                    plot_ui.line(
                        Line::new(self.plot_data.get_y_line())
                            .color(Color32::GREEN)
                            .name("Y")
                    );
                    plot_ui.line(
                        Line::new(self.plot_data.get_z_line())
                            .color(Color32::BLUE)
                            .name("Z")
                    );
                });
        });
    }

    fn show_phase_space(&mut self, ui: &mut Ui) {
        if !self.show_3d {
            return;
        }

        ui.collapsing("🌌 3D Phase Space", |ui| {
            // XY plane
            Plot::new("phase_xy")
                .height(200.0)
                .width(200.0)
                .show(ui, |plot_ui| {
                    let xy_points: PlotPoints = self.history.iter()
                        .map(|v| [v.x as f64, v.y as f64])
                        .collect();
                    plot_ui.line(
                        Line::new(xy_points)
                            .color(Color32::YELLOW)
                            .name("XY Phase")
                    );
                });
            
            ui.horizontal(|ui| {
                // XZ plane
                Plot::new("phase_xz")
                    .height(150.0)
                    .width(150.0)
                    .show(ui, |plot_ui| {
                        let xz_points: PlotPoints = self.history.iter()
                            .map(|v| [v.x as f64, v.z as f64])
                            .collect();
                        plot_ui.line(
                            Line::new(xz_points)
                                .color(Color32::from_rgb(0, 255, 255))  // Cyan
                                .name("XZ Phase")
                        );
                    });
                
                // YZ plane
                Plot::new("phase_yz")
                    .height(150.0)
                    .width(150.0)
                    .show(ui, |plot_ui| {
                        let yz_points: PlotPoints = self.history.iter()
                            .map(|v| [v.y as f64, v.z as f64])
                            .collect();
                        plot_ui.line(
                            Line::new(yz_points)
                                .color(Color32::from_rgb(255, 0, 255))  // Magenta
                                .name("YZ Phase")
                        );
                    });
            });
        });
    }
}

impl App for LinossWebVisualizerApp {
    fn update(&mut self, ctx: &Context, _frame: &mut Frame) {
        // Update simulation
        self.update_simulation();
        
        // Request continuous repainting for smooth animation
        ctx.request_repaint();
        
        // Main UI
        egui::TopBottomPanel::top("header").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new("🧠 LinOSS Neural Oscillator Web Demo").size(20.0));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("?").clicked() {
                        // TODO: Show help dialog
                    }
                });
            });
        });
        
        egui::SidePanel::left("controls").show(ctx, |ui| {
            ui.set_width(300.0);
            self.show_controls(ui);
        });
        
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical(|ui| {
                self.show_time_series(ui);
                self.show_phase_space(ui);
            });
        });
    }
}

fn main() -> Result<(), eframe::Error> {
    env_logger::init();
    
    #[cfg(not(target_arch = "wasm32"))]
    {
        let options = eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_inner_size([1200.0, 800.0])
                .with_title("LinOSS Neural Oscillator Demo"),
            ..Default::default()
        };
        
        eframe::run_native(
            "LinOSS Web Demo",
            options,
            Box::new(|_cc| {
                Box::new(LinossWebVisualizerApp::new())
            }),
        )
    }
    
    #[cfg(target_arch = "wasm32")]
    {
        // For WASM, we need to get the canvas element from the DOM
        use wasm_bindgen::JsCast;
        use web_sys::HtmlCanvasElement;
        
        let web_options = eframe::WebOptions::default();
        wasm_bindgen_futures::spawn_local(async {
            let window = web_sys::window().expect("no global `window` exists");
            let document = window.document().expect("should have a document on window");
            let canvas = document
                .get_element_by_id("linoss_canvas")
                .expect("should have a canvas element with id 'linoss_canvas'")
                .dyn_into::<HtmlCanvasElement>()
                .expect("canvas should be an HtmlCanvasElement");
                
            eframe::WebRunner::new()
                .start(
                    canvas,
                    web_options,
                    Box::new(|_cc| Ok(Box::new(LinossWebVisualizerApp::new()))),
                )
                .await
                .expect("failed to start eframe");
        });
        Ok(())
    }
}
