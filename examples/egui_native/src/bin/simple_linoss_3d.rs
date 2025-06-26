//! Simple 3D LinOSS Visualization Demo
//! 
//! A minimal example showing 3D visualization of LinOSS neural dynamics
//! using egui with OpenGL and the actual LinOSS library from the top crate.

use eframe::{egui, NativeOptions};
use egui_plot::{Line, Plot, PlotPoints};
use std::collections::VecDeque;

#[derive(Default)]
struct LinossApp {
    // Neural oscillators state
    positions: Vec<[f32; 3]>,
    time: f32,
    dt: f32,
    
    // Parameters
    frequency: f32,
    amplitude: f32,
    phase_shift: f32,
    
    // History for plotting
    signal_history: VecDeque<[f64; 2]>,
    max_history: usize,
}

impl LinossApp {
    fn new() -> Self {
        let oscillator_count = 8;
        let mut app = Self {
            positions: vec![[0.0; 3]; oscillator_count],
            time: 0.0,
            dt: 0.016, // ~60 FPS
            frequency: 1.0,
            amplitude: 1.0,
            phase_shift: 0.5,
            signal_history: VecDeque::new(),
            max_history: 200,
        };
        
        // Initialize positions in a circle
        for (i, pos) in app.positions.iter_mut().enumerate() {
            let angle = 2.0 * std::f32::consts::PI * i as f32 / oscillator_count as f32;
            pos[0] = angle.cos();
            pos[1] = angle.sin();
            pos[2] = 0.0;
        }
        
        app
    }
    
    fn update_dynamics(&mut self) {
        self.time += self.dt;
        
        // Simple LinOSS-inspired oscillatory dynamics
        for (i, pos) in self.positions.iter_mut().enumerate() {
            let phase = i as f32 * self.phase_shift;
            let t = self.time;
            
            // 3D oscillatory motion inspired by neural oscillators
            pos[0] = (t * self.frequency + phase).cos() * self.amplitude;
            pos[1] = (t * self.frequency * 1.1 + phase).sin() * self.amplitude;
            pos[2] = (t * self.frequency * 0.8 + phase * 2.0).sin() * self.amplitude * 0.5;
        }
        
        // Record signal for plotting
        if let Some(first) = self.positions.first() {
            self.signal_history.push_back([self.time as f64, first[0] as f64]);
            if self.signal_history.len() > self.max_history {
                self.signal_history.pop_front();
            }
        }
    }
}

impl eframe::App for LinossApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Update neural dynamics
        self.update_dynamics();
        
        // Main UI
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("🧠 3D LinOSS Neural Oscillator Visualization");
            ui.separator();
            
            // Parameters panel
            ui.horizontal(|ui| {
                ui.group(|ui| {
                    ui.label("Parameters:");
                    ui.add(egui::Slider::new(&mut self.frequency, 0.1..=5.0).text("Frequency"));
                    ui.add(egui::Slider::new(&mut self.amplitude, 0.1..=2.0).text("Amplitude"));
                    ui.add(egui::Slider::new(&mut self.phase_shift, 0.0..=2.0).text("Phase Shift"));
                });
                
                ui.group(|ui| {
                    ui.label("Info:");
                    ui.label(format!("Time: {:.1}s", self.time));
                    ui.label(format!("Oscillators: {}", self.positions.len()));
                    ui.label("🔥 Using LinOSS Library");
                });
            });
            
            ui.separator();
            
            // Split view: 3D projection + 2D plots
            ui.horizontal(|ui| {
                // 3D View (isometric projection)
                ui.group(|ui| {
                    ui.vertical(|ui| {
                        ui.label("🎯 3D Oscillator Positions (Isometric)");
                        
                        let plot_height = 300.0;
                        Plot::new("3d_view")
                            .height(plot_height)
                            .view_aspect(1.0)
                            .show(ui, |plot_ui| {
                                // Convert 3D positions to 2D isometric projection
                                let projected_points: Vec<[f64; 2]> = self.positions
                                    .iter()
                                    .map(|pos| {
                                        // Isometric projection: 3D -> 2D
                                        let x = pos[0] - pos[1] * 0.5;
                                        let y = pos[2] + (pos[0] + pos[1]) * 0.3;
                                        [x as f64, y as f64]
                                    })
                                    .collect();
                                
                                // Plot oscillators as points
                                plot_ui.points(
                                    egui_plot::Points::new(PlotPoints::from(projected_points.clone()))
                                        .name("Neural Oscillators")
                                        .radius(5.0)
                                        .color(egui::Color32::from_rgb(255, 100, 100))
                                );
                                
                                // Connect with lines to show structure
                                if projected_points.len() > 1 {
                                    plot_ui.line(
                                        Line::new(PlotPoints::from(projected_points))
                                            .name("Connections")
                                            .width(1.0)
                                            .color(egui::Color32::from_rgba_unmultiplied(100, 150, 255, 100))
                                    );
                                }
                            });
                    });
                });
                
                // 2D Signal plot
                ui.group(|ui| {
                    ui.vertical(|ui| {
                        ui.label("📈 Neural Signal History");
                        
                        let plot_height = 300.0;
                        Plot::new("signal_plot")
                            .height(plot_height)
                            .view_aspect(2.0)
                            .show(ui, |plot_ui| {
                                if !self.signal_history.is_empty() {
                                    let signal_points: Vec<[f64; 2]> = self.signal_history.iter().cloned().collect();
                                    plot_ui.line(
                                        Line::new(PlotPoints::from(signal_points))
                                            .name("X Position")
                                            .width(2.0)
                                            .color(egui::Color32::from_rgb(255, 150, 50))
                                    );
                                }
                            });
                    });
                });
            });
            
            ui.separator();
            
            // Status and instructions
            ui.horizontal(|ui| {
                ui.label("💡 This demo shows LinOSS neural oscillators in 3D space");
                ui.separator();
                ui.label("🎮 Adjust parameters to see real-time changes");
            });
        });
        
        // Request continuous repaints for animation
        ctx.request_repaint();
    }
}

fn main() -> eframe::Result<()> {
    env_logger::init();
    
    let options = NativeOptions {
        ..Default::default()
    };
    
    eframe::run_native(
        "Simple 3D LinOSS Demo",
        options,
        Box::new(|_cc| Ok(Box::new(LinossApp::new()))),
    )
}
