use eframe::egui;
use burn_ndarray::{NdArray, NdArrayDevice};
use burn_tensor::{Tensor, Distribution};
use linoss_rust::linoss::{DLinossLayer, DLinossLayerConfig};
use nalgebra::Vector2;
use std::f64::consts::PI;
use wasm_bindgen::prelude::*;

// Type alias for our WASM-compatible backend
type WasmBackend = NdArray<f32>;

#[wasm_bindgen]
pub fn main_burn() {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    
    let web_options = eframe::WebOptions::default();
    
    wasm_bindgen_futures::spawn_local(async {
        let document = web_sys::window().unwrap().document().unwrap();
        let canvas = document
            .get_element_by_id("linoss_canvas")
            .unwrap()
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .unwrap();
            
        let start_result = eframe::WebRunner::new()
            .start(
                canvas,
                web_options,
                Box::new(|_cc| {
                    Ok(Box::new(BurnLinossWebApp::new()))
                }),
            )
            .await;

        if let Err(e) = start_result {
            eframe::web_sys::console::error_1(&format!("Failed to start eframe: {e:?}").into());
        }
    });
}

/// Burn-powered D-LinOSS neural dynamics web demo
struct BurnLinossWebApp {
    time: f64,
    dt: f64,
    dlinoss_layer: DLinossLayer<WasmBackend>,
    device: NdArrayDevice,
    
    // Neural state visualization
    neural_state: Vec<f32>,
    history: Vec<Vec<f32>>,
    max_history: usize,
    
    // Controls
    paused: bool,
    damping_factor: f64,
    frequency: f64,
    amplitude: f64,
    
    // Performance monitoring
    frame_count: u64,
    last_fps_time: f64,
    fps: f64,
}

impl BurnLinossWebApp {
    fn new() -> Self {
        let device = NdArrayDevice::Cpu;
        
        // Configure D-LinOSS layer for neural oscillations
        let config = DLinossLayerConfig {
            d_input: 2,          // 2D input (x, y coordinates or phase space)
            d_model: 8,          // Hidden oscillatory dimensions (4 pairs)
            d_output: 2,         // 2D output for visualization
            delta_t: 0.01,       // 10ms time step for smooth dynamics
            init_std: 0.1,       // Small initialization for stability
            enable_layer_norm: true,
            enable_damping: true, // Key D-LinOSS feature
            a_parameterization: linoss_rust::linoss::AParameterization::ReLU,
        };
        
        let dlinoss_layer = DLinossLayer::new(&config, &device);
        
        Self {
            time: 0.0,
            dt: 0.01,
            dlinoss_layer,
            device,
            neural_state: vec![0.0; 2],
            history: Vec::new(),
            max_history: 1000,
            paused: false,
            damping_factor: 0.1,
            frequency: 2.0,
            amplitude: 1.0,
            frame_count: 0,
            last_fps_time: 0.0,
            fps: 0.0,
        }
    }
    
    fn update_neural_dynamics(&mut self, ctx: &egui::Context) {
        if self.paused {
            return;
        }
        
        // Create input tensor: simple 2D oscillatory input
        let input_data = vec![
            (self.time * self.frequency * 2.0 * PI).sin() as f32 * self.amplitude as f32,
            (self.time * self.frequency * 2.0 * PI).cos() as f32 * self.amplitude as f32,
        ];
        
        // Convert to Burn tensor [batch_size=1, seq_len=1, d_input=2]
        let input_tensor = Tensor::<WasmBackend, 3>::from_floats(
            [[[input_data[0], input_data[1]]]],
            &self.device
        );
        
        // Forward pass through D-LinOSS
        let output = self.dlinoss_layer.forward(input_tensor);
        
        // Extract output data
        let output_data = output.to_data().as_slice::<f32>().unwrap().to_vec();
        
        // Update neural state
        self.neural_state = output_data;
        
        // Add to history
        self.history.push(self.neural_state.clone());
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }
        
        // Update time
        self.time += self.dt;
        
        // Request repaint for animation
        ctx.request_repaint();
    }
    
    fn update_fps(&mut self) {
        self.frame_count += 1;
        let current_time = self.time;
        
        if current_time - self.last_fps_time >= 1.0 {
            self.fps = self.frame_count as f64 / (current_time - self.last_fps_time);
            self.frame_count = 0;
            self.last_fps_time = current_time;
        }
    }
}

impl eframe::App for BurnLinossWebApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.update_fps();
        
        // Main controls panel
        egui::SidePanel::left("controls").show(ctx, |ui| {
            ui.heading("🔥 Burn D-LinOSS Neural Dynamics");
            ui.separator();
            
            // Playback controls
            ui.horizontal(|ui| {
                if ui.button(if self.paused { "▶ Play" } else { "⏸ Pause" }).clicked() {
                    self.paused = !self.paused;
                }
                if ui.button("🔄 Reset").clicked() {
                    self.time = 0.0;
                    self.history.clear();
                    self.neural_state = vec![0.0; 2];
                }
            });
            
            ui.separator();
            
            // Neural dynamics parameters
            ui.label("Neural Parameters:");
            ui.add(egui::Slider::new(&mut self.frequency, 0.1..=10.0).text("Frequency"));
            ui.add(egui::Slider::new(&mut self.amplitude, 0.1..=3.0).text("Amplitude"));
            ui.add(egui::Slider::new(&mut self.damping_factor, 0.0..=1.0).text("Damping"));
            ui.add(egui::Slider::new(&mut self.dt, 0.001..=0.1).text("Time Step"));
            
            ui.separator();
            
            // Performance info
            ui.label(format!("Time: {:.2}s", self.time));
            ui.label(format!("FPS: {:.1}", self.fps));
            ui.label(format!("Neural State: [{:.3}, {:.3}]", 
                self.neural_state.get(0).unwrap_or(&0.0),
                self.neural_state.get(1).unwrap_or(&0.0)
            ));
            
            ui.separator();
            
            // Backend info
            ui.label("Backend: NdArray (WASM-compatible)");
            ui.label(format!("History: {}/{}", self.history.len(), self.max_history));
            
            if ui.button("📋 Copy State").clicked() {
                let state_json = format!(
                    "{{\"time\": {}, \"state\": [{}, {}], \"freq\": {}}}",
                    self.time,
                    self.neural_state.get(0).unwrap_or(&0.0),
                    self.neural_state.get(1).unwrap_or(&0.0),
                    self.frequency
                );
                
                // Try to copy to clipboard (may not work in all WASM contexts)
                if let Some(window) = web_sys::window() {
                    if let Ok(navigator) = window.navigator().clipboard() {
                        let _ = navigator.write_text(&state_json);
                    }
                }
            }
        });
        
        // Main visualization area
        egui::CentralPanel::default().show(ctx, |ui| {
            // Phase space plot
            egui::Frame::canvas(ui.style()).show(ui, |ui| {
                let desired_size = ui.available_width() * egui::Vec2::splat(0.8);
                let (_id, rect) = ui.allocate_space(desired_size);
                
                let to_screen = eframe::emath::RectTransform::from_to(
                    eframe::emath::Rect::from_center_size(
                        eframe::emath::Pos2::ZERO,
                        eframe::emath::Vec2::splat(4.0)
                    ),
                    rect
                );
                
                let painter = ui.painter();
                
                // Draw coordinate axes
                let x_axis_start = to_screen.transform_pos(eframe::emath::pos2(-2.0, 0.0));
                let x_axis_end = to_screen.transform_pos(eframe::emath::pos2(2.0, 0.0));
                let y_axis_start = to_screen.transform_pos(eframe::emath::pos2(0.0, -2.0));
                let y_axis_end = to_screen.transform_pos(eframe::emath::pos2(0.0, 2.0));
                
                painter.line_segment(
                    [x_axis_start, x_axis_end],
                    egui::Stroke::new(1.0, egui::Color32::GRAY)
                );
                painter.line_segment(
                    [y_axis_start, y_axis_end],
                    egui::Stroke::new(1.0, egui::Color32::GRAY)
                );
                
                // Draw neural trajectory
                if self.history.len() > 1 {
                    let points: Vec<_> = self.history.iter()
                        .enumerate()
                        .map(|(i, state)| {
                            let alpha = (i as f32 / self.history.len() as f32).powf(0.5);
                            let pos = to_screen.transform_pos(eframe::emath::pos2(
                                state.get(0).cloned().unwrap_or(0.0),
                                state.get(1).cloned().unwrap_or(0.0)
                            ));
                            (pos, alpha)
                        })
                        .collect();
                    
                    // Draw trajectory with fading trail
                    for i in 1..points.len() {
                        let (start_pos, start_alpha) = points[i-1];
                        let (end_pos, end_alpha) = points[i];
                        let alpha = (start_alpha + end_alpha) / 2.0;
                        
                        painter.line_segment(
                            [start_pos, end_pos],
                            egui::Stroke::new(
                                2.0,
                                egui::Color32::from_rgba_unmultiplied(
                                    255, 100, 100, (alpha * 255.0) as u8
                                )
                            )
                        );
                    }
                    
                    // Draw current position
                    if let Some((current_pos, _)) = points.last() {
                        painter.circle_filled(
                            *current_pos,
                            6.0,
                            egui::Color32::from_rgb(255, 0, 0)
                        );
                    }
                }
                
                // Title
                painter.text(
                    rect.center_top() + egui::vec2(0.0, 10.0),
                    egui::Align2::CENTER_TOP,
                    "D-LinOSS Neural Phase Space",
                    egui::FontId::proportional(16.0),
                    egui::Color32::WHITE
                );
            });
        });
        
        // Update neural dynamics
        self.update_neural_dynamics(ctx);
    }
}
