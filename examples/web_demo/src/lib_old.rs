#![cfg_attr(not(test), no_std)]

extern crate alloc;
use alloc::{string::String, vec::Vec, boxed::Box};

use wasm_bindgen::prelude::*;
use console_error_panic_hook;

// Use Burn with NdArray backend for WASM compatibility
use burn::{
    backend::ndarray::{NdArray, NdArrayDevice},
    prelude::*,
};

// Import the real D-LinOSS layer using Burn backend
use linoss_rust::dlinoss::dlinoss_layer::DLinOSSLayer;

// Type alias for our WASM-compatible backend
type Backend = NdArray<f32>;

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
    wasm_logger::init(wasm_logger::Config::default());
}

#[wasm_bindgen]
pub struct DLinOSSDemo {
    layer: DLinOSSLayer<Backend>,
    device: NdArrayDevice,
}

#[wasm_bindgen]
impl DLinOSSDemo {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        let device = NdArrayDevice::Cpu;
        
        // Create a simple D-LinOSS layer for demonstration
        let layer = DLinOSSLayer::new(
            10,  // input_size
            5,   // output_size
            &device,
        );
        
        Self { layer, device }
    }
    
    #[wasm_bindgen]
    pub fn forward(&self, input_data: &[f32]) -> Vec<f32> {
        // Convert input to Burn tensor
        let input_tensor = Tensor::<Backend, 2>::from_data(
            TensorData::new(input_data.to_vec(), [1, input_data.len()]),
            &self.device,
        );
        
        // Run forward pass through D-LinOSS layer
        let output = self.layer.forward(input_tensor);
        
        // Convert back to Vec<f32> for JavaScript
        output.to_data().to_vec().unwrap()
    }
    
    #[wasm_bindgen]
    pub fn get_info(&self) -> String {
        String::from("D-LinOSS Neural Dynamics Layer running in WebAssembly with Burn NdArray backend")
    }
}
    
    // Visualization data
    signal_history: Vec<Vec<f32>>,
    max_history: usize,
    
    // Animation
    time: f64,
    running: bool,
}

impl DLinossWebApp {
    fn new() -> Self {
        let input_dim = 4;
        let hidden_dim = 8;
        let output_dim = 4;
        
        // Create D-LinOSS configuration
        let config = DLinossConfig::new(input_dim, hidden_dim, output_dim)
            .with_dt(0.1)
            .with_a_parameterization(AParameterization::ReLU)
            .with_layer_norm(true);
        
        // Initialize the REAL D-LinOSS layer
        let device = Default::default();
        let dlinoss_layer = config.init(&device);
        
        // Create some demo input data
        let input_data = Tensor::random(
            [1, input_dim], 
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device
        );
        
        Self {
            dlinoss_layer,
            input_data,
            output_data: None,
            input_dim,
            hidden_dim,
            output_dim,
            dt: 0.1,
            parameterization: AParameterization::ReLU,
            use_layer_norm: true,
            signal_history: Vec::new(),
            max_history: 1000,
            time: 0.0,
            running: false,
        }
    }
    
    fn run_forward_pass(&mut self) {
        // Run the REAL D-LinOSS forward pass
        let output = self.dlinoss_layer.forward(self.input_data.clone());
        
        // Store output for visualization
        let output_data = output.to_data().convert::<f32>();
        if let Data::Float(values) = output_data {
            // Add to history for plotting
            self.signal_history.push(values.value);
            if self.signal_history.len() > self.max_history {
                self.signal_history.remove(0);
            }
        }
        
        self.output_data = Some(output);
    }
    
    fn regenerate_layer(&mut self) {
        let config = DLinossConfig::new(self.input_dim, self.hidden_dim, self.output_dim)
            .with_dt(self.dt)
            .with_a_parameterization(self.parameterization.clone())
            .with_layer_norm(self.use_layer_norm);
        
        let device = Default::default();
        self.dlinoss_layer = config.init(&device);
        self.signal_history.clear();
        self.output_data = None;
    }
    
    fn generate_new_input(&mut self) {
        let device = Default::default();
        self.input_data = Tensor::random(
            [1, self.input_dim], 
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device
        );
    }
}

impl eframe::App for DLinossWebApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Continuous animation
        if self.running {
            self.time += 0.016; // ~60 FPS
            self.run_forward_pass();
            ctx.request_repaint();
        }
        
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("🔥 Real D-LinOSS Neural Dynamics - Web Demo");
            ui.separator();
            
            // Status and controls
            ui.horizontal(|ui| {
                if ui.button(if self.running { "⏸️ Pause" } else { "▶️ Start" }).clicked() {
                    self.running = !self.running;
                }
                
                if ui.button("🔄 Reset Layer").clicked() {
                    self.regenerate_layer();
                }
                
                if ui.button("🎲 New Input").clicked() {
                    self.generate_new_input();
                }
                
                if ui.button("→ Single Step").clicked() {
                    self.run_forward_pass();
                }
            });
            
            ui.separator();
            
            // D-LinOSS Configuration
            ui.collapsing("🧮 D-LinOSS Configuration", |ui| {
                ui.horizontal(|ui| {
                    let mut changed = false;
                    
                    changed |= ui.add(egui::Slider::new(&mut self.input_dim, 2..=16).text("Input Dim")).changed();
                    changed |= ui.add(egui::Slider::new(&mut self.hidden_dim, 4..=32).text("Hidden Dim")).changed();
                    changed |= ui.add(egui::Slider::new(&mut self.output_dim, 2..=16).text("Output Dim")).changed();
                    
                    if changed {
                        self.regenerate_layer();
                        self.generate_new_input();
                    }
                });
                
                ui.horizontal(|ui| {
                    let mut changed = false;
                    
                    changed |= ui.add(egui::Slider::new(&mut self.dt, 0.01..=1.0).text("dt (time step)")).changed();
                    
                    // A Parameterization selection
                    ui.label("A Parameterization:");
                    changed |= ui.radio_value(&mut self.parameterization, AParameterization::ReLU, "ReLU").changed();
                    changed |= ui.radio_value(&mut self.parameterization, AParameterization::GELU, "GELU").changed();
                    changed |= ui.radio_value(&mut self.parameterization, AParameterization::Squared, "Squared").changed();
                    
                    changed |= ui.checkbox(&mut self.use_layer_norm, "Layer Norm").changed();
                    
                    if changed {
                        self.regenerate_layer();
                    }
                });
            });
            
            ui.separator();
            
            // Current tensors display
            ui.horizontal(|ui| {
                ui.group(|ui| {
                    ui.vertical(|ui| {
                        ui.label("📥 Input Tensor");
                        let input_data = self.input_data.to_data().convert::<f32>();
                        if let Data::Float(values) = input_data {
                            for (i, &val) in values.value.iter().enumerate() {
                                ui.label(format!("[{}]: {:.4}", i, val));
                            }
                        }
                    });
                });
                
                if let Some(ref output) = self.output_data {
                    ui.group(|ui| {
                        ui.vertical(|ui| {
                            ui.label("📤 Output Tensor");
                            let output_data = output.to_data().convert::<f32>();
                            if let Data::Float(values) = output_data {
                                for (i, &val) in values.value.iter().enumerate() {
                                    ui.label(format!("[{}]: {:.4}", i, val));
                                }
                            }
                        });
                    });
                }
            });
            
            ui.separator();
            
            // Signal visualization
            if !self.signal_history.is_empty() {
                ui.group(|ui| {
                    ui.vertical(|ui| {
                        ui.label("📈 D-LinOSS Output Signal Evolution");
                        
                        egui_plot::Plot::new("dlinoss_signals")
                            .height(300.0)
                            .show(ui, |plot_ui| {
                                // Plot each output dimension
                                for dim in 0..self.output_dim {
                                    let points: Vec<[f64; 2]> = self.signal_history
                                        .iter()
                                        .enumerate()
                                        .filter_map(|(t, values)| {
                                            if dim < values.len() {
                                                Some([t as f64, values[dim] as f64])
                                            } else {
                                                None
                                            }
                                        })
                                        .collect();
                                    
                                    if !points.is_empty() {
                                        let color = match dim {
                                            0 => egui::Color32::RED,
                                            1 => egui::Color32::GREEN, 
                                            2 => egui::Color32::BLUE,
                                            3 => egui::Color32::YELLOW,
                                            _ => egui::Color32::WHITE,
                                        };
                                        
                                        plot_ui.line(
                                            egui_plot::Line::new(egui_plot::PlotPoints::from(points))
                                                .color(color)
                                                .width(2.0)
                                                .name(format!("Output[{}]", dim))
                                        );
                                    }
                                }
                            });
                    });
                });
            }
            
            ui.separator();
            ui.label("🎯 This uses the REAL D-LinOSS layer from linoss_rust crate!");
            ui.label("✨ All UI handled by egui - minimal HTML needed");
        });
    }
}
