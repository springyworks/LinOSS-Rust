use eframe::{egui, NativeOptions};
use egui_plot::{Line, Plot, PlotPoints};
use burn::{
    tensor::{backend::Backend, Tensor},
    backend::{ndarray::NdArrayDevice, NdArray},
};
use std::collections::VecDeque;

use linoss_rust::linoss::production_dlinoss::{
    ProductionDLinossConfig, ProductionSSMLayer,
};

type MyBackend = NdArray;

#[derive(Debug)]
struct DLinossResponseApp {
    // D-LinOSS layer
    dlinoss_layer: ProductionSSMLayer<MyBackend>,
    device: <MyBackend as Backend>::Device,
    
    // Signal generation
    time: f32,
    dt: f32,
    input_frequency: f32,
    input_amplitude: f32,
    
    // Response history for plotting
    input_history: VecDeque<[f64; 2]>,  // (time, value)
    output_history: VecDeque<[f64; 2]>, // (time, value)
    max_history: usize,
    
    // UI state
    paused: bool,
    input_type: InputType,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum InputType {
    Sine,
    Square,
    Impulse,
    Step,
    Noise,
}

impl Default for InputType {
    fn default() -> Self {
        InputType::Sine
    }
}

impl std::fmt::Display for InputType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InputType::Sine => write!(f, "Sine Wave"),
            InputType::Square => write!(f, "Square Wave"),
            InputType::Impulse => write!(f, "Impulse"),
            InputType::Step => write!(f, "Step Input"),
            InputType::Noise => write!(f, "Random Noise"),
        }
    }
}

impl DLinossResponseApp {
    fn new() -> Self {
        let device = NdArrayDevice::default();
        
        // Create D-LinOSS configuration with moderate damping
        let config = ProductionDLinossConfig {
            d_model: 1,
            d_inner: 16,
            learnable_diagonal: true,
            learnable_off_diagonal: true,
            learnable_b: true,
            learnable_c: true,
            learnable_damping: true,
            activation: "tanh".to_string(),
            discretization: "rk4".to_string(),
            damping_init_range: (0.1, 0.3),
            eigenvalue_init_std: 1.0,
            bias: true,
            dropout: 0.0,
        };
        
        let dlinoss_layer = ProductionSSMLayer::new(&config, &device);
        
        Self {
            dlinoss_layer,
            device,
            time: 0.0,
            dt: 0.02, // 50 Hz update rate
            input_frequency: 1.0,
            input_amplitude: 1.0,
            input_history: VecDeque::new(),
            output_history: VecDeque::new(),
            max_history: 500,
            paused: false,
            input_type: InputType::Sine,
        }
    }
    
    fn generate_input(&self, time: f32) -> f32 {
        match self.input_type {
            InputType::Sine => {
                self.input_amplitude * (2.0 * std::f32::consts::PI * self.input_frequency * time).sin()
            },
            InputType::Square => {
                let phase = (2.0 * std::f32::consts::PI * self.input_frequency * time).sin();
                self.input_amplitude * if phase > 0.0 { 1.0 } else { -1.0 }
            },
            InputType::Impulse => {
                let pulse_period = 1.0 / self.input_frequency;
                let phase = time % pulse_period;
                if phase < 0.05 { self.input_amplitude } else { 0.0 }
            },
            InputType::Step => {
                let step_time = 2.0; // Step at t=2s
                if time > step_time { self.input_amplitude } else { 0.0 }
            },
            InputType::Noise => {
                self.input_amplitude * (((time * 1000.0) as u32 % 2147483647) as f32 / 2147483647.0 - 0.5) * 2.0
            },
        }
    }
    
    fn update_simulation(&mut self) {
        if self.paused {
            return;
        }
        
        // Generate input signal
        let input_value = self.generate_input(self.time);
        
        // Create input tensor [batch=1, seq_len=1, features=1]
        let input_tensor = Tensor::<MyBackend, 3>::from_floats(
            [[[input_value]]],
            &self.device
        );
        
        // Run through D-LinOSS layer
        let output_tensor = self.dlinoss_layer.forward(input_tensor, "rk4");
        
        // Extract output value
        let output_data: Vec<f32> = output_tensor
            .flatten::<1>(0, 2)
            .to_data()
            .to_vec()
            .unwrap();
        let output_value = output_data[0];
        
        // Update history
        self.input_history.push_back([self.time as f64, input_value as f64]);
        self.output_history.push_back([self.time as f64, output_value as f64]);
        
        // Limit history size
        if self.input_history.len() > self.max_history {
            self.input_history.pop_front();
        }
        if self.output_history.len() > self.max_history {
            self.output_history.pop_front();
        }
        
        // Advance time
        self.time += self.dt;
    }
    
    fn clear_history(&mut self) {
        self.input_history.clear();
        self.output_history.clear();
        self.time = 0.0;
    }
}

impl eframe::App for DLinossResponseApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Update simulation
        self.update_simulation();
        
        // Request continuous repaint for real-time updates
        ctx.request_repaint();
        
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("🌊 D-LinOSS Response Visualizer");
            ui.separator();
            
            // Controls
            ui.horizontal(|ui| {
                if ui.button(if self.paused { "▶ Resume" } else { "⏸ Pause" }).clicked() {
                    self.paused = !self.paused;
                }
                
                if ui.button("🔄 Reset").clicked() {
                    self.clear_history();
                }
                
                ui.separator();
                
                ui.label("Input Type:");
                egui::ComboBox::from_label("input_type")
                    .selected_text(format!("{}", self.input_type))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.input_type, InputType::Sine, "Sine Wave");
                        ui.selectable_value(&mut self.input_type, InputType::Square, "Square Wave");
                        ui.selectable_value(&mut self.input_type, InputType::Impulse, "Impulse");
                        ui.selectable_value(&mut self.input_type, InputType::Step, "Step Input");
                        ui.selectable_value(&mut self.input_type, InputType::Noise, "Random Noise");
                    });
            });
            
            // Parameter controls
            ui.horizontal(|ui| {
                ui.label("Input Frequency:");
                ui.add(egui::Slider::new(&mut self.input_frequency, 0.1..=5.0).suffix(" Hz"));
                
                ui.separator();
                
                ui.label("Input Amplitude:");
                ui.add(egui::Slider::new(&mut self.input_amplitude, 0.1..=3.0));
                
                ui.separator();
                
                ui.label("Update Rate:");
                ui.add(egui::Slider::new(&mut self.dt, 0.005..=0.1).suffix(" s"));
            });
            
            ui.separator();
            
            // Statistics
            if !self.output_history.is_empty() {
                let latest_input = self.input_history.back().map(|x| x[1]).unwrap_or(0.0);
                let latest_output = self.output_history.back().map(|x| x[1]).unwrap_or(0.0);
                
                let output_variance = if self.output_history.len() > 10 {
                    let recent_outputs: Vec<f64> = self.output_history
                        .iter()
                        .rev()
                        .take(50)
                        .map(|x| x[1])
                        .collect();
                    let mean = recent_outputs.iter().sum::<f64>() / recent_outputs.len() as f64;
                    let variance = recent_outputs.iter()
                        .map(|x| (x - mean).powi(2))
                        .sum::<f64>() / recent_outputs.len() as f64;
                    variance.sqrt()
                } else {
                    0.0
                };
                
                ui.horizontal(|ui| {
                    ui.label(format!("📈 Latest Input: {:.3}", latest_input));
                    ui.separator();
                    ui.label(format!("📊 Latest Output: {:.3}", latest_output));
                    ui.separator();
                    ui.label(format!("📉 Output Std Dev: {:.3}", output_variance));
                    ui.separator();
                    ui.label(format!("⏱ Time: {:.1}s", self.time));
                });
            }
            
            ui.separator();
            
            // Main plot
            let plot_height = ui.available_height() * 0.8;
            
            Plot::new("dlinoss_response")
                .height(plot_height)
                .legend(egui_plot::Legend::default().position(egui_plot::Corner::LeftTop))
                .show(ui, |plot_ui| {
                    // Input signal
                    if !self.input_history.is_empty() {
                        let input_points = PlotPoints::from(
                            self.input_history.iter().cloned().collect::<Vec<_>>()
                        );
                        plot_ui.line(
                            Line::new(input_points)
                                .color(egui::Color32::BLUE)
                                .name("Input Signal")
                                .width(2.0)
                        );
                    }
                    
                    // Output signal
                    if !self.output_history.is_empty() {
                        let output_points = PlotPoints::from(
                            self.output_history.iter().cloned().collect::<Vec<_>>()
                        );
                        plot_ui.line(
                            Line::new(output_points)
                                .color(egui::Color32::RED)
                                .name("D-LinOSS Output")
                                .width(2.5)
                        );
                    }
                });
            
            // Help text
            ui.separator();
            ui.small("🎯 This demo shows how D-LinOSS responds to different input signals with learnable damping.");
            ui.small("🔬 Notice how the output exhibits controlled oscillatory behavior and energy dissipation.");
        });
    }
}

fn main() -> Result<(), eframe::Error> {
    env_logger::init();
    
    let options = NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 800.0])
            .with_title("D-LinOSS Response Visualizer"),
        ..Default::default()
    };
    
    eframe::run_native(
        "D-LinOSS Response Visualizer",
        options,
        Box::new(|_cc| {
            Ok(Box::new(DLinossResponseApp::new()))
        }),
    )
}
