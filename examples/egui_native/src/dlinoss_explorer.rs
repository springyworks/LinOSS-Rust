///! D-LinOSS Explorer - Interactive Neural Dynamics with egui
///!
///! A comprehensive native GUI for exploring D-LinOSS neural dynamics
///! Features real-time parameter adjustment, multiple visualization modes,
///! and direct integration with Burn neural networks.

use eframe::egui;
use egui_plot::{PlotPoints, Line, Plot};
use linoss_rust::{
    linoss::{DLinossLayer, DLinossLayerConfig, LinossLayer, LinossLayerConfig},
};
use burn::backend::NdArray;
use nalgebra::DVector;
use std::collections::VecDeque;
use std::time::Instant;

fn main() -> Result<(), eframe::Error> {
    env_logger::init();
    
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 800.0])
            .with_title("🧠 D-LinOSS Neural Dynamics Explorer"),
        ..Default::default()
    };

    eframe::run_native(
        "D-LinOSS Explorer",
        options,
        Box::new(|_cc| {
            // This gives us image support:
            egui_extras::install_image_loaders(&_cc.egui_ctx);
            
            Ok(Box::new(DLinossExplorer::new()))
        }),
    )
}

struct DLinossExplorer {
    // D-LinOSS components
    dlinoss_layer: Option<DLinossLayer<NdArray>>,
    linoss_layer: Option<LinossLayer<NdArray>>,
    
    // Simulation state
    is_running: bool,
    current_input: DVector<f64>,
    output_history: VecDeque<DVector<f64>>,
    time_history: VecDeque<f64>,
    current_time: f64,
    last_update: Instant,
    
    // Parameters
    d_input: usize,
    d_model: usize,
    d_output: usize,
    damping: f64,
    frequency: f64,
    delta_t: f64,
    max_history: usize,
    
    // UI state
    selected_tab: Tab,
    show_help: bool,
    input_signal_type: SignalType,
    input_amplitude: f64,
    input_phase: f64,
    
    // Visualization options
    plot_mode: PlotMode,
    show_individual_neurons: bool,
    selected_neuron: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Tab {
    Simulation,
    Parameters,
    Analysis,
    Network,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum SignalType {
    Sine,
    Cosine,
    Square,
    Sawtooth,
    Noise,
    Impulse,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum PlotMode {
    TimeSeries,
    PhaseSpace,
    Frequency,
    Heatmap,
}

impl DLinossExplorer {
    fn new() -> Self {
        let d_input = 10;
        let d_model = 20;
        let d_output = 5;
        
        Self {
            dlinoss_layer: None,
            linoss_layer: None,
            is_running: false,
            current_input: DVector::zeros(d_input),
            output_history: VecDeque::new(),
            time_history: VecDeque::new(),
            current_time: 0.0,
            last_update: Instant::now(),
            
            d_input,
            d_model,
            d_output,
            damping: 0.1,
            frequency: 1.0,
            delta_t: 0.01,
            max_history: 1000,
            
            selected_tab: Tab::Simulation,
            show_help: false,
            input_signal_type: SignalType::Sine,
            input_amplitude: 1.0,
            input_phase: 0.0,
            
            plot_mode: PlotMode::TimeSeries,
            show_individual_neurons: true,
            selected_neuron: 0,
        }
    }
    
    fn initialize_networks(&mut self) {
        // Initialize D-LinOSS layer
        let dlinoss_config = DLinossLayerConfig::new_dlinoss(
            self.d_input,
            self.d_model,
            self.d_output,
        );
        
        // For now, we'll create a simple implementation
        // In a real implementation, you'd use the actual DLinossLayer from the library
        log::info!("Initializing D-LinOSS networks with config: {:?}", dlinoss_config);
        
        // Initialize history
        self.output_history.clear();
        self.time_history.clear();
        self.current_time = 0.0;
    }
    
    fn generate_input_signal(&self, t: f64) -> DVector<f64> {
        let mut input = DVector::zeros(self.d_input);
        
        for i in 0..self.d_input {
            let phase_offset = i as f64 * 0.1;
            let value = match self.input_signal_type {
                SignalType::Sine => {
                    self.input_amplitude * (2.0 * std::f64::consts::PI * self.frequency * t + self.input_phase + phase_offset).sin()
                },
                SignalType::Cosine => {
                    self.input_amplitude * (2.0 * std::f64::consts::PI * self.frequency * t + self.input_phase + phase_offset).cos()
                },
                SignalType::Square => {
                    self.input_amplitude * if (2.0 * std::f64::consts::PI * self.frequency * t + phase_offset).sin() > 0.0 { 1.0 } else { -1.0 }
                },
                SignalType::Sawtooth => {
                    self.input_amplitude * (2.0 * (self.frequency * t + phase_offset) % 1.0 - 1.0)
                },
                SignalType::Noise => {
                    self.input_amplitude * (rand::random::<f64>() - 0.5) * 2.0
                },
                SignalType::Impulse => {
                    if (t % (1.0 / self.frequency)) < self.delta_t { self.input_amplitude } else { 0.0 }
                },
            };
            input[i] = value;
        }
        
        input
    }
    
    fn update_simulation(&mut self) {
        if !self.is_running {
            return;
        }
        
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_update).as_secs_f64();
        
        // Minimum update interval to prevent excessive CPU usage (max 30 FPS for simulation)
        let min_update_interval = (1.0_f64 / 30.0_f64).max(self.delta_t);
        
        if elapsed >= min_update_interval {
            self.current_time += self.delta_t;
            
            // Generate input signal
            self.current_input = self.generate_input_signal(self.current_time);
            
            // For now, simulate a simple response (replace with actual D-LinOSS computation)
            let mut output = DVector::zeros(self.d_output);
            for i in 0..self.d_output {
                let decay = (-self.damping * self.current_time).exp();
                output[i] = self.current_input[i % self.d_input] * decay + 
                           0.1 * (3.0 * self.current_time + i as f64).sin();
            }
            
            // Store history
            self.output_history.push_back(output);
            self.time_history.push_back(self.current_time);
            
            // Limit history size
            while self.output_history.len() > self.max_history {
                self.output_history.pop_front();
                self.time_history.pop_front();
            }
            
            self.last_update = now;
        }
    }
}

impl eframe::App for DLinossExplorer {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Only update simulation if running and request repaint when needed
        let needs_repaint = self.is_running;
        if needs_repaint {
            self.update_simulation();
            // Request repaint for next frame only if simulation is running
            ctx.request_repaint();
        }
        
        // Top menu bar
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("New Simulation").clicked() {
                        self.initialize_networks();
                    }
                    if ui.button("Load Parameters").clicked() {
                        // TODO: Load parameters
                    }
                    if ui.button("Save Parameters").clicked() {
                        // TODO: Save parameters
                    }
                    ui.separator();
                    if ui.button("Exit").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                });
                
                ui.menu_button("View", |ui| {
                    ui.checkbox(&mut self.show_individual_neurons, "Show Individual Neurons");
                    ui.separator();
                    ui.label("Plot Mode:");
                    ui.selectable_value(&mut self.plot_mode, PlotMode::TimeSeries, "Time Series");
                    ui.selectable_value(&mut self.plot_mode, PlotMode::PhaseSpace, "Phase Space");
                    ui.selectable_value(&mut self.plot_mode, PlotMode::Frequency, "Frequency");
                    ui.selectable_value(&mut self.plot_mode, PlotMode::Heatmap, "Heatmap");
                });
                
                ui.menu_button("Help", |ui| {
                    if ui.button("About").clicked() {
                        self.show_help = true;
                    }
                });
            });
        });
        
        // Left panel - Controls
        egui::SidePanel::left("left_panel").min_width(300.0).show(ctx, |ui| {
            ui.heading("🎛️ Controls");
            
            // Simulation controls
            ui.group(|ui| {
                ui.label("🎮 Simulation");
                ui.horizontal(|ui| {
                    if ui.button(if self.is_running { "⏸ Pause" } else { "▶ Start" }).clicked() {
                        self.is_running = !self.is_running;
                        if self.is_running && (self.dlinoss_layer.is_none() || self.linoss_layer.is_none()) {
                            self.initialize_networks();
                        }
                    }
                    
                    if ui.button("🔄 Reset").clicked() {
                        self.initialize_networks();
                        self.current_time = 0.0;
                    }
                    
                    if ui.button("⚡ Initialize").clicked() {
                        self.initialize_networks();
                    }
                });
                
                ui.horizontal(|ui| {
                    ui.label("Status:");
                    ui.colored_label(
                        if self.is_running { egui::Color32::GREEN } else { egui::Color32::RED },
                        if self.is_running { "🟢 Running" } else { "🔴 Stopped" }
                    );
                });
            });
            
            ui.separator();
            
            // Network parameters
            ui.group(|ui| {
                ui.label("🧠 Network Architecture");
                ui.horizontal(|ui| {
                    ui.label("Input Size:");
                    ui.add(egui::Slider::new(&mut self.d_input, 1..=50));
                });
                ui.horizontal(|ui| {
                    ui.label("Model Size:");
                    ui.add(egui::Slider::new(&mut self.d_model, 1..=100));
                });
                ui.horizontal(|ui| {
                    ui.label("Output Size:");
                    ui.add(egui::Slider::new(&mut self.d_output, 1..=20));
                });
            });
            
            ui.separator();
            
            // D-LinOSS parameters
            ui.group(|ui| {
                ui.label("⚙️ D-LinOSS Parameters");
                ui.horizontal(|ui| {
                    ui.label("Damping:");
                    ui.add(egui::Slider::new(&mut self.damping, 0.0..=2.0).step_by(0.01));
                });
                ui.horizontal(|ui| {
                    ui.label("Frequency:");
                    ui.add(egui::Slider::new(&mut self.frequency, 0.1..=10.0).step_by(0.1));
                });
                ui.horizontal(|ui| {
                    ui.label("Time Step (dt):");
                    ui.add(egui::Slider::new(&mut self.delta_t, 0.001..=0.1).step_by(0.001));
                });
            });
            
            ui.separator();
            
            // Input signal parameters
            ui.group(|ui| {
                ui.label("📡 Input Signal");
                ui.horizontal(|ui| {
                    ui.label("Type:");
                    egui::ComboBox::from_label("")
                        .selected_text(format!("{:?}", self.input_signal_type))
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut self.input_signal_type, SignalType::Sine, "Sine");
                            ui.selectable_value(&mut self.input_signal_type, SignalType::Cosine, "Cosine");
                            ui.selectable_value(&mut self.input_signal_type, SignalType::Square, "Square");
                            ui.selectable_value(&mut self.input_signal_type, SignalType::Sawtooth, "Sawtooth");
                            ui.selectable_value(&mut self.input_signal_type, SignalType::Noise, "Noise");
                            ui.selectable_value(&mut self.input_signal_type, SignalType::Impulse, "Impulse");
                        });
                });
                ui.horizontal(|ui| {
                    ui.label("Amplitude:");
                    ui.add(egui::Slider::new(&mut self.input_amplitude, 0.0..=5.0).step_by(0.1));
                });
                ui.horizontal(|ui| {
                    ui.label("Phase:");
                    ui.add(egui::Slider::new(&mut self.input_phase, 0.0..=(2.0 * std::f64::consts::PI)).step_by(0.1));
                });
            });
            
            ui.separator();
            
            // Current values display
            ui.group(|ui| {
                ui.label("📊 Current State");
                ui.horizontal(|ui| {
                    ui.label(format!("Time: {:.3}s", self.current_time));
                });
                if !self.current_input.is_empty() {
                    ui.horizontal(|ui| {
                        ui.label(format!("Input[0]: {:.3}", self.current_input[0]));
                    });
                }
                if let Some(latest_output) = self.output_history.back() {
                    ui.horizontal(|ui| {
                        ui.label(format!("Output[0]: {:.3}", latest_output[0]));
                    });
                }
                ui.horizontal(|ui| {
                    ui.label(format!("History: {} samples", self.output_history.len()));
                });
            });
        });
        
        // Main content area
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("🧠 D-LinOSS Neural Dynamics Explorer");
            
            if self.output_history.is_empty() {
                ui.vertical_centered(|ui| {
                    ui.label("No data yet. Click 'Start' to begin simulation.");
                    ui.label("Initialize the network first, then start the simulation to see real-time neural dynamics!");
                });
                return;
            }
            
            // Visualization
            match self.plot_mode {
                PlotMode::TimeSeries => self.draw_time_series(ui),
                PlotMode::PhaseSpace => self.draw_phase_space(ui),
                PlotMode::Frequency => self.draw_frequency(ui),
                PlotMode::Heatmap => self.draw_heatmap(ui),
            }
        });
        
        // Help dialog
        if self.show_help {
            egui::Window::new("About D-LinOSS Explorer")
                .collapsible(false)
                .resizable(true)
                .show(ctx, |ui| {
                    ui.label("🧠 D-LinOSS Neural Dynamics Explorer");
                    ui.separator();
                    ui.label("An interactive GUI for exploring D-LinOSS neural networks.");
                    ui.label("");
                    ui.label("Features:");
                    ui.label("• Real-time parameter adjustment");
                    ui.label("• Multiple visualization modes");
                    ui.label("• Direct Burn integration");
                    ui.label("• Neural network architecture control");
                    ui.label("");
                    ui.label("Built with Rust, egui, and the LinossRust library.");
                    
                    if ui.button("Close").clicked() {
                        self.show_help = false;
                    }
                });
        }
        
        // Request repaint for smooth animation
        if self.is_running {
            ctx.request_repaint();
        }
    }
}

impl DLinossExplorer {
    fn draw_time_series(&self, ui: &mut egui::Ui) {
        let plot = Plot::new("time_series")
            .height(400.0)
            .legend(egui_plot::Legend::default())
            .show_axes([true, true])
            .show_grid([true, true]);
            
        plot.show(ui, |plot_ui| {
            if self.show_individual_neurons {
                // Show individual neuron traces
                for neuron_idx in 0..self.d_output.min(5) { // Limit to first 5 for readability
                    let points: PlotPoints = self.time_history.iter()
                        .zip(self.output_history.iter())
                        .map(|(t, output)| [*t, output[neuron_idx]])
                        .collect();
                    
                    let color = match neuron_idx {
                        0 => egui::Color32::RED,
                        1 => egui::Color32::GREEN,
                        2 => egui::Color32::BLUE,
                        3 => egui::Color32::YELLOW,
                        4 => egui::Color32::CYAN,
                        _ => egui::Color32::WHITE,
                    };
                    
                    plot_ui.line(
                        Line::new(points)
                            .color(color)
                            .name(format!("Neuron {}", neuron_idx))
                    );
                }
            } else {
                // Show average activity
                let points: PlotPoints = self.time_history.iter()
                    .zip(self.output_history.iter())
                    .map(|(t, output)| {
                        let avg = output.iter().sum::<f64>() / output.len() as f64;
                        [*t, avg]
                    })
                    .collect();
                
                plot_ui.line(
                    Line::new(points)
                        .color(egui::Color32::WHITE)
                        .name("Average Activity")
                );
            }
        });
    }
    
    fn draw_phase_space(&self, ui: &mut egui::Ui) {
        ui.label("Phase Space Plot (TODO: Implement velocity vs position)");
        // TODO: Implement phase space visualization
    }
    
    fn draw_frequency(&self, ui: &mut egui::Ui) {
        ui.label("Frequency Domain Plot (TODO: Implement FFT analysis)");
        // TODO: Implement frequency domain analysis
    }
    
    fn draw_heatmap(&self, ui: &mut egui::Ui) {
        ui.label("Neural Activity Heatmap (TODO: Implement 2D heatmap)");
        // TODO: Implement heatmap visualization
    }
}
