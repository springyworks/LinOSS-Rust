use eframe::egui;
use wasm_bindgen::prelude::*;
use std::f64::consts::PI;

// D-LinOSS activation functions for parameter stability
#[derive(Debug, Clone, Copy)]
enum AParameterization {
    /// A = ReLU(A_hat) - allows complete dimension switch-off
    ReLU,
    /// A = GELU(A_hat) - smooth activation with some negative values
    GELU, 
    /// A = A_hat ⊙ A_hat - element-wise square ensures non-negativity
    Squared,
    /// Direct parameterization (user ensures non-negativity)
    Direct,
}

impl Default for AParameterization {
    fn default() -> Self {
        Self::ReLU
    }
}

// Activation function implementations
fn relu(x: f64) -> f64 {
    x.max(0.0)
}

fn gelu(x: f64) -> f64 {
    // GELU(x) = x * Φ(x) where Φ is the standard Gaussian CDF
    // Approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    let sqrt_2_over_pi = (2.0 / PI).sqrt();
    0.5 * x * (1.0 + (sqrt_2_over_pi * (x + 0.044715 * x.powi(3))).tanh())
}

fn apply_parameterization(x: f64, param_type: AParameterization) -> f64 {
    match param_type {
        AParameterization::ReLU => relu(x),
        AParameterization::GELU => gelu(x),
        AParameterization::Squared => x * x,
        AParameterization::Direct => x,
    }
}

// WASM-compatible simplified neural dynamics
// Based on the D-LinOSS principles but without Burn tensors for WASM compatibility
#[wasm_bindgen]
pub fn main_simplified_burn() {
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
                    Ok(Box::new(SimplifiedBurnLinossApp::new()))
                }),
            )
            .await;

        if let Err(e) = start_result {
            eframe::web_sys::console::error_1(&format!("Failed to start eframe: {e:?}").into());
        }
    });
}

/// Simplified D-LinOSS Neural Dynamics for WASM
/// Based on the mathematical principles from our main D-LinOSS implementation
struct SimplifiedBurnLinossApp {
    time: f64,
    dt: f64,
    
    // Neural oscillator states (position, velocity pairs)
    oscillators: Vec<OscillatorState>,
    num_oscillators: usize,
    
    // D-LinOSS parameters
    damping_coeffs: Vec<f64>,
    frequencies: Vec<f64>,
    coupling_strength: f64,
    
    // Visualization
    signal_history: Vec<[f64; 2]>,
    max_history: usize,
    
    // Controls
    paused: bool,
    base_frequency: f64,
    damping_factor: f64,
    
    // D-LinOSS parameterization
    a_parameterization: AParameterization,
    raw_damping_params: Vec<f64>,  // Raw parameters before activation
    raw_frequency_params: Vec<f64>, // Raw parameters before activation
    
    // Learning components
    learner: SimpleLearner,
    learning_enabled: bool,
    prediction_buffer: Vec<f64>,
    target_display: Vec<[f64; 2]>,
    
    // Performance
    fps: f64,
    frame_count: u64,
    last_fps_time: f64,
}

#[derive(Clone, Debug)]
struct OscillatorState {
    position: [f64; 2],  // x, y position
    velocity: [f64; 2],  // x, y velocity
    phase: f64,          // oscillator phase
    energy: f64,         // current energy level
}

impl OscillatorState {
    fn new(index: usize, num_total: usize) -> Self {
        let angle = 2.0 * PI * index as f64 / num_total as f64;
        Self {
            position: [angle.cos(), angle.sin()],
            velocity: [0.0, 0.0],
            phase: angle,
            energy: 1.0,
        }
    }
    
    fn update_dlinoss(&mut self, dt: f64, frequency: f64, damping: f64, input: [f64; 2]) {
        // D-LinOSS dynamics: damped harmonic oscillator with coupling
        let omega = frequency * 2.0 * PI;
        let gamma = damping;
        
        // Calculate forces
        let spring_force_x = -omega * omega * self.position[0];
        let spring_force_y = -omega * omega * self.position[1];
        
        let damping_force_x = -2.0 * gamma * omega * self.velocity[0];
        let damping_force_y = -2.0 * gamma * omega * self.velocity[1];
        
        // External input (coupling)
        let input_force_x = input[0] * 0.1;
        let input_force_y = input[1] * 0.1;
        
        // Update velocity (acceleration = force)
        self.velocity[0] += (spring_force_x + damping_force_x + input_force_x) * dt;
        self.velocity[1] += (spring_force_y + damping_force_y + input_force_y) * dt;
        
        // Update position
        self.position[0] += self.velocity[0] * dt;
        self.position[1] += self.velocity[1] * dt;
        
        // Update phase and energy
        self.phase += omega * dt;
        self.energy = (self.velocity[0] * self.velocity[0] + self.velocity[1] * self.velocity[1] +
                      omega * omega * (self.position[0] * self.position[0] + self.position[1] * self.position[1])) * 0.5;
    }
}

impl SimplifiedBurnLinossApp {
    fn new() -> Self {
        let num_oscillators = 8;
        let mut oscillators = Vec::new();
        let mut damping_coeffs = Vec::new();
        let mut frequencies = Vec::new();
        let mut raw_damping_params = Vec::new();
        let mut raw_frequency_params = Vec::new();
        
        for i in 0..num_oscillators {
            oscillators.push(OscillatorState::new(i, num_oscillators));
            
            // Raw parameters (before activation)
            let raw_damping = -0.5 + (i as f64 * 0.1); // Some negative values
            let raw_frequency = 0.5 + (i as f64 * 0.2);
            
            raw_damping_params.push(raw_damping);
            raw_frequency_params.push(raw_frequency);
            
            // Apply parameterization for stability
            damping_coeffs.push(apply_parameterization(raw_damping, AParameterization::ReLU));
            frequencies.push(apply_parameterization(raw_frequency, AParameterization::ReLU));
        }
        
        Self {
            time: 0.0,
            dt: 0.016, // ~60 FPS
            oscillators,
            num_oscillators,
            damping_coeffs,
            frequencies,
            coupling_strength: 0.1,
            signal_history: Vec::new(),
            max_history: 1000,
            paused: false,
            base_frequency: 1.0,
            damping_factor: 0.1,
            
            // D-LinOSS parameterization
            a_parameterization: AParameterization::ReLU,
            raw_damping_params,
            raw_frequency_params,
            
            // Learning components
            learner: SimpleLearner::new(0.01), // Learning rate
            learning_enabled: false,
            prediction_buffer: Vec::new(),
            target_display: Vec::new(),
            
            // Performance
            fps: 0.0,
            frame_count: 0,
            last_fps_time: 0.0,
        }
    }
    
    fn update_neural_dynamics(&mut self) {
        if self.paused {
            return;
        }
        
        // Create coupling input (simplified network coupling)
        let mut coupling_inputs = vec![[0.0; 2]; self.num_oscillators];
        
        // Calculate mean field coupling
        let mean_x: f64 = self.oscillators.iter().map(|o| o.position[0]).sum::<f64>() / self.num_oscillators as f64;
        let mean_y: f64 = self.oscillators.iter().map(|o| o.position[1]).sum::<f64>() / self.num_oscillators as f64;
        
        for i in 0..self.num_oscillators {
            coupling_inputs[i][0] = (mean_x - self.oscillators[i].position[0]) * self.coupling_strength;
            coupling_inputs[i][1] = (mean_y - self.oscillators[i].position[1]) * self.coupling_strength;
        }
        
        // Update each oscillator with D-LinOSS dynamics
        for i in 0..self.num_oscillators {
            let frequency = self.frequencies[i] * self.base_frequency;
            let damping = self.damping_coeffs[i] * self.damping_factor;
            self.oscillators[i].update_dlinoss(self.dt, frequency, damping, coupling_inputs[i]);
        }
        
        // Record signal for plotting (use first oscillator)
        if !self.oscillators.is_empty() {
            let signal_point = [self.time, self.oscillators[0].position[0]];
            self.signal_history.push(signal_point);
            if self.signal_history.len() > self.max_history {
                self.signal_history.remove(0);
            }
            
            // Learning update
            if self.learning_enabled {
                self.prediction_buffer.push(self.oscillators[0].position[0]);
                
                // Update parameters every 100 steps
                if self.prediction_buffer.len() >= 100 {
                    let learning_happened = self.learner.update_parameters(
                        &mut self.raw_frequency_params,
                        &self.prediction_buffer,
                        self.a_parameterization
                    );
                    
                    if learning_happened {
                        // Recompute applied parameters
                        for i in 0..self.num_oscillators {
                            self.frequencies[i] = apply_parameterization(
                                self.raw_frequency_params[i], 
                                self.a_parameterization
                            );
                        }
                    }
                    
                    self.prediction_buffer.clear();
                }
            }
        }
        
        // Update target display for visualization
        if self.target_display.len() != self.learner.target_signal.len() {
            self.target_display = self.learner.target_signal.iter()
                .enumerate()
                .map(|(i, &val)| [i as f64 * 0.1, val])
                .collect();
        }
        
        self.time += self.dt;
        self.update_fps();
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

impl eframe::App for SimplifiedBurnLinossApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.update_neural_dynamics();
        
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("🔥 Simplified D-LinOSS Neural Dynamics (WASM Compatible)");
            
            ui.horizontal(|ui| {
                ui.label("🧠 Status:");
                if self.paused {
                    ui.colored_label(egui::Color32::YELLOW, "⏸️ PAUSED");
                } else {
                    ui.colored_label(egui::Color32::GREEN, "▶️ RUNNING");
                }
                ui.separator();
                ui.label(format!("🕐 Time: {:.2}s", self.time));
                ui.separator();
                ui.label(format!("📊 FPS: {:.1}", self.fps));
                ui.separator();
                ui.label(format!("🎛️ Oscillators: {}", self.num_oscillators));
            });
            
            ui.separator();
            
            // D-LinOSS Parameterization Controls
            ui.collapsing("🧮 D-LinOSS Parameterization", |ui| {
                ui.horizontal(|ui| {
                    ui.label("A Parameter Method:");
                    let mut changed = false;
                    changed |= ui.radio_value(&mut self.a_parameterization, AParameterization::ReLU, "ReLU").changed();
                    changed |= ui.radio_value(&mut self.a_parameterization, AParameterization::GELU, "GELU").changed();
                    changed |= ui.radio_value(&mut self.a_parameterization, AParameterization::Squared, "Squared").changed();
                    changed |= ui.radio_value(&mut self.a_parameterization, AParameterization::Direct, "Direct").changed();
                    
                    if changed {
                        // Recompute parameters with new activation
                        for i in 0..self.num_oscillators {
                            self.damping_coeffs[i] = apply_parameterization(
                                self.raw_damping_params[i], 
                                self.a_parameterization
                            );
                            self.frequencies[i] = apply_parameterization(
                                self.raw_frequency_params[i], 
                                self.a_parameterization
                            );
                        }
                    }
                });
                
                ui.label("📈 Shows how different activations affect parameter stability");
                match self.a_parameterization {
                    AParameterization::ReLU => ui.label("ReLU: Clips negative values to 0 (allows dimension switch-off)"),
                    AParameterization::GELU => ui.label("GELU: Smooth activation (preserves some negative values)"),
                    AParameterization::Squared => ui.label("Squared: Ensures all values are non-negative"),
                    AParameterization::Direct => ui.label("Direct: Raw parameters (may include negative values)"),
                };
            });
            
            ui.separator();
            
            // Regular Controls
            ui.horizontal(|ui| {
                if ui.button(if self.paused { "▶️ Resume" } else { "⏸️ Pause" }).clicked() {
                    self.paused = !self.paused;
                }
                
                if ui.button("🔄 Reset").clicked() {
                    *self = Self::new();
                }
                
                ui.separator();
                ui.add(egui::Slider::new(&mut self.base_frequency, 0.1..=3.0).text("🎵 Base Frequency"));
                ui.add(egui::Slider::new(&mut self.damping_factor, 0.0..=0.5).text("🌊 Damping Factor"));
                ui.add(egui::Slider::new(&mut self.coupling_strength, 0.0..=0.5).text("🔗 Coupling Strength"));
            });
            
            ui.separator();
            
            // Split view: oscillator states and signal plot
            ui.horizontal(|ui| {
                // Oscillator visualization (phase space)
                ui.group(|ui| {
                    ui.vertical(|ui| {
                        ui.label("🎯 Oscillator Phase Space");
                        
                        egui_plot::Plot::new("phase_space")
                            .view_aspect(1.0)
                            .height(300.0)
                            .show(ui, |plot_ui| {
                                for (i, osc) in self.oscillators.iter().enumerate() {
                
                // Activation Function Visualization
                ui.group(|ui| {
                    ui.vertical(|ui| {
                        ui.label("📊 Activation Function Comparison");
                        
                        egui_plot::Plot::new("activation_plot")
                            .height(200.0)
                            .show(ui, |plot_ui| {
                                let x_range = -2.0..=2.0;
                                let points_count = 100;
                                let step = (x_range.end() - x_range.start()) / points_count as f64;
                                
                                // Generate points for each activation function
                                let relu_points: Vec<[f64; 2]> = (0..=points_count)
                                    .map(|i| {
                                        let x = x_range.start() + i as f64 * step;
                                        [x, relu(x)]
                                    })
                                    .collect();
                                    
                                let gelu_points: Vec<[f64; 2]> = (0..=points_count)
                                    .map(|i| {
                                        let x = x_range.start() + i as f64 * step;
                                        [x, gelu(x)]
                                    })
                                    .collect();
                                    
                                let squared_points: Vec<[f64; 2]> = (0..=points_count)
                                    .map(|i| {
                                        let x = x_range.start() + i as f64 * step;
                                        [x, x * x]
                                    })
                                    .collect();
                                    
                                let direct_points: Vec<[f64; 2]> = (0..=points_count)
                                    .map(|i| {
                                        let x = x_range.start() + i as f64 * step;
                                        [x, x]
                                    })
                                    .collect();
                                
                                plot_ui.line(
                                    egui_plot::Line::new(egui_plot::PlotPoints::from(relu_points))
                                        .color(egui::Color32::RED)
                                        .width(2.0)
                                        .name("ReLU")
                                );
                                
                                plot_ui.line(
                                    egui_plot::Line::new(egui_plot::PlotPoints::from(gelu_points))
                                        .color(egui::Color32::GREEN)
                                        .width(2.0)
                                        .name("GELU")
                                );
                                
                                plot_ui.line(
                                    egui_plot::Line::new(egui_plot::PlotPoints::from(squared_points))
                                        .color(egui::Color32::BLUE)
                                        .width(2.0)
                                        .name("Squared")
                                );
                                
                                plot_ui.line(
                                    egui_plot::Line::new(egui_plot::PlotPoints::from(direct_points))
                                        .color(egui::Color32::GRAY)
                                        .width(2.0)
                                        .name("Direct")
                                );
                            });
                    });
                });
                
                ui.separator();
                
                // Signal time series
                ui.group(|ui| {
                    ui.vertical(|ui| {
                        ui.label("📈 Neural Signal (First Oscillator)");
                        
                        egui_plot::Plot::new("signal_plot")
                            .height(300.0)
                            .show(ui, |plot_ui| {
                                if !self.signal_history.is_empty() {
                                    plot_ui.line(
                                        egui_plot::Line::new(egui_plot::PlotPoints::from(self.signal_history.clone()))
                                            .color(egui::Color32::LIGHT_BLUE)
                                            .width(2.0)
                                            .name("Position X")
                                    );
                                }
                            });
                    });
                });
            });
            
            ui.separator();
            
            // Oscillator status table
            ui.group(|ui| {
                ui.vertical(|ui| {
                    ui.label("🔢 Oscillator Status");
                    
                    egui::ScrollArea::vertical().max_height(150.0).show(ui, |ui| {
                        egui::Grid::new("osc_grid").striped(true).show(ui, |ui| {
                            ui.label("ID");
                            ui.label("Position X");
                            ui.label("Position Y");
                            ui.label("Velocity X");
                            ui.label("Velocity Y");
                            ui.label("Energy");
                            ui.label("Frequency");
                            ui.label("Damping");
                            ui.end_row();
                            
                            for (i, osc) in self.oscillators.iter().enumerate() {
                                ui.label(format!("{}", i));
                                ui.label(format!("{:.3}", osc.position[0]));
                                ui.label(format!("{:.3}", osc.position[1]));
                                ui.label(format!("{:.3}", osc.velocity[0]));
                                ui.label(format!("{:.3}", osc.velocity[1]));
                                ui.label(format!("{:.3}", osc.energy));
                                ui.label(format!("{:.2}", self.frequencies[i] * self.base_frequency));
                                ui.label(format!("{:.3}", self.damping_coeffs[i] * self.damping_factor));
                                ui.end_row();
                            }
                        });
                    });
                });
            });
            
            ui.separator();
            
            // Information
            ui.horizontal(|ui| {
                ui.label("💡 Info:");
                ui.colored_label(egui::Color32::LIGHT_BLUE, "WASM-compatible D-LinOSS simulation");
                ui.separator();
                ui.label("🔬 Based on Damped Linear Oscillatory State-Space Models");
                ui.separator();
                ui.label("🌐 Running in WebAssembly");
            });
        });
        
        // Request repaint for animation
        ctx.request_repaint();
    }
}

// Simple gradient-based learning for demonstration
struct SimpleLearner {
    learning_rate: f64,
    target_signal: Vec<f64>,
    current_loss: f64,
    loss_history: Vec<f64>,
}

impl SimpleLearner {
    fn new(learning_rate: f64) -> Self {
        // Generate a target sinusoidal signal for learning
        let target_signal: Vec<f64> = (0..100)
            .map(|i| (2.0 * PI * i as f64 / 100.0 * 3.0).sin()) // 3 Hz sine wave
            .collect();
            
        Self {
            learning_rate,
            target_signal,
            current_loss: 0.0,
            loss_history: Vec::new(),
        }
    }
    
    fn compute_loss(&self, predicted: &[f64]) -> f64 {
        if predicted.len() != self.target_signal.len() {
            return f64::INFINITY;
        }
        
        // Mean squared error
        predicted.iter()
            .zip(self.target_signal.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>() / predicted.len() as f64
    }
    
    fn update_parameters(&mut self, 
                        raw_params: &mut [f64], 
                        predicted: &[f64],
                        param_type: AParameterization) -> bool {
        if predicted.len() != self.target_signal.len() {
            return false;
        }
        
        let current_loss = self.compute_loss(predicted);
        self.current_loss = current_loss;
        self.loss_history.push(current_loss);
        
        if self.loss_history.len() > 1000 {
            self.loss_history.remove(0);
        }
        
        // Simple finite difference gradient approximation
        const EPSILON: f64 = 1e-6;
        let mut gradients = vec![0.0; raw_params.len()];
        
        for i in 0..raw_params.len() {
            // Compute loss with parameter + epsilon
            raw_params[i] += EPSILON;
            let applied_param = apply_parameterization(raw_params[i], param_type);
            
            // For simplicity, assume each parameter affects output linearly
            // In practice, you'd need to re-run the simulation
            let perturbed_loss = current_loss + EPSILON * applied_param; // Simplified
            
            raw_params[i] -= EPSILON;
            
            // Gradient approximation
            gradients[i] = (perturbed_loss - current_loss) / EPSILON;
        }
        
        // Parameter update with gradient descent
        let mut updated = false;
        for i in 0..raw_params.len() {
            let old_param = raw_params[i];
            raw_params[i] -= self.learning_rate * gradients[i];
            
            // Clip parameters to reasonable range
            raw_params[i] = raw_params[i].clamp(-2.0, 2.0);
            
            if (old_param - raw_params[i]).abs() > 1e-8 {
                updated = true;
            }
        }
        
        updated
    }
}
