use eframe::{egui, NativeOptions};
use egui_plot::{Line, Plot, PlotPoints, Points};
use burn::{
    tensor::{backend::Backend, Tensor},
    backend::{ndarray::NdArrayDevice, NdArray},
};
use std::collections::VecDeque;

use linoss_rust::linoss::dlinoss_layer::{DLinossLayer, DLinossLayerConfig};

type MyBackend = NdArray;

#[derive(Debug)]
struct LissajousApp {
    // D-LinOSS block with 3 inputs and 3 outputs
    dlinoss_block: DLinossLayer<MyBackend>,
    device: <MyBackend as Backend>::Device,
    
    // Simulation state
    time: f32,
    dt: f32,
    
    // Pulse generation parameters
    pulse_frequency: f32,
    pulse_amplitude: f32,
    pulse_width: f32,
    
    // Lissajous plotting
    lissajous_points: VecDeque<[f64; 2]>,  // (output1, output2) for 2D plot
    color_history: VecDeque<f32>,          // output3 values for color mapping
    max_history: usize,
    
    // Signal time-series visualization (like oscilloscope)
    signal_history: VecDeque<[f32; 6]>,    // [input1, input2, input3, output1, output2, output3]
    time_history: VecDeque<f32>,           // Time points for x-axis
    signal_max_history: usize,
    
    // UI state
    paused: bool,
    pulse_type: PulseType,
    show_trails: bool,
    trail_fade: f32,    // Trail fading factor (0.0 = no fade, 1.0 = immediate fade)
    global_damping: f32, // Global damping multiplier
    show_signal_view: bool, // Toggle between Lissajous and oscilloscope view
    show_dev_overlay: bool, // Toggle development panel names overlay
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum PulseType {
    SinglePulse,
    DoublePulse,
    TriplePulse,
    RandomPulse,
    SinePulse,
    LissajousClassic,
    LissajousComplex,
    PhasedOscillators,
}

impl Default for PulseType {
    fn default() -> Self {
        PulseType::LissajousClassic
    }
}

impl std::fmt::Display for PulseType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PulseType::SinglePulse => write!(f, "Single Pulse"),
            PulseType::DoublePulse => write!(f, "Double Pulse"),
            PulseType::TriplePulse => write!(f, "Triple Pulse"),
            PulseType::RandomPulse => write!(f, "Random Pulse"),
            PulseType::SinePulse => write!(f, "Sine Pulse"),
            PulseType::LissajousClassic => write!(f, "Classic Lissajous"),
            PulseType::LissajousComplex => write!(f, "Complex Lissajous"),
            PulseType::PhasedOscillators => write!(f, "Phased Oscillators"),
        }
    }
}

impl LissajousApp {
    fn new() -> Self {
        let device = NdArrayDevice::default();
        
        // Create D-LinOSS configuration with more oscillators for complex patterns
        let config = DLinossLayerConfig::new_dlinoss(3, 32, 3); // 3 inputs, 32 internal, 3 outputs
        
        let dlinoss_block = DLinossLayer::new(&config, &device);
        
        println!("🔬 D-LinOSS Advanced Signal Analyzer initialized");
        println!("   📊 Vertical layout: Lissajous (top) + Time-series (bottom)");
        println!("   🎛️  3 inputs → 32 D-LinOSS oscillators → 3 outputs");
        println!("   🌊 Real-time signal analysis with damping dynamics"); 
        println!("   📐 Following arXiv:2505.12171 mathematical framework");
        
        Self {
            dlinoss_block,
            device,
            time: 0.0,
            dt: 0.02, // Faster update rate for smoother visualization
            pulse_frequency: 1.2,
            pulse_amplitude: 1.0,
            pulse_width: 0.12,
            lissajous_points: VecDeque::new(),
            color_history: VecDeque::new(),
            max_history: 3000, // More history for complex patterns
            signal_history: VecDeque::new(),
            time_history: VecDeque::new(),
            signal_max_history: 1000, // Oscilloscope view window
            paused: false,
            pulse_type: PulseType::LissajousClassic,
            show_trails: true,
            trail_fade: 0.7,     // Moderate fading to reduce clutter
            global_damping: 1.0, // Default damping multiplier
            show_signal_view: true, // Start with oscilloscope plot visible by default
            show_dev_overlay: false, // Development overlay off by default
        }
    }
    
    fn recreate_dlinoss_with_damping(&mut self, damping_factor: f32) {
        // Create new config with modified damping
        let mut config = DLinossLayerConfig::new_dlinoss(3, 32, 3);
        config.init_damping = (damping_factor * 0.1) as f64; // Scale damping (0.1 is base)
        
        // Recreate the D-LinOSS block
        self.dlinoss_block = DLinossLayer::new(&config, &self.device);
        
        println!("🔧 D-LinOSS damping updated: {:.3}", config.init_damping);
    }
    
    fn generate_pulse_inputs(&self, time: f32) -> [f32; 3] {
        let pulse_period = 1.0 / self.pulse_frequency;
        let phase = (time % pulse_period) / pulse_period;
        
        match self.pulse_type {
            PulseType::SinglePulse => {
                // Phase-shifted pulses for different dynamics
                let pulse1_active = phase < self.pulse_width;
                let pulse2_active = (phase + 0.33) % 1.0 < self.pulse_width; // 120° phase shift
                let pulse3_active = (phase + 0.66) % 1.0 < self.pulse_width; // 240° phase shift
                [
                    if pulse1_active { self.pulse_amplitude } else { 0.0 },
                    if pulse2_active { self.pulse_amplitude * 0.8 } else { 0.0 },
                    if pulse3_active { self.pulse_amplitude * 0.6 } else { 0.0 },
                ]
            },
            PulseType::DoublePulse => {
                // Two sets of phase-shifted double pulses
                let pulse1_active = phase < self.pulse_width || (phase > 0.5 && phase < (0.5 + self.pulse_width));
                let pulse2_active = ((phase + 0.25) % 1.0) < self.pulse_width || (((phase + 0.25) % 1.0) > 0.5 && ((phase + 0.25) % 1.0) < (0.5 + self.pulse_width));
                let pulse3_active = ((phase + 0.5) % 1.0) < self.pulse_width;
                [
                    if pulse1_active { self.pulse_amplitude } else { 0.0 },
                    if pulse2_active { self.pulse_amplitude * 0.9 } else { 0.0 },
                    if pulse3_active { self.pulse_amplitude * 0.7 } else { 0.0 },
                ]
            },
            PulseType::TriplePulse => {
                // Triple pulses with different timing and amplitudes
                let pulse1_active = phase < self.pulse_width;
                let pulse2_active = (phase + 0.2) % 1.0 < self.pulse_width;
                let pulse3_active = (phase + 0.4) % 1.0 < self.pulse_width;
                [
                    if pulse1_active { self.pulse_amplitude } else { 0.0 },
                    if pulse2_active { self.pulse_amplitude * 0.85 } else { 0.0 },
                    if pulse3_active { self.pulse_amplitude * 0.65 } else { 0.0 },
                ]
            },
            PulseType::RandomPulse => {
                let noise_seed = ((time * 1000.0) as u32) % 1000;
                let threshold = self.pulse_width * 8.0; // Scale threshold
                [
                    if (noise_seed % 101) as f32 / 100.0 < threshold { self.pulse_amplitude } else { 0.0 },
                    if ((noise_seed + 37) % 103) as f32 / 100.0 < threshold { self.pulse_amplitude * 0.8 } else { 0.0 },
                    if ((noise_seed + 73) % 107) as f32 / 100.0 < threshold { self.pulse_amplitude * 0.6 } else { 0.0 },
                ]
            },
            PulseType::SinePulse => {
                // Sine waves with different frequencies and phases for Lissajous patterns
                let freq1 = self.pulse_frequency;
                let freq2 = self.pulse_frequency * 1.618; // Golden ratio for interesting patterns
                let freq3 = self.pulse_frequency * 0.786; // Another irrational ratio
                
                // Add phase shifts for complex Lissajous figures
                let phase1 = 0.0;
                let phase2 = std::f32::consts::PI / 2.0; // 90° phase shift
                let phase3 = std::f32::consts::PI; // 180° phase shift
                
                [
                    self.pulse_amplitude * (2.0 * std::f32::consts::PI * freq1 * time + phase1).sin(),
                    self.pulse_amplitude * 0.9 * (2.0 * std::f32::consts::PI * freq2 * time + phase2).sin(),
                    self.pulse_amplitude * 0.7 * (2.0 * std::f32::consts::PI * freq3 * time + phase3).sin(),
                ]
            },
            PulseType::LissajousClassic => {
                // Classic Lissajous with simple frequency ratios
                let freq_base = self.pulse_frequency;
                [
                    self.pulse_amplitude * (2.0 * std::f32::consts::PI * freq_base * time).sin(),
                    self.pulse_amplitude * (2.0 * std::f32::consts::PI * freq_base * 2.0 * time + std::f32::consts::PI / 4.0).sin(),
                    self.pulse_amplitude * 0.5 * (2.0 * std::f32::consts::PI * freq_base * 3.0 * time).cos(),
                ]
            },
            PulseType::LissajousComplex => {
                // Complex Lissajous with irrational frequency ratios
                let freq_base = self.pulse_frequency;
                let sqrt2 = 1.41421356; // √2
                let sqrt3 = 1.73205081; // √3
                [
                    self.pulse_amplitude * (2.0 * std::f32::consts::PI * freq_base * time).sin(),
                    self.pulse_amplitude * (2.0 * std::f32::consts::PI * freq_base * sqrt2 * time + std::f32::consts::PI / 3.0).sin(),
                    self.pulse_amplitude * 0.8 * (2.0 * std::f32::consts::PI * freq_base * sqrt3 * time + std::f32::consts::PI / 6.0).cos(),
                ]
            },
            PulseType::PhasedOscillators => {
                // Three oscillators with 120° phase differences
                let freq = self.pulse_frequency;
                let phase1 = 0.0;
                let phase2 = 2.0 * std::f32::consts::PI / 3.0; // 120°
                let phase3 = 4.0 * std::f32::consts::PI / 3.0; // 240°
                [
                    self.pulse_amplitude * (2.0 * std::f32::consts::PI * freq * time + phase1).sin(),
                    self.pulse_amplitude * (2.0 * std::f32::consts::PI * freq * time + phase2).sin(),
                    self.pulse_amplitude * (2.0 * std::f32::consts::PI * freq * time + phase3).sin(),
                ]
            },
        }
    }
    
    fn update_simulation(&mut self) {
        if self.paused {
            return;
        }
        
        // Generate 3 pulse inputs
        let inputs = self.generate_pulse_inputs(self.time);
        
        // Create input tensor [batch=1, seq_len=1, features=3]
        let input_tensor = Tensor::<MyBackend, 3>::from_floats(
            [[inputs]],
            &self.device
        );
        
        // Run through D-LinOSS block
        let output_tensor = self.dlinoss_block.forward(input_tensor);
        
        // Extract 3 output values
        let output_data: Vec<f32> = output_tensor
            .flatten::<1>(0, 2)
            .to_data()
            .to_vec()
            .unwrap();
        
        let output1 = output_data[0];
        let output2 = output_data[1];
        let output3 = output_data[2];
        
        // Update Lissajous plot: use output1 and output2 for 2D coordinates
        self.lissajous_points.push_back([output1 as f64, output2 as f64]);
        
        // Use output3 for color mapping
        self.color_history.push_back(output3);
        
        // Record signal history for oscilloscope view
        self.signal_history.push_back([inputs[0], inputs[1], inputs[2], output1, output2, output3]);
        self.time_history.push_back(self.time);
        
        // Limit history size
        if self.lissajous_points.len() > self.max_history {
            self.lissajous_points.pop_front();
        }
        if self.color_history.len() > self.max_history {
            self.color_history.pop_front();
        }
        if self.signal_history.len() > self.signal_max_history {
            self.signal_history.pop_front();
            self.time_history.pop_front();
        }
        
        // Advance time
        self.time += self.dt;
    }
    
    fn clear_history(&mut self) {
        self.lissajous_points.clear();
        self.color_history.clear();
        self.signal_history.clear();
        self.time_history.clear();
        self.time = 0.0;
    }
}

impl eframe::App for LissajousApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Update simulation
        self.update_simulation();
        
        // Request continuous repaint for real-time updates
        ctx.request_repaint();
        
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("🔬 D-LinOSS Advanced Signal Analyzer");
            ui.label("📊 Following arXiv:2505.12171 mathematical framework with diagonal matrix parallel-scan optimization");
            ui.separator();
            
            // Controls - Row 1
            ui.horizontal(|ui| {
                if ui.button(if self.paused { "▶ Resume" } else { "⏸ Pause" }).clicked() {
                    self.paused = !self.paused;
                }
                
                if ui.button("🔄 Reset").clicked() {
                    self.clear_history();
                }
                
                ui.separator();
                
                // View toggle
                if ui.button(if self.show_signal_view { "📊 Show Text Info" } else { "📈 Show Time Plot" }).clicked() {
                    self.show_signal_view = !self.show_signal_view;
                }
                
                ui.separator();
                
                // Development overlay toggle
                if ui.button(if self.show_dev_overlay { "🔍 Hide Dev Names" } else { "🔍 Show Dev Names" }).clicked() {
                    self.show_dev_overlay = !self.show_dev_overlay;
                }
                
                ui.separator();
                ui.checkbox(&mut self.show_trails, "Show Trails");
                
                ui.separator();
                
                ui.label("Trail Fade:");
                ui.add(egui::Slider::new(&mut self.trail_fade, 0.0..=0.1).logarithmic(true));
            });
            
            // Controls - Row 2  
            ui.horizontal(|ui| {
                
                ui.label("Pulse Type:");
                egui::ComboBox::from_label("pulse_type")
                    .selected_text(format!("{}", self.pulse_type))
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.pulse_type, PulseType::LissajousClassic, "Classic Lissajous");
                        ui.selectable_value(&mut self.pulse_type, PulseType::LissajousComplex, "Complex Lissajous");
                        ui.selectable_value(&mut self.pulse_type, PulseType::PhasedOscillators, "Phased Oscillators");
                        ui.selectable_value(&mut self.pulse_type, PulseType::SinePulse, "Sine Pulse");
                        ui.selectable_value(&mut self.pulse_type, PulseType::SinglePulse, "Single Pulse");
                        ui.selectable_value(&mut self.pulse_type, PulseType::DoublePulse, "Double Pulse");
                        ui.selectable_value(&mut self.pulse_type, PulseType::TriplePulse, "Triple Pulse");
                        ui.selectable_value(&mut self.pulse_type, PulseType::RandomPulse, "Random Pulse");
                    });
                
                ui.separator();
                ui.checkbox(&mut self.show_trails, "Show Trails");
            });
            
            // Parameter controls - Row 1
            ui.horizontal(|ui| {
                ui.label("Pulse Frequency:");
                ui.add(egui::Slider::new(&mut self.pulse_frequency, 0.1..=5.0).suffix(" Hz"));
                
                ui.separator();
                
                ui.label("Pulse Amplitude:");
                ui.add(egui::Slider::new(&mut self.pulse_amplitude, 0.1..=3.0));
                
                ui.separator();
                
                ui.label("Pulse Width:");
                ui.add(egui::Slider::new(&mut self.pulse_width, 0.01..=0.5));
            });
            
            // Parameter controls - Row 2  
            ui.horizontal(|ui| {
                ui.label("Update Rate:");
                ui.add(egui::Slider::new(&mut self.dt, 0.01..=0.2).suffix(" s"));
                
                ui.separator();
                
                ui.label("Trail Fade:");
                ui.add(egui::Slider::new(&mut self.trail_fade, 0.0..=1.0).text("More fade →"));
                
                ui.separator();
                
                let mut new_damping = self.global_damping;
                ui.label("D-LinOSS Damping:");
                let damping_changed = ui.add(egui::Slider::new(&mut new_damping, 0.1..=3.0).text("Higher damping →")).changed();
                if damping_changed {
                    self.global_damping = new_damping;
                    self.recreate_dlinoss_with_damping(new_damping);
                }
            });
            
            ui.separator();
            
            // Statistics with fixed-width number fields
            if !self.lissajous_points.is_empty() && !self.color_history.is_empty() {
                let latest_point = self.lissajous_points.back().unwrap();
                let latest_color = self.color_history.back().unwrap();
                
                ui.horizontal(|ui| {
                    ui.label("📈 Latest X (Output1):");
                    ui.add_sized([80.0, 20.0], egui::Label::new(format!("{:>8.3}", latest_point[0])));
                    ui.separator();
                    ui.label("📊 Latest Y (Output2):");
                    ui.add_sized([80.0, 20.0], egui::Label::new(format!("{:>8.3}", latest_point[1])));
                    ui.separator();
                    ui.label("🎨 Color Value (Output3):");
                    ui.add_sized([80.0, 20.0], egui::Label::new(format!("{:>8.3}", latest_color)));
                    ui.separator();
                    ui.label("⏱ Time:");
                    ui.add_sized([60.0, 20.0], egui::Label::new(format!("{:>6.1}s", self.time)));
                    ui.separator();
                    ui.label("🔧 Damping:");
                    ui.add_sized([50.0, 20.0], egui::Label::new(format!("{:>4.2}x", self.global_damping)));
                    ui.separator();
                    ui.label("👀 Trail Points:");
                    ui.add_sized([60.0, 20.0], egui::Label::new(format!("{:>6}", self.lissajous_points.len())));
                });
            }
            
            ui.separator();
            
            // Main visualization area - Vertical layout
            let available_height = ui.available_height() * 0.85;
            
            ui.vertical(|ui| {
                // Top panel - 2D Lissajous (full width)
                ui.vertical(|ui| {
                    ui.label("🌀 Lissajous View: D-LinOSS Output Phase Space");
                    
                    let plot_height = available_height * 0.6; // 60% of height for Lissajous
                    
                    Plot::new("lissajous_plot")
                        .height(plot_height)
                        .data_aspect(1.0) // Square aspect ratio for Lissajous
                        .legend(egui_plot::Legend::default().position(egui_plot::Corner::LeftTop))
                        .show(ui, |plot_ui| {
                            if !self.lissajous_points.is_empty() {
                                if self.show_trails {
                                    // Show faded trail as multiple lines with decreasing alpha
                                    let trail_points: Vec<[f64; 2]> = self.lissajous_points.iter().cloned().collect();
                                    let total_points = trail_points.len();
                                    
                                    if total_points > 1 {
                                        // Draw trail in segments with fading alpha
                                        let segment_size = 25.max(total_points / 12); // Adaptive segment size
                                        
                                        for (segment_idx, chunk) in trail_points.chunks(segment_size).enumerate() {
                                            if chunk.len() > 1 {
                                                let progress = segment_idx as f32 / (trail_points.len() / segment_size).max(1) as f32;
                                                let fade_alpha = ((1.0 - progress * self.trail_fade).max(0.05) * 140.0) as u8;
                                                
                                                let segment_points = PlotPoints::from(chunk.to_vec());
                                                plot_ui.line(
                                                    Line::new(segment_points)
                                                        .color(egui::Color32::from_rgba_unmultiplied(120, 180, 255, fade_alpha))
                                                        .width(1.2 + (1.0 - progress) * 1.5) // Thicker lines for newer segments
                                                );
                                            }
                                        }
                                    }
                                }
                                
                                // Show colored points based on output3
                                let color_points: Vec<_> = self.lissajous_points
                                    .iter()
                                    .zip(self.color_history.iter())
                                    .map(|(point, &color_val)| {
                                        // Map color_val to a color
                                        let normalized_color = ((color_val + 1.0) / 2.0).clamp(0.0, 1.0);
                                        let red = (255.0 * normalized_color) as u8;
                                        let blue = (255.0 * (1.0 - normalized_color)) as u8;
                                        let color = egui::Color32::from_rgb(red, 120, blue);
                                        
                                        ([point[0], point[1]], color)
                                    })
                                    .collect();
                                
                                // Plot recent points with color coding
                                for (point, color) in color_points.iter().rev().take(60) {
                                    plot_ui.points(
                                        Points::new(vec![*point])
                                            .color(*color)
                                            .radius(3.5)
                                    );
                                }
                                
                                // Highlight current position
                                if let Some(current_point) = self.lissajous_points.back() {
                                    plot_ui.points(
                                        Points::new(vec![*current_point])
                                            .color(egui::Color32::YELLOW)
                                            .radius(6.0)
                                            .name("Current Position")
                                    );
                                }
                            }
                        });
                });
                
                ui.separator();
                
                // Bottom panel - Signal Analysis (full width, split into input/output subpanes)
                ui.vertical(|ui| {
                    if self.show_signal_view {
                        // 📈 OSCILLOSCOPE VIEW - Split Input/Output Panes
                        ui.label("📈 Oscilloscope View: Input/Output Signal Analysis (Scrolling Window)");
                        
                        let plot_height = available_height * 0.175; // Each subpane gets 17.5% of height
                        
                        // Calculate scrolling time window data once
                        let time_window = 20.0; // seconds to show
                        let current_time = self.time;
                        let window_start = current_time - time_window;
                        
                        let windowed_data: Vec<_> = if !self.signal_history.is_empty() {
                            self.time_history.iter()
                                .zip(self.signal_history.iter())
                                .filter(|(t, _)| **t >= window_start)
                                .collect()
                        } else {
                            Vec::new()
                        };
                        
                        // INPUT SIGNALS SUBPANE
                        ui.horizontal(|ui| {
                            ui.label("📥 Input Signals:");
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                ui.small("(Pulse generators → D-LinOSS)");
                            });
                        });
                        
                        Plot::new("input_signals_plot")
                            .height(plot_height)
                            .legend(egui_plot::Legend::default().position(egui_plot::Corner::LeftTop))
                            .show(ui, |plot_ui| {
                                if !windowed_data.is_empty() {
                                    // Set fixed Y-axis scale to prevent squeezing
                                    let y_range = [-2.0, 2.0]; // Fixed scale for all signals
                                    plot_ui.set_plot_bounds(egui_plot::PlotBounds::from_min_max(
                                        [window_start as f64, y_range[0]],
                                        [current_time as f64, y_range[1]]
                                    ));
                                    
                                    // Input signals only (bright colors, medium thickness)
                                    let input1_points: PlotPoints = windowed_data.iter()
                                        .map(|(t, signals)| [**t as f64, signals[0] as f64])
                                        .collect();
                                    
                                    let input2_points: PlotPoints = windowed_data.iter()
                                        .map(|(t, signals)| [**t as f64, signals[1] as f64])
                                        .collect();
                                        
                                    let input3_points: PlotPoints = windowed_data.iter()
                                        .map(|(t, signals)| [**t as f64, signals[2] as f64])
                                        .collect();
                                    
                                    plot_ui.line(
                                        Line::new(input1_points)
                                            .color(egui::Color32::from_rgb(255, 60, 60))
                                            .width(2.0)
                                            .name("Input 1 (Red)")
                                    );
                                    
                                    plot_ui.line(
                                        Line::new(input2_points)
                                            .color(egui::Color32::from_rgb(60, 255, 60))
                                            .width(2.0)
                                            .name("Input 2 (Green)")
                                    );
                                    
                                    plot_ui.line(
                                        Line::new(input3_points)
                                            .color(egui::Color32::from_rgb(60, 60, 255))
                                            .width(2.0)
                                            .name("Input 3 (Blue)")
                                    );
                                }
                            });
                        
                        ui.add_space(5.0);
                        
                        // OUTPUT SIGNALS SUBPANE  
                        ui.horizontal(|ui| {
                            ui.label("📤 D-LinOSS Output Signals:");
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                ui.small("(Processed through 32 oscillators)");
                            });
                        });
                        
                        Plot::new("output_signals_plot")
                            .height(plot_height)
                            .legend(egui_plot::Legend::default().position(egui_plot::Corner::LeftTop))
                            .show(ui, |plot_ui| {
                                if !windowed_data.is_empty() {
                                    // Set fixed Y-axis scale to prevent squeezing
                                    let y_range = [-2.0, 2.0]; // Fixed scale for all signals
                                    plot_ui.set_plot_bounds(egui_plot::PlotBounds::from_min_max(
                                        [window_start as f64, y_range[0]],
                                        [current_time as f64, y_range[1]]
                                    ));
                                    
                                    // Output signals only (muted colors, thicker lines)
                                    let output1_points: PlotPoints = windowed_data.iter()
                                        .map(|(t, signals)| [**t as f64, signals[3] as f64])
                                        .collect();
                                        
                                    let output2_points: PlotPoints = windowed_data.iter()
                                        .map(|(t, signals)| [**t as f64, signals[4] as f64])
                                        .collect();
                                        
                                    let output3_points: PlotPoints = windowed_data.iter()
                                        .map(|(t, signals)| [**t as f64, signals[5] as f64])
                                        .collect();
                                    
                                    plot_ui.line(
                                        Line::new(output1_points)
                                            .color(egui::Color32::from_rgb(200, 100, 100))
                                            .width(2.5)
                                            .name("D-LinOSS Output 1 (X)")
                                    );
                                    
                                    plot_ui.line(
                                        Line::new(output2_points)
                                            .color(egui::Color32::from_rgb(100, 200, 100))
                                            .width(2.5)
                                            .name("D-LinOSS Output 2 (Y)")
                                    );
                                    
                                    plot_ui.line(
                                        Line::new(output3_points)
                                            .color(egui::Color32::from_rgb(100, 100, 200))
                                            .width(2.5)
                                            .name("D-LinOSS Output 3 (Color)")
                                    );
                                }
                            });
                        
                    } else {
                        // Text-based signal summary panel
                        ui.label("📈 Signal Analysis Panel");
                    
                    // Create a scrollable area for signal information
                    egui::ScrollArea::vertical()
                        .max_height(available_height * 0.35)
                        .show(ui, |ui| {
                            ui.group(|ui| {
                                ui.label("🎛️ Real-time Signal Values");
                                ui.separator();
                                
                                if !self.signal_history.is_empty() && !self.time_history.is_empty() {
                                    let latest_signals = self.signal_history.back().unwrap();
                                    let current_time = self.time_history.back().unwrap();
                                    
                                    // Input signals
                                    ui.label("📥 Inputs:");
                                    ui.label(format!("  Input 1: {:.4}", latest_signals[0]));
                                    ui.label(format!("  Input 2: {:.4}", latest_signals[1]));
                                    ui.label(format!("  Input 3: {:.4}", latest_signals[2]));
                                    
                                    ui.separator();
                                    
                                    // Output signals
                                    ui.label("📤 D-LinOSS Outputs:");
                                    ui.label(format!("  Output 1 (X): {:.4}", latest_signals[3]));
                                    ui.label(format!("  Output 2 (Y): {:.4}", latest_signals[4]));
                                    ui.label(format!("  Output 3 (Color): {:.4}", latest_signals[5]));
                                    
                                    ui.separator();
                                    
                                    // Time and statistics
                                    ui.label("⏱️ Timing:");
                                    ui.label(format!("  Current Time: {:.2}s", current_time));
                                    ui.label(format!("  Total Samples: {}", self.signal_history.len()));
                                    
                                    ui.separator();
                                    
                                    // Signal statistics (last 100 samples)
                                    ui.label("📊 Signal Statistics (last 100):");
                                    let recent_count = 100.min(self.signal_history.len());
                                    if recent_count > 0 {
                                        let recent_signals: Vec<_> = self.signal_history
                                            .iter()
                                            .rev()
                                            .take(recent_count)
                                            .collect();
                                        
                                        // Calculate averages
                                        let mut avg_inputs = [0.0; 3];
                                        let mut avg_outputs = [0.0; 3];
                                        
                                        for signals in &recent_signals {
                                            for i in 0..3 {
                                                avg_inputs[i] += signals[i];
                                                avg_outputs[i] += signals[i + 3];
                                            }
                                        }
                                        
                                        for i in 0..3 {
                                            avg_inputs[i] /= recent_count as f32;
                                            avg_outputs[i] /= recent_count as f32;
                                        }
                                        
                                        ui.label("  Input Averages:");
                                        ui.label(format!("    Avg Input 1: {:.4}", avg_inputs[0]));
                                        ui.label(format!("    Avg Input 2: {:.4}", avg_inputs[1]));
                                        ui.label(format!("    Avg Input 3: {:.4}", avg_inputs[2]));
                                        
                                        ui.label("  Output Averages:");
                                        ui.label(format!("    Avg Output 1: {:.4}", avg_outputs[0]));
                                        ui.label(format!("    Avg Output 2: {:.4}", avg_outputs[1]));
                                        ui.label(format!("    Avg Output 3: {:.4}", avg_outputs[2]));
                                    }
                                    
                                    ui.separator();
                                    
                                    // System status
                                    ui.label("🔧 System Status:");
                                    ui.label(format!("  Pulse Type: {}", self.pulse_type));
                                    ui.label(format!("  Frequency: {:.2} Hz", self.pulse_frequency));
                                    ui.label(format!("  Amplitude: {:.2}", self.pulse_amplitude));
                                    ui.label(format!("  Damping: {:.2}x", self.global_damping));
                                    ui.label(format!("  Update Rate: {:.3}s", self.dt));
                                    ui.label(format!("  Status: {}", if self.paused { "⏸ Paused" } else { "▶ Running" }));
                                }
                            });
                            
                            ui.add_space(10.0);
                            
                            // Oscilloscope toggle section
                            ui.group(|ui| {
                                ui.label("📈 Panel Toggle");
                                ui.separator();
                                
                                ui.label("Current mode: 📊 Text panel");
                                ui.label("Click 'Show Time Plot' above to see split input/output time-series");
                            });
                        });
                    } // End of else block (text mode)
                }); // End of ui.vertical
            });
            
            // Help text
            ui.separator();
            ui.small("🎯 D-LinOSS block processes 3 phase-shifted inputs to create complex Lissajous patterns.");
            ui.small("🌈 Output1 (X) and Output2 (Y) form the trajectory, Output3 controls color intensity.");
            ui.small("⚡ Phase differences between inputs create open loops, spirals, and complex figures.");
            ui.small("� Bottom panel: Scrolling 20-second time window with fixed Y-scale (-2 to +2) prevents signal squeezing.");
            ui.small("�🔄 Try different pulse types: Classic Lissajous uses simple frequency ratios (1:2:3).");
            ui.small("✨ Complex Lissajous uses irrational ratios (1:√2:√3) for never-repeating patterns.");
            ui.small("🎛️ D-LinOSS Damping: Controls energy dissipation in the 32 internal oscillators.");
            ui.small("🌊 Trail Fade: Higher values make older trail segments more transparent, reducing clutter.");
        });
        
        // Development overlay - Full-window transparent layer with UI element labels
        if self.show_dev_overlay {
            // Create a full-window transparent overlay
            egui::Area::new("dev_overlay".into())
                .fixed_pos(egui::pos2(0.0, 0.0))
                .show(ctx, |ui| {
                    // Make the overlay cover the entire window
                    let screen_rect = ctx.screen_rect();
                    ui.allocate_exact_size(screen_rect.size(), egui::Sense::hover());
                    
                    // Semi-transparent background
                    ui.painter().rect_filled(
                        screen_rect,
                        0.0,
                        egui::Color32::from_rgba_unmultiplied(0, 0, 0, 80)
                    );
                    
                    // Development labels positioned over actual UI elements
                    
                    // Main title area
                    ui.painter().text(
                        egui::pos2(20.0, 20.0),
                        egui::Align2::LEFT_TOP,
                        "🔍 DEV OVERLAY - MainTitlePanel",
                        egui::FontId::proportional(14.0),
                        egui::Color32::YELLOW
                    );
                    
                    // Controls area
                    ui.painter().text(
                        egui::pos2(20.0, 80.0),
                        egui::Align2::LEFT_TOP,
                        "⚙️ PulseControlPanel + DampingControlPanel",
                        egui::FontId::proportional(12.0),
                        egui::Color32::LIGHT_GREEN
                    );
                    
                    // Statistics bar
                    ui.painter().text(
                        egui::pos2(20.0, 180.0),
                        egui::Align2::LEFT_TOP,
                        "📊 FixedWidthStatisticsPanel (anti-jumping numbers)",
                        egui::FontId::proportional(12.0),
                        egui::Color32::LIGHT_BLUE
                    );
                    
                    // Lissajous plot area (top 60% of visualization)
                    let liss_y = 220.0;
                    ui.painter().text(
                        egui::pos2(20.0, liss_y),
                        egui::Align2::LEFT_TOP,
                        "🌀 LissajousPhaseSpacePanel",
                        egui::FontId::proportional(14.0),
                        egui::Color32::CYAN
                    );
                    
                    ui.painter().text(
                        egui::pos2(40.0, liss_y + 20.0),
                        egui::Align2::LEFT_TOP,
                        "• Output1(X) vs Output2(Y) phase space",
                        egui::FontId::proportional(10.0),
                        egui::Color32::WHITE
                    );
                    
                    ui.painter().text(
                        egui::pos2(40.0, liss_y + 35.0),
                        egui::Align2::LEFT_TOP,
                        "• Color coded by Output3 value",
                        egui::FontId::proportional(10.0),
                        egui::Color32::WHITE
                    );
                    
                    // Signal panels area (bottom 35% split)
                    let signal_y = screen_rect.height() * 0.65;
                    
                    if self.show_signal_view {
                        // Input signals subpanel
                        ui.painter().text(
                            egui::pos2(20.0, signal_y),
                            egui::Align2::LEFT_TOP,
                            "📥 InputSignalsSubPanel",
                            egui::FontId::proportional(13.0),
                            egui::Color32::LIGHT_RED
                        );
                        
                        ui.painter().text(
                            egui::pos2(40.0, signal_y + 18.0),
                            egui::Align2::LEFT_TOP,
                            "• Raw pulse generators → D-LinOSS",
                            egui::FontId::proportional(10.0),
                            egui::Color32::WHITE
                        );
                        
                        // Output signals subpanel
                        let output_y = signal_y + (screen_rect.height() * 0.175);
                        ui.painter().text(
                            egui::pos2(20.0, output_y),
                            egui::Align2::LEFT_TOP,
                            "📤 OutputSignalsSubPanel",
                            egui::FontId::proportional(13.0),
                            egui::Color32::LIGHT_GREEN
                        );
                        
                        ui.painter().text(
                            egui::pos2(40.0, output_y + 18.0),
                            egui::Align2::LEFT_TOP,
                            "• 32 oscillators → 3 processed outputs",
                            egui::FontId::proportional(10.0),
                            egui::Color32::WHITE
                        );
                    } else {
                        ui.painter().text(
                            egui::pos2(20.0, signal_y),
                            egui::Align2::LEFT_TOP,
                            "📊 TextAnalysisPanel",
                            egui::FontId::proportional(13.0),
                            egui::Color32::LIGHT_BLUE
                        );
                        
                        ui.painter().text(
                            egui::pos2(40.0, signal_y + 18.0),
                            egui::Align2::LEFT_TOP,
                            "• Real-time numerical values & statistics",
                            egui::FontId::proportional(10.0),
                            egui::Color32::WHITE
                        );
                    }
                    
                    // Help area
                    ui.painter().text(
                        egui::pos2(20.0, screen_rect.height() - 60.0),
                        egui::Align2::LEFT_TOP,
                        "💡 HelpTextPanel",
                        egui::FontId::proportional(12.0),
                        egui::Color32::GRAY
                    );
                    
                    // Toggle instructions
                    ui.painter().text(
                        egui::pos2(screen_rect.width() - 300.0, 20.0),
                        egui::Align2::LEFT_TOP,
                        "🔍 Development Overlay Active",
                        egui::FontId::proportional(14.0),
                        egui::Color32::YELLOW
                    );
                    
                    ui.painter().text(
                        egui::pos2(screen_rect.width() - 300.0, 40.0),
                        egui::Align2::LEFT_TOP,
                        "Click 'Hide Dev Names' to disable",
                        egui::FontId::proportional(11.0),
                        egui::Color32::WHITE
                    );
                    
                    // Panel boundary indicators (using rect_filled for outlines)
                    let stroke_color = egui::Color32::from_rgba_unmultiplied(255, 255, 0, 100);
                    
                    // Lissajous panel boundary (simple rect outline)
                    let liss_rect = egui::Rect::from_min_size(
                        egui::pos2(10.0, liss_y - 10.0),
                        egui::vec2(screen_rect.width() - 20.0, screen_rect.height() * 0.4)
                    );
                    // Draw outline as 4 thin rectangles
                    ui.painter().rect_filled(egui::Rect::from_min_size(liss_rect.min, egui::vec2(liss_rect.width(), 2.0)), 0.0, stroke_color); // top
                    ui.painter().rect_filled(egui::Rect::from_min_size(liss_rect.min, egui::vec2(2.0, liss_rect.height())), 0.0, stroke_color); // left
                    ui.painter().rect_filled(egui::Rect::from_min_size(egui::pos2(liss_rect.max.x-2.0, liss_rect.min.y), egui::vec2(2.0, liss_rect.height())), 0.0, stroke_color); // right
                    ui.painter().rect_filled(egui::Rect::from_min_size(egui::pos2(liss_rect.min.x, liss_rect.max.y-2.0), egui::vec2(liss_rect.width(), 2.0)), 0.0, stroke_color); // bottom
                    
                    // Signal panels boundary
                    if self.show_signal_view {
                        // Input panel boundary
                        let input_color = egui::Color32::from_rgba_unmultiplied(255, 100, 100, 120);
                        let input_rect = egui::Rect::from_min_size(
                            egui::pos2(10.0, signal_y - 10.0),
                            egui::vec2(screen_rect.width() - 20.0, screen_rect.height() * 0.15)
                        );
                        ui.painter().rect_filled(egui::Rect::from_min_size(input_rect.min, egui::vec2(input_rect.width(), 2.0)), 0.0, input_color);
                        ui.painter().rect_filled(egui::Rect::from_min_size(input_rect.min, egui::vec2(2.0, input_rect.height())), 0.0, input_color);
                        ui.painter().rect_filled(egui::Rect::from_min_size(egui::pos2(input_rect.max.x-2.0, input_rect.min.y), egui::vec2(2.0, input_rect.height())), 0.0, input_color);
                        ui.painter().rect_filled(egui::Rect::from_min_size(egui::pos2(input_rect.min.x, input_rect.max.y-2.0), egui::vec2(input_rect.width(), 2.0)), 0.0, input_color);
                        
                        // Output panel boundary  
                        let output_color = egui::Color32::from_rgba_unmultiplied(100, 255, 100, 120);
                        let output_rect = egui::Rect::from_min_size(
                            egui::pos2(10.0, signal_y + screen_rect.height() * 0.16),
                            egui::vec2(screen_rect.width() - 20.0, screen_rect.height() * 0.15)
                        );
                        ui.painter().rect_filled(egui::Rect::from_min_size(output_rect.min, egui::vec2(output_rect.width(), 2.0)), 0.0, output_color);
                        ui.painter().rect_filled(egui::Rect::from_min_size(output_rect.min, egui::vec2(2.0, output_rect.height())), 0.0, output_color);
                        ui.painter().rect_filled(egui::Rect::from_min_size(egui::pos2(output_rect.max.x-2.0, output_rect.min.y), egui::vec2(2.0, output_rect.height())), 0.0, output_color);
                        ui.painter().rect_filled(egui::Rect::from_min_size(egui::pos2(output_rect.min.x, output_rect.max.y-2.0), egui::vec2(output_rect.width(), 2.0)), 0.0, output_color);
                    }
                });
        }
    }
}

fn main() -> Result<(), eframe::Error> {
    env_logger::init();
    
    let options = NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1600.0, 900.0])  // Wide layout for vertical panel arrangement
            .with_title("D-LinOSS Signal Analyzer - Vertical Layout"),
        ..Default::default()
    };
    
    eframe::run_native(
        "D-LinOSS Signal Analyzer - Vertical Layout",
        options,
        Box::new(|_cc| {
            Ok(Box::new(LissajousApp::new()))
        }),
    )
}