//! dLinOSS Mesmerizing Visualization System
//! 
//! Creates beautiful, brain-like oscillatory patterns using dLinOSS as universal building blocks
//! Inspired by screensavers and mathematical art applications

use burn::{
    tensor::{Tensor, TensorData},
    backend::NdArray,
};
use crate::linoss::{
    dlinoss_layer::{DLinossLayer, DLinossLayerConfig, AParameterization},
};
use ratatui::{
    backend::CrosstermBackend,
    Terminal,
    widgets::{canvas::Canvas, Block, Borders},
    style::{Color, Style},
    layout::{Layout, Constraint, Direction},
    Frame,
};
use crossterm::{
    terminal::{enable_raw_mode, disable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    execute,
    event::{self, Event, KeyCode},
    cursor,
};
use std::{
    io::{stdout, Write},
    time::{Duration, Instant},
    sync::{Arc, atomic::{AtomicBool, Ordering}},
};

type MyBackend = NdArray<f32>;

/// Configuration for the mesmerizing dLinOSS visualization
#[derive(Clone)]
pub struct DLinossVisualizerConfig {
    pub num_oscillators: usize,
    pub canvas_width: f64,
    pub canvas_height: f64,
    pub time_step: f64,
    pub damping_strength: f64,
    pub frequency_range: (f64, f64),
    pub color_cycle_speed: f64,
}

impl Default for DLinossVisualizerConfig {
    fn default() -> Self {
        Self {
            num_oscillators: 8,
            canvas_width: 200.0,
            canvas_height: 100.0,
            time_step: 0.05,
            damping_strength: 0.1,
            frequency_range: (0.5, 3.0),
            color_cycle_speed: 0.1,
        }
    }
}

/// Brain-like oscillatory pattern generator using dLinOSS
pub struct DLinossVisualizer {
    dlinoss_layers: Vec<DLinossLayer<MyBackend>>,
    pub config: DLinossVisualizerConfig,
    time: f64,
    history: Vec<Vec<f64>>, // Store oscillation history for trails
    pub eeg_electrodes: Vec<EEGElectrode>, // EEG-like distance measurements
    device: <MyBackend as burn::tensor::backend::Backend>::Device,
}

impl DLinossVisualizer {
    pub fn new(config: DLinossVisualizerConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let device = <MyBackend as burn::tensor::backend::Backend>::Device::default();
        let mut dlinoss_layers = Vec::new();
        
        // Create multiple dLinOSS layers with well-separated parameters for distinct, discriminable patterns
        for i in 0..config.num_oscillators {
            let layer_config = DLinossLayerConfig {
                d_input: 2,  // 2D input for x,y coordinates
                d_model: 8,  // Multiple oscillator pairs
                d_output: 2, // 2D output for x,y movements
                delta_t: config.time_step,
                init_std: 0.15 + (i as f64 * 0.05), // Varied initialization for spatial separation
                enable_layer_norm: false,
                enable_damping: true,
                init_damping: config.damping_strength * (1.0 + i as f64 * 0.4), // More varied damping for distinct behaviors
                num_damping_scales: 3,
                a_parameterization: AParameterization::GELU,
            };
            
            dlinoss_layers.push(DLinossLayer::new(&layer_config, &device));
        }
        
        // Create EEG electrodes at strategic positions (like real EEG montage)
        let eeg_electrodes = vec![
            EEGElectrode::new(config.canvas_width * 0.2, config.canvas_height * 0.2, "F3".to_string()),
            EEGElectrode::new(config.canvas_width * 0.8, config.canvas_height * 0.2, "F4".to_string()),
            EEGElectrode::new(config.canvas_width * 0.1, config.canvas_height * 0.5, "T3".to_string()),
            EEGElectrode::new(config.canvas_width * 0.9, config.canvas_height * 0.5, "T4".to_string()),
            EEGElectrode::new(config.canvas_width * 0.2, config.canvas_height * 0.8, "P3".to_string()),
            EEGElectrode::new(config.canvas_width * 0.8, config.canvas_height * 0.8, "P4".to_string()),
        ];
        
        Ok(Self {
            dlinoss_layers,
            config: config.clone(),
            time: 0.0,
            history: vec![Vec::new(); config.num_oscillators],
            eeg_electrodes,
            device,
        })
    }
    
    /// Get current points for external access
    pub fn get_current_points(&self) -> Vec<(f64, f64, Color)> {
        let mut points = Vec::new();
        
        for (layer_idx, layer) in self.dlinoss_layers.iter().enumerate() {
            // Create dynamic input based on time and layer index
            let freq = self.config.frequency_range.0 + 
                      (self.config.frequency_range.1 - self.config.frequency_range.0) * 
                      (layer_idx as f64 / self.config.num_oscillators as f64);
            
            let input_x = (self.time * freq).sin();
            let input_y = (self.time * freq * 1.618).cos(); // Golden ratio for interesting patterns
            
            // Create input tensor [1, 1, 2] for single timestep, single batch
            let input = Tensor::<MyBackend, 3>::from_data(
                TensorData::new(vec![input_x as f32, input_y as f32], [1, 1, 2]),
                &self.device,
            );
            
            // Forward through dLinOSS
            let output = layer.forward(input);
            let output_data: Vec<f32> = output.into_data().convert::<f32>().into_vec().unwrap();
            
            // Convert to screen coordinates with much wider spatial separation
            let angle = (layer_idx as f64 / self.config.num_oscillators as f64) * 2.0 * std::f64::consts::PI;
            let radius_base = 0.4; // Much larger base radius for wide separation
            let radius_variation = output_data[0] as f64 * 0.08; // Smaller dynamic variation
            
            let center_x = self.config.canvas_width * 0.5;
            let center_y = self.config.canvas_height * 0.5;
            
            let x = center_x + (radius_base + radius_variation) * self.config.canvas_width * angle.cos() +
                    output_data[1] as f64 * self.config.canvas_width * 0.05; // Smaller local movement
            let y = center_y + (radius_base + radius_variation) * self.config.canvas_height * angle.sin() +
                    output_data[0] as f64 * self.config.canvas_height * 0.05; // Smaller local movement
            
            // Create color based on layer index and time
            let hue = ((layer_idx as f64 / self.config.num_oscillators as f64) * 360.0 + 
                      self.time * self.config.color_cycle_speed * 50.0) % 360.0;
            let color = hue_to_color(hue);
            
            points.push((x, y, color));
        }
        
        points
    }

    /// Generate mesmerizing oscillatory patterns
    pub fn step(&mut self) -> Vec<(f64, f64, Color)> {
        let mut points = Vec::new();
        
        for (layer_idx, layer) in self.dlinoss_layers.iter().enumerate() {
            // Create dynamic input based on time and layer index
            let freq = self.config.frequency_range.0 + 
                      (self.config.frequency_range.1 - self.config.frequency_range.0) * 
                      (layer_idx as f64 / self.config.num_oscillators as f64);
            
            let input_x = (self.time * freq).sin();
            let input_y = (self.time * freq * 1.618).cos(); // Golden ratio for interesting patterns
            
            // Create input tensor [1, 1, 2] for single timestep, single batch
            let input = Tensor::<MyBackend, 3>::from_data(
                TensorData::new(vec![input_x as f32, input_y as f32], [1, 1, 2]),
                &self.device,
            );
            
            // Forward through dLinOSS
            let output = layer.forward(input);
            let output_data: Vec<f32> = output.into_data().convert::<f32>().into_vec().unwrap();
            
            // Convert to screen coordinates with much wider spatial separation
            let angle = (layer_idx as f64 / self.config.num_oscillators as f64) * 2.0 * std::f64::consts::PI;
            let radius_base = 0.4; // Much larger base radius for wide separation
            let radius_variation = output_data[0] as f64 * 0.08; // Smaller dynamic variation
            
            let center_x = self.config.canvas_width * 0.5;
            let center_y = self.config.canvas_height * 0.5;
            
            let x = center_x + (radius_base + radius_variation) * self.config.canvas_width * angle.cos() +
                    output_data[1] as f64 * self.config.canvas_width * 0.05; // Smaller local movement
            let y = center_y + (radius_base + radius_variation) * self.config.canvas_height * angle.sin() +
                    output_data[0] as f64 * self.config.canvas_height * 0.05; // Smaller local movement
            
            // Create color based on layer index and time
            let hue = ((layer_idx as f64 / self.config.num_oscillators as f64) * 360.0 + 
                      self.time * self.config.color_cycle_speed * 50.0) % 360.0;
            let color = hue_to_color(hue);
            
            points.push((x, y, color));
            
            // Store in history for trails
            self.history[layer_idx].push(x);
            self.history[layer_idx].push(y);
            
            // Limit history size
            if self.history[layer_idx].len() > 200 {
                self.history[layer_idx].drain(0..4); // Remove 2 points (4 values)
            }
        }
        
        // Update EEG electrode measurements
        for electrode in &mut self.eeg_electrodes {
            electrode.measure_signal(&points, self.time);
        }
        
        self.time += self.config.time_step;
        points
    }
    
    /// Get trail points for mesmerizing effects
    pub fn get_trails(&self) -> Vec<Vec<(f64, f64, Color)>> {
        let mut trails = Vec::new();
        
        for (layer_idx, history) in self.history.iter().enumerate() {
            let mut trail = Vec::new();
            
            for i in (0..history.len()).step_by(2) {
                if i + 1 < history.len() {
                    let x = history[i];
                    let y = history[i + 1];
                    
                    // Fade color based on age
                    let age = (history.len() - i) as f64 / history.len() as f64;
                    let alpha = (age * 0.8) as u8;
                    
                    let base_hue = (layer_idx as f64 / self.config.num_oscillators as f64) * 360.0;
                    let color = hue_to_color_with_alpha(base_hue, alpha);
                    
                    trail.push((x, y, color));
                }
            }
            
            trails.push(trail);
        }
        
        trails
    }
}

/// Convert HSV hue to RGB color
fn hue_to_color(hue: f64) -> Color {
    hue_to_color_with_alpha(hue, 255)
}

fn hue_to_color_with_alpha(hue: f64, _alpha: u8) -> Color {
    let h = (hue % 360.0) / 60.0;
    let c = 1.0;
    let x = c * (1.0 - ((h % 2.0) - 1.0).abs());
    
    let (r, g, b) = match h as i32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    
    Color::Rgb(
        (r * 255.0) as u8,
        (g * 255.0) as u8,
        (b * 255.0) as u8,
    )
}

/// EEG electrode simulation for distance-based neural signal measurement
#[derive(Clone)]
pub struct EEGElectrode {
    pub x: f64,
    pub y: f64,
    pub label: String,
    pub signal_history: Vec<f64>,
    pub max_history: usize,
}

impl EEGElectrode {
    pub fn new(x: f64, y: f64, label: String) -> Self {
        Self {
            x,
            y,
            label,
            signal_history: Vec::new(),
            max_history: 200, // Store enough for visible wave patterns
        }
    }
    
    /// Measure blurred signal from all oscillators based on distance
    pub fn measure_signal(&mut self, oscillators: &[(f64, f64, Color)], time: f64) {
        let mut total_signal = 0.0;
        
        for (osc_x, osc_y, _) in oscillators {
            // Calculate distance from electrode to oscillator
            let distance = ((self.x - osc_x).powi(2) + (self.y - osc_y).powi(2)).sqrt();
            
            // Signal strength decreases with distance (1/r falloff like real EEG)
            let distance_factor = 1.0 / (1.0 + distance * 0.01);
            
            // Add some oscillatory content based on position and time
            let signal_strength = (time * 2.0 + osc_x * 0.1 + osc_y * 0.1).sin() * distance_factor;
            total_signal += signal_strength;
        }
        
        // Add some realistic EEG noise
        let noise = (time * 13.7).sin() * 0.1;
        total_signal += noise;
        
        // Store in history
        self.signal_history.push(total_signal);
        if self.signal_history.len() > self.max_history {
            self.signal_history.remove(0);
        }
    }
}

/// Render the mesmerizing dLinOSS visualization
pub fn render_dlinoss_art(frame: &mut Frame, visualizer: &mut DLinossVisualizer) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(70), Constraint::Percentage(30)])
        .split(frame.area());
    
    // Left panel: Neural oscillator visualization
    let neural_area = chunks[0];
    let eeg_area = chunks[1];
    
    // Get trails and current points outside the closure
    let trails = visualizer.get_trails();
    let current_points = visualizer.step();
    
    // Calculate canvas bounds based on terminal size for maximum usage - ZOOMED IN 12X
    let canvas_width = (neural_area.width as f64 - 4.0) * 24.0; // Triple the 4x: was 8.0, now 24.0
    let canvas_height = (neural_area.height as f64 - 4.0) * 48.0; // Triple the 4x: was 16.0, now 48.0
    
    let neural_canvas = Canvas::default()
        .block(Block::default()
            .title("dLinOSS Neural Oscillators - ZOOMED 12X")
            .borders(Borders::ALL)
            .style(Style::default().fg(Color::Cyan)))
        .x_bounds([0.0, canvas_width])
        .y_bounds([0.0, canvas_height])
        .paint(|ctx| {
            // Scale factor to map from fixed config bounds to dynamic canvas bounds
            let scale_x = canvas_width / visualizer.config.canvas_width;
            let scale_y = canvas_height / visualizer.config.canvas_height;
            
            // Draw trails as connecting lines instead of blobs
            for trail in &trails {
                let mut prev_point: Option<(f64, f64)> = None;
                for (x, y, color) in trail {
                    let scaled_x = x * scale_x;
                    let scaled_y = y * scale_y;
                    
                    if let Some((prev_x, prev_y)) = prev_point {
                        // Draw line connecting trail points
                        ctx.draw(&ratatui::widgets::canvas::Line {
                            x1: prev_x,
                            y1: prev_y,
                            x2: scaled_x,
                            y2: scaled_y,
                            color: *color,
                        });
                    }
                    prev_point = Some((scaled_x, scaled_y));
                }
            }
            
            // Draw current oscillator positions as distinct shapes
            for (i, (x, y, color)) in current_points.iter().enumerate() {
                let scaled_x = x * scale_x;
                let scaled_y = y * scale_y;
                
                match i % 3 {
                    0 => {
                        // Cross pattern
                        let size = 8.0 * scale_x.min(scale_y);
                        ctx.draw(&ratatui::widgets::canvas::Line {
                            x1: scaled_x - size, y1: scaled_y, x2: scaled_x + size, y2: scaled_y, color: *color,
                        });
                        ctx.draw(&ratatui::widgets::canvas::Line {
                            x1: scaled_x, y1: scaled_y - size, x2: scaled_x, y2: scaled_y + size, color: *color,
                        });
                    },
                    1 => {
                        // Diamond pattern
                        let size = 6.0 * scale_x.min(scale_y);
                        ctx.draw(&ratatui::widgets::canvas::Line {
                            x1: scaled_x, y1: scaled_y - size, x2: scaled_x + size, y2: scaled_y, color: *color,
                        });
                        ctx.draw(&ratatui::widgets::canvas::Line {
                            x1: scaled_x + size, y1: scaled_y, x2: scaled_x, y2: scaled_y + size, color: *color,
                        });
                        ctx.draw(&ratatui::widgets::canvas::Line {
                            x1: scaled_x, y1: scaled_y + size, x2: scaled_x - size, y2: scaled_y, color: *color,
                        });
                        ctx.draw(&ratatui::widgets::canvas::Line {
                            x1: scaled_x - size, y1: scaled_y, x2: scaled_x, y2: scaled_y - size, color: *color,
                        });
                    },
                    _ => {
                        // Small circle outline only
                        ctx.draw(&ratatui::widgets::canvas::Circle {
                            x: scaled_x,
                            y: scaled_y,
                            radius: 4.0 * scale_x.min(scale_y),
                            color: *color,
                        });
                    }
                }
            }
            
            // Draw EEG electrode positions
            for electrode in &visualizer.eeg_electrodes {
                let electrode_x = electrode.x * scale_x / visualizer.config.canvas_width * canvas_width;
                let electrode_y = electrode.y * scale_y / visualizer.config.canvas_height * canvas_height;
                
                ctx.draw(&ratatui::widgets::canvas::Circle {
                    x: electrode_x,
                    y: electrode_y,
                    radius: 3.0 * scale_x.min(scale_y),
                    color: Color::White,
                });
            }
        });
    
    frame.render_widget(neural_canvas, neural_area);
    
    // Right panel: EEG-like waves
    let eeg_canvas = Canvas::default()
        .block(Block::default()
            .title("EEG-like Distance Measurements")
            .borders(Borders::ALL)
            .style(Style::default().fg(Color::Yellow)))
        .x_bounds([0.0, 200.0])
        .y_bounds([-3.0, 3.0])
        .paint(|ctx| {
            // Draw EEG waveforms
            for (electrode_idx, electrode) in visualizer.eeg_electrodes.iter().enumerate() {
                if electrode.signal_history.len() > 1 {
                    let y_offset = 2.5 - (electrode_idx as f64 * 1.0); // Stack waveforms vertically
                    
                    for i in 1..electrode.signal_history.len() {
                        let x1 = (i - 1) as f64;
                        let x2 = i as f64;
                        let y1 = electrode.signal_history[i - 1] * 0.3 + y_offset;
                        let y2 = electrode.signal_history[i] * 0.3 + y_offset;
                        
                        ctx.draw(&ratatui::widgets::canvas::Line {
                            x1, y1, x2, y2,
                            color: match electrode_idx {
                                0 => Color::Red,
                                1 => Color::Green,
                                2 => Color::Blue,
                                3 => Color::Magenta,
                                4 => Color::Cyan,
                                _ => Color::Yellow,
                            },
                        });
                    }
                }
            }
        });
    
    frame.render_widget(eeg_canvas, eeg_area);
}

/// Run the mesmerizing dLinOSS visualization
pub fn run_dlinoss_visualizer() -> Result<(), Box<dyn std::error::Error>> {
    // Create a shared flag for graceful shutdown
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    
    // Create a more robust cleanup function
    let cleanup = || {
        // Force terminal restoration
        let _ = disable_raw_mode();
        let _ = execute!(std::io::stderr(), LeaveAlternateScreen);
        let _ = execute!(std::io::stderr(), cursor::Show);
        // Additional cleanup - restore cursor and clear screen
        print!("\x1b[?25h\x1b[2J\x1b[H"); // Show cursor, clear screen, go to home
        let _ = std::io::stdout().flush();
    };
    
    // Setup signal handler that does immediate cleanup
    ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
        // Do immediate cleanup in signal handler
        let _ = disable_raw_mode();
        let _ = execute!(std::io::stderr(), LeaveAlternateScreen);
        let _ = execute!(std::io::stderr(), cursor::Show);
        print!("\x1b[?25h\x1b[2J\x1b[H"); // Show cursor, clear screen, go to home
        let _ = std::io::stdout().flush();
        std::process::exit(0);
    })?;
    
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    
    // Create visualizer
    let config = DLinossVisualizerConfig {
        num_oscillators: 6,  // Reduced from 12 to 6 for less clutter
        canvas_width: 200.0,
        canvas_height: 80.0,
        time_step: 0.05,    // Slightly slower for more visible motion
        damping_strength: 0.08,
        frequency_range: (0.3, 1.5), // Narrower range for more distinct frequencies
        color_cycle_speed: 0.15,
    };
    
    let result = (|| -> Result<(), Box<dyn std::error::Error>> {
        let mut visualizer = DLinossVisualizer::new(config)?;
        let mut last_time = Instant::now();
        
        while running.load(Ordering::SeqCst) {
            // Check for input (non-blocking)
            if event::poll(Duration::from_millis(16))? {
                if let Event::Key(key) = event::read()? {
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => break,
                        KeyCode::Char('r') => {
                            // Reset the visualization
                            visualizer = DLinossVisualizer::new(DLinossVisualizerConfig::default())?;
                        },
                        _ => {}
                    }
                }
            }
            
            // Render at ~60 FPS
            if last_time.elapsed() >= Duration::from_millis(16) {
                terminal.draw(|frame| {
                    render_dlinoss_art(frame, &mut visualizer);
                })?;
                last_time = Instant::now();
            }
            
            // Check if we should exit due to signal
            if !running.load(Ordering::SeqCst) {
                break;
            }
        }
        Ok(())
    })();
    
    // Always cleanup, regardless of how we exit (normal, signal, or error)
    cleanup();
    
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_dlinoss_visualizer_creation() {
        let config = DLinossVisualizerConfig::default();
        let visualizer = DLinossVisualizer::new(config);
        assert!(visualizer.is_ok());
    }
    
    #[test]
    fn test_oscillator_step() {
        let mut visualizer = DLinossVisualizer::new(DLinossVisualizerConfig::default()).unwrap();
        let points = visualizer.step();
        assert_eq!(points.len(), visualizer.config.num_oscillators);
    }
}
