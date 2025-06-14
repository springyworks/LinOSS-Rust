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
    device: <MyBackend as burn::tensor::backend::Backend>::Device,
}

impl DLinossVisualizer {
    pub fn new(config: DLinossVisualizerConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let device = <MyBackend as burn::tensor::backend::Backend>::Device::default();
        let mut dlinoss_layers = Vec::new();
        
        // Create multiple dLinOSS layers with different parameters for rich dynamics
        for i in 0..config.num_oscillators {
            let layer_config = DLinossLayerConfig {
                d_input: 2,  // 2D input for x,y coordinates
                d_model: 8,  // Multiple oscillator pairs
                d_output: 2, // 2D output for x,y movements
                delta_t: config.time_step,
                init_std: 0.1,
                enable_layer_norm: false,
                enable_damping: true,
                init_damping: config.damping_strength * (1.0 + i as f64 * 0.2), // Varied damping
                num_damping_scales: 3,
                a_parameterization: AParameterization::GELU,
            };
            
            dlinoss_layers.push(DLinossLayer::new(&layer_config, &device));
        }
        
        Ok(Self {
            dlinoss_layers,
            config: config.clone(),
            time: 0.0,
            history: vec![Vec::new(); config.num_oscillators],
            device,
        })
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
            
            // Convert to screen coordinates
            let x = (output_data[0] as f64 * self.config.canvas_width * 0.3) + self.config.canvas_width * 0.5;
            let y = (output_data[1] as f64 * self.config.canvas_height * 0.3) + self.config.canvas_height * 0.5;
            
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

/// Render the mesmerizing dLinOSS visualization
pub fn render_dlinoss_art(frame: &mut Frame, visualizer: &mut DLinossVisualizer) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(100)])
        .split(frame.area());
    
    // Get trails and current points outside the closure
    let trails = visualizer.get_trails();
    let current_points = visualizer.step();
    
    let canvas = Canvas::default()
        .block(Block::default()
            .title("dLinOSS Brain Dynamics - Universal Building Blocks")
            .borders(Borders::ALL)
            .style(Style::default().fg(Color::Cyan)))
        .x_bounds([0.0, visualizer.config.canvas_width])
        .y_bounds([0.0, visualizer.config.canvas_height])
        .paint(|ctx| {
            // Draw trails for mesmerizing effect
            for trail in &trails {
                for (x, y, color) in trail {
                    ctx.draw(&ratatui::widgets::canvas::Circle {
                        x: *x,
                        y: *y,
                        radius: 0.5,
                        color: *color,
                    });
                }
            }
            
            // Draw current oscillator positions
            for (x, y, color) in &current_points {
                ctx.draw(&ratatui::widgets::canvas::Circle {
                    x: *x,
                    y: *y,
                    radius: 2.0,
                    color: *color,
                });
            }
        });
    
    frame.render_widget(canvas, chunks[0]);
}

/// Run the mesmerizing dLinOSS visualization
pub fn run_dlinoss_screensaver() -> Result<(), Box<dyn std::error::Error>> {
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
        num_oscillators: 12,
        canvas_width: 200.0,
        canvas_height: 80.0,
        time_step: 0.03,
        damping_strength: 0.05,
        frequency_range: (0.2, 2.0),
        color_cycle_speed: 0.2,
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
