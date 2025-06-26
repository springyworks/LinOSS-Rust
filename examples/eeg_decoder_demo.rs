//! EEG Decoder Demo - Complete Brain-Computer Interface Simulation
//! 
//! This example demonstrates the full pipeline:
//! 1. Generate synthetic brain activity using dLinOSS oscillators
//! 2. Simulate EEG electrode measurements with realistic noise and distance falloff
//! 3. Collect training data (true positions + EEG signals)
//! 4. Train a neural network to decode the internal brain activity from EEG
//! 5. Visualize both the original and reconstructed brain activity in real-time
//! 
//! This simulates real-world BCI challenges where we only have access to noisy
//! external measurements but want to reconstruct the internal neural dynamics.

use linoss_rust::{
    visualization::dlinoss_art::{DLinossVisualizer, DLinossVisualizerConfig, EEGElectrode},
    analysis::eeg_decoder::{EEGDecoder, EEGDecoderConfig, TrainingData, train_eeg_decoder},
};
use burn::{
    prelude::*,
    backend::{NdArray, Autodiff},
    tensor::{Tensor, TensorData},
};
use ratatui::{
    backend::CrosstermBackend,
    Terminal,
    widgets::{canvas::Canvas, Block, Borders, Paragraph, Gauge, Clear, Sparkline},
    style::{Color, Style, Modifier},
    layout::{Layout, Constraint, Direction, Alignment, Rect},
    text::{Line, Span},
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

type MyBackend = Autodiff<NdArray<f32>>;

/// EEG Decoder Demo with complete BCI pipeline
pub struct EEGDecoderDemo {
    brain_visualizer: DLinossVisualizer,
    decoder: Option<EEGDecoder<MyBackend>>,
    decoder_config: EEGDecoderConfig,
    device: <MyBackend as burn::tensor::backend::Backend>::Device,
    
    // Data collection
    training_data: Vec<TrainingData<MyBackend>>,
    collecting_data: bool,
    collection_progress: usize,
    target_samples: usize,
    
    // Real-time decoding
    current_eeg_buffer: Vec<Vec<f64>>, // [channels][time_history]
    decoded_positions: Vec<(f64, f64)>, // Current decoder predictions
    
    // Visualization state
    show_training_data: bool,
    show_decoder_output: bool,
    frame_count: u64,
    last_collection_time: Instant,
}

impl EEGDecoderDemo {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let mut config = DLinossVisualizerConfig::default();
        config.num_oscillators = 6;
        config.canvas_width = 120.0;
        config.canvas_height = 60.0;
        
        let brain_visualizer = DLinossVisualizer::new(config)?;
        
        let decoder_config = EEGDecoderConfig {
            num_eeg_channels: 6,
            sequence_length: 50,
            num_oscillators: 6,
            hidden_dim: 128,
            dropout_rate: 0.1,
        };
        
        let device = Default::default();
        
        // Initialize EEG buffer
        let current_eeg_buffer = vec![Vec::new(); decoder_config.num_eeg_channels];
        let decoded_positions = vec![(0.0, 0.0); decoder_config.num_oscillators];
        
        Ok(Self {
            brain_visualizer,
            decoder: None,
            decoder_config,
            device,
            training_data: Vec::new(),
            collecting_data: false,
            collection_progress: 0,
            target_samples: 200, // Collect 200 training samples
            current_eeg_buffer,
            decoded_positions,
            show_training_data: false,
            show_decoder_output: false,
            frame_count: 0,
            last_collection_time: Instant::now(),
        })
    }
    
    /// Start collecting training data
    pub fn start_data_collection(&mut self) {
        println!("🧠 Starting EEG training data collection...");
        println!("📊 Target: {} samples", self.target_samples);
        self.collecting_data = true;
        self.collection_progress = 0;
        self.training_data.clear();
        self.last_collection_time = Instant::now();
    }
    
    /// Collect one training sample
    pub fn collect_training_sample(&mut self) {
        if !self.collecting_data || self.collection_progress >= self.target_samples {
            return;
        }
        
        // Get current brain activity (ground truth)
        let current_points = self.brain_visualizer.step();
        
        // Extract true oscillator positions
        let mut true_positions = Vec::new();
        for (x, y, _) in &current_points {
            true_positions.push(*x as f32);
            true_positions.push(*y as f32);
        }
        
        // Get EEG measurements from electrodes
        let eeg_readings = self.get_current_eeg_readings();
        
        // Update EEG buffer for sequence
        for (channel_idx, reading) in eeg_readings.iter().enumerate() {
            self.current_eeg_buffer[channel_idx].push(*reading);
            
            // Keep only the last sequence_length samples
            if self.current_eeg_buffer[channel_idx].len() > self.decoder_config.sequence_length {
                self.current_eeg_buffer[channel_idx].remove(0);
            }
        }
        
        // Only create training sample if we have enough EEG history
        if self.current_eeg_buffer[0].len() >= self.decoder_config.sequence_length {
            // Create EEG tensor [1, channels, time]
            let mut eeg_data = Vec::new();
            for channel in &self.current_eeg_buffer {
                let start_idx = channel.len() - self.decoder_config.sequence_length;
                for i in start_idx..channel.len() {
                    eeg_data.push(channel[i] as f32);
                }
            }
            
            let eeg_tensor = Tensor::<MyBackend, 3>::from_data(
                TensorData::new(
                    eeg_data,
                    [1, self.decoder_config.num_eeg_channels, self.decoder_config.sequence_length]
                ),
                &self.device,
            );
            
            let positions_tensor = Tensor::<MyBackend, 2>::from_data(
                TensorData::new(true_positions, [1, self.decoder_config.num_oscillators * 2]),
                &self.device,
            );
            
            let training_sample = TrainingData {
                eeg_signals: eeg_tensor,
                true_positions: positions_tensor,
            };
            
            self.training_data.push(training_sample);
            self.collection_progress += 1;
            
            // Print progress periodically
            if self.collection_progress % 20 == 0 {
                println!("📊 Collected {}/{} training samples", 
                        self.collection_progress, self.target_samples);
            }
        }
        
        // Finish collection when we have enough samples
        if self.collection_progress >= self.target_samples {
            self.collecting_data = false;
            println!("✅ Data collection complete! Starting decoder training...");
            self.train_decoder();
        }
    }
    
    /// Train the EEG decoder neural network
    pub fn train_decoder(&mut self) {
        if self.training_data.is_empty() {
            println!("❌ No training data available!");
            return;
        }
        
        let decoder = train_eeg_decoder(
            self.training_data.clone(),
            &self.decoder_config,
            &self.device,
            100, // 100 training epochs
        );
        
        self.decoder = Some(decoder);
        self.show_decoder_output = true;
        println!("🎯 EEG Decoder ready for real-time reconstruction!");
    }
    
    /// Get current EEG readings from all electrodes
    pub fn get_current_eeg_readings(&self) -> Vec<f64> {
        self.brain_visualizer.eeg_electrodes
            .iter()
            .map(|electrode| {
                electrode.signal_history.last().copied().unwrap_or(0.0)
            })
            .collect()
    }
    
    /// Run real-time EEG decoding
    pub fn decode_current_eeg(&mut self) {
        if let Some(ref decoder) = self.decoder {
            // Update EEG buffer
            let eeg_readings = self.get_current_eeg_readings();
            for (channel_idx, reading) in eeg_readings.iter().enumerate() {
                self.current_eeg_buffer[channel_idx].push(*reading);
                if self.current_eeg_buffer[channel_idx].len() > self.decoder_config.sequence_length {
                    self.current_eeg_buffer[channel_idx].remove(0);
                }
            }
            
            // Only decode if we have enough history
            if self.current_eeg_buffer[0].len() >= self.decoder_config.sequence_length {
                // Prepare EEG input tensor
                let mut eeg_data = Vec::new();
                for channel in &self.current_eeg_buffer {
                    let start_idx = channel.len() - self.decoder_config.sequence_length;
                    for i in start_idx..channel.len() {
                        eeg_data.push(channel[i] as f32);
                    }
                }
                
                let eeg_tensor = Tensor::<MyBackend, 2>::from_data(
                    TensorData::new(
                        eeg_data,
                        [self.decoder_config.num_eeg_channels * self.decoder_config.sequence_length]
                    ),
                    &self.device,
                );
                
                // Add batch dimension and decode
                let eeg_input = eeg_tensor.unsqueeze_dim(0); // [1, features]
                let decoded = decoder.forward(eeg_input);
                let decoded_data: Vec<f32> = decoded.into_data().convert::<f32>().into_vec().unwrap();
                
                // Extract decoded positions
                self.decoded_positions.clear();
                for i in (0..decoded_data.len()).step_by(2) {
                    if i + 1 < decoded_data.len() {
                        self.decoded_positions.push((
                            decoded_data[i] as f64,
                            decoded_data[i + 1] as f64,
                        ));
                    }
                }
            }
        }
    }
    
    /// Update simulation
    pub fn update(&mut self) {
        self.frame_count += 1;
        
        // Collect training data if needed
        if self.collecting_data {
            self.collect_training_sample();
        }
        
        // Run real-time decoding
        if self.decoder.is_some() && !self.collecting_data {
            self.decode_current_eeg();
        }
        
        // Always step the brain visualizer to generate new activity
        if !self.collecting_data {
            self.brain_visualizer.step();
        }
    }
}

/// Handle keyboard input
pub fn handle_input(demo: &mut EEGDecoderDemo) -> Result<bool, Box<dyn std::error::Error>> {
    if event::poll(Duration::from_millis(16))? {
        if let Event::Key(key) = event::read()? {
            match key.code {
                KeyCode::Char('q') | KeyCode::Esc => return Ok(true),
                KeyCode::Char('c') => {
                    if demo.decoder.is_none() && !demo.collecting_data {
                        demo.start_data_collection();
                    }
                },
                KeyCode::Char('t') => {
                    demo.show_training_data = !demo.show_training_data;
                },
                KeyCode::Char('d') => {
                    demo.show_decoder_output = !demo.show_decoder_output;
                },
                _ => {}
            }
        }
    }
    Ok(false)
}

/// Render the complete EEG BCI demo
pub fn render_demo(frame: &mut Frame, demo: &mut EEGDecoderDemo) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // Status bar
            Constraint::Min(10),    // Main content
            Constraint::Length(4),  // Help/controls
        ])
        .split(frame.area());
    
    // Status bar
    render_status_bar(frame, demo, chunks[0]);
    
    // Main content area
    let main_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(50), // Original brain activity
            Constraint::Percentage(25), // EEG signals
            Constraint::Percentage(25), // Decoded reconstruction
        ])
        .split(chunks[1]);
    
    // Original brain activity (left panel)
    render_brain_activity(frame, demo, main_chunks[0]);
    
    // EEG signals (middle panel)
    render_eeg_signals(frame, demo, main_chunks[1]);
    
    // Decoded reconstruction (right panel)
    render_decoded_activity(frame, demo, main_chunks[2]);
    
    // Help panel
    render_help(frame, demo, chunks[2]);
}

/// Render status bar
fn render_status_bar(frame: &mut Frame, demo: &EEGDecoderDemo, area: Rect) {
    let status = if demo.collecting_data {
        format!("🧠 COLLECTING TRAINING DATA: {}/{} samples", 
               demo.collection_progress, demo.target_samples)
    } else if demo.decoder.is_some() {
        format!("🎯 EEG DECODER ACTIVE - Real-time reconstruction running")
    } else {
        "⚡ dLinOSS Brain Dynamics - Press 'c' to start data collection".to_string()
    };
    
    let status_paragraph = Paragraph::new(status)
        .style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))
        .block(Block::default().borders(Borders::ALL).title("Status"));
    
    frame.render_widget(status_paragraph, area);
}

/// Render original brain activity
fn render_brain_activity(frame: &mut Frame, demo: &mut EEGDecoderDemo, area: Rect) {
    let current_points = demo.brain_visualizer.get_current_points();
    let trails = demo.brain_visualizer.get_trails();
    
    let canvas = Canvas::default()
        .block(Block::default()
            .borders(Borders::ALL)
            .title("🧠 Original Brain Activity"))
        .x_bounds([0.0, demo.brain_visualizer.config.canvas_width])
        .y_bounds([0.0, demo.brain_visualizer.config.canvas_height])
        .paint(|ctx| {
            // Draw trails
            for (layer_idx, trail) in trails.iter().enumerate() {
                let color = match layer_idx % 6 {
                    0 => Color::Red,
                    1 => Color::Green,
                    2 => Color::Blue,
                    3 => Color::Yellow,
                    4 => Color::Magenta,
                    _ => Color::Cyan,
                };
                
                for (x, y, _) in trail {
                    ctx.draw(&ratatui::widgets::canvas::Points {
                        coords: &[(*x, *y)],
                        color,
                    });
                }
            }
            
            // Draw current oscillator positions
            for (i, (x, y, _)) in current_points.iter().enumerate() {
                let color = match i % 6 {
                    0 => Color::Red,
                    1 => Color::Green,
                    2 => Color::Blue,
                    3 => Color::Yellow,
                    4 => Color::Magenta,
                    _ => Color::Cyan,
                };
                
                ctx.draw(&ratatui::widgets::canvas::Circle {
                    x: *x,
                    y: *y,
                    radius: 3.0,
                    color,
                });
            }
            
            // Draw EEG electrodes
            for electrode in &demo.brain_visualizer.eeg_electrodes {
                ctx.draw(&ratatui::widgets::canvas::Points {
                    coords: &[(electrode.x, electrode.y)],
                    color: Color::White,
                });
            }
        });
    
    frame.render_widget(canvas, area);
}

/// Render EEG signals
fn render_eeg_signals(frame: &mut Frame, demo: &EEGDecoderDemo, area: Rect) {
    let eeg_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints((0..6).map(|_| Constraint::Percentage(16)).collect::<Vec<_>>())
        .split(area);
    
    for (i, electrode) in demo.brain_visualizer.eeg_electrodes.iter().enumerate() {
        if i < eeg_chunks.len() {
            let sparkline_data: Vec<u64> = electrode.signal_history
                .iter()
                .map(|&x| ((x + 1.0) * 50.0) as u64)
                .collect();
            
            let sparkline = Sparkline::default()
                .block(Block::default()
                    .borders(Borders::ALL)
                    .title(format!("EEG {}", electrode.label)))
                .data(&sparkline_data)
                .style(Style::default().fg(Color::Green));
            
            frame.render_widget(sparkline, eeg_chunks[i]);
        }
    }
}

/// Render decoded brain activity reconstruction
fn render_decoded_activity(frame: &mut Frame, demo: &EEGDecoderDemo, area: Rect) {
    let title = if demo.decoder.is_some() {
        "🎯 Decoded Reconstruction"
    } else {
        "🎯 Decoder (Training Required)"
    };
    
    let canvas = Canvas::default()
        .block(Block::default()
            .borders(Borders::ALL)
            .title(title))
        .x_bounds([0.0, demo.brain_visualizer.config.canvas_width])
        .y_bounds([0.0, demo.brain_visualizer.config.canvas_height])
        .paint(|ctx| {
            if demo.show_decoder_output && demo.decoder.is_some() {
                // Draw decoded positions
                for (i, (x, y)) in demo.decoded_positions.iter().enumerate() {
                    let color = match i % 6 {
                        0 => Color::Red,
                        1 => Color::Green,
                        2 => Color::Blue,
                        3 => Color::Yellow,
                        4 => Color::Magenta,
                        _ => Color::Cyan,
                    };
                    
                    // Scale decoded positions to canvas
                    let canvas_x = x * demo.brain_visualizer.config.canvas_width * 0.5 + 
                                  demo.brain_visualizer.config.canvas_width * 0.5;
                    let canvas_y = y * demo.brain_visualizer.config.canvas_height * 0.5 + 
                                  demo.brain_visualizer.config.canvas_height * 0.5;
                    
                    ctx.draw(&ratatui::widgets::canvas::Circle {
                        x: canvas_x,
                        y: canvas_y,
                        radius: 3.0,
                        color,
                    });
                }
            }
        });
    
    frame.render_widget(canvas, area);
}

/// Render help panel
fn render_help(frame: &mut Frame, demo: &EEGDecoderDemo, area: Rect) {
    let help_text = vec![
        Line::from(vec![
            Span::styled("Controls: ", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
            Span::raw("'c' = Collect Data | 't' = Toggle Training View | 'd' = Toggle Decoder | 'q' = Quit"),
        ]),
        Line::from(vec![
            Span::styled("Pipeline: ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
            Span::raw("Brain Activity → EEG Electrodes → Neural Decoder → Reconstructed Activity"),
        ]),
    ];
    
    let help_paragraph = Paragraph::new(help_text)
        .block(Block::default().borders(Borders::ALL).title("Help & Status"))
        .alignment(Alignment::Left);
    
    frame.render_widget(help_paragraph, area);
}

/// Setup signal handling for clean exit
fn setup_signal_handling() -> Arc<AtomicBool> {
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    
    ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
    }).expect("Error setting Ctrl-C handler");
    
    running
}

/// Cleanup terminal state
fn cleanup_terminal() -> Result<(), Box<dyn std::error::Error>> {
    disable_raw_mode()?;
    execute!(stdout(), LeaveAlternateScreen, cursor::Show)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 EEG Decoder Demo - Complete BCI Pipeline");
    println!("🎯 Simulating real-time brain-computer interface");
    println!("⚡ Starting visualization...\n");
    
    // Setup terminal
    enable_raw_mode()?;
    execute!(stdout(), EnterAlternateScreen, cursor::Hide)?;
    
    let backend = CrosstermBackend::new(stdout());
    let mut terminal = Terminal::new(backend)?;
    
    // Setup signal handling
    let running = setup_signal_handling();
    
    // Create demo
    let mut demo = EEGDecoderDemo::new()?;
    
    let frame_duration = Duration::from_millis(50); // 20 FPS
    let mut last_frame = Instant::now();
    
    // Main loop
    while running.load(Ordering::SeqCst) {
        // Handle input
        if handle_input(&mut demo)? {
            break;
        }
        
        // Update at consistent frame rate
        if last_frame.elapsed() >= frame_duration {
            demo.update();
            
            terminal.draw(|frame| {
                render_demo(frame, &mut demo);
            })?;
            
            last_frame = Instant::now();
        }
        
        std::thread::sleep(Duration::from_millis(1));
    }
    
    // Cleanup
    cleanup_terminal()?;
    
    println!("\n🎯 EEG Decoder Demo completed!");
    if demo.decoder.is_some() {
        println!("✅ Successfully trained decoder with {} samples", demo.training_data.len());
        println!("🧠 Demonstrated complete BCI pipeline: Brain → EEG → Neural Decoder → Reconstruction");
    }
    
    Ok(())
}
