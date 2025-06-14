//! Interactive dLinOSS Brain Dynamics Explorer
//! 
//! Real-time parameter adjustment and multiple visualization modes
//! Demonstrates dLinOSS as universal building blocks for complex brain-like behaviors

use linoss_rust::{
    visualization::dlinoss_art::{DLinossVisualizer, DLinossVisualizerConfig},
};
use ratatui::{
    backend::CrosstermBackend,
    Terminal,
    widgets::{canvas::Canvas, Block, Borders, Paragraph, Gauge, Clear},
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

/// Interactive brain dynamics explorer with real-time parameter control
pub struct BrainDynamicsExplorer {
    visualizer: DLinossVisualizer,
    damping: f64,
    frequency: f64,
    complexity: usize,
    mode: VisualizationMode,
    show_help: bool,
    frame_count: u64,
}

#[derive(Debug, Clone, Copy)]
pub enum VisualizationMode {
    Oscillators,     // Individual oscillator trails
    Network,         // Connected network visualization  
    BrainWaves,      // EEG-like wave patterns
    Synchronization, // Phase synchronization patterns
}

impl BrainDynamicsExplorer {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let config = DLinossVisualizerConfig {
            num_oscillators: 6,
            canvas_width: 180.0,
            canvas_height: 50.0,
            time_step: 0.04,
            damping_strength: 0.08,
            frequency_range: (0.5, 2.5),
            color_cycle_speed: 0.15,
        };
        
        let visualizer = DLinossVisualizer::new(config)?;
        
        Ok(Self {
            visualizer,
            damping: 0.08,
            frequency: 1.5,
            complexity: 6,
            mode: VisualizationMode::Oscillators,
            show_help: false,
            frame_count: 0,
        })
    }
    
    /// Handle user input for real-time parameter adjustment
    pub fn handle_input(&mut self, key: KeyCode) -> bool {
        match key {
            KeyCode::Char('q') | KeyCode::Esc => return false,
            KeyCode::Char('h') => self.show_help = !self.show_help,
            KeyCode::Char('1') => self.mode = VisualizationMode::Oscillators,
            KeyCode::Char('2') => self.mode = VisualizationMode::Network,
            KeyCode::Char('3') => self.mode = VisualizationMode::BrainWaves,
            KeyCode::Char('4') => self.mode = VisualizationMode::Synchronization,
            KeyCode::Up => {
                self.damping = (self.damping + 0.01).min(0.5);
                self.update_visualizer();
            },
            KeyCode::Down => {
                self.damping = (self.damping - 0.01).max(0.0);
                self.update_visualizer();
            },
            KeyCode::Right => {
                self.frequency = (self.frequency + 0.1).min(5.0);
                self.update_visualizer();
            },
            KeyCode::Left => {
                self.frequency = (self.frequency - 0.1).max(0.1);
                self.update_visualizer();
            },
            KeyCode::Char('+') => {
                self.complexity = (self.complexity + 1).min(20);
                self.recreate_visualizer();
            },
            KeyCode::Char('-') => {
                self.complexity = (self.complexity.saturating_sub(1)).max(2);
                self.recreate_visualizer();
            },
            KeyCode::Char('r') => {
                self.recreate_visualizer();
            },
            _ => {}
        }
        true
    }
    
    fn update_visualizer(&mut self) {
        // Update existing visualizer parameters
        self.visualizer.config.damping_strength = self.damping;
        self.visualizer.config.frequency_range = (self.frequency * 0.3, self.frequency * 1.7);
    }
    
    fn recreate_visualizer(&mut self) {
        let config = DLinossVisualizerConfig {
            num_oscillators: self.complexity,
            canvas_width: 180.0,
            canvas_height: 50.0,
            time_step: 0.04,
            damping_strength: self.damping,
            frequency_range: (self.frequency * 0.3, self.frequency * 1.7),
            color_cycle_speed: 0.15,
        };
        
        if let Ok(new_visualizer) = DLinossVisualizer::new(config) {
            self.visualizer = new_visualizer;
        }
    }
    
    /// Render the brain dynamics visualization with controls
    pub fn render(&mut self, frame: &mut Frame) {
        self.frame_count += 1;
        
        let main_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),  // Title
                Constraint::Min(20),    // Main visualization
                Constraint::Length(8),  // Controls
            ])
            .split(frame.area());
        
        // Title
        let title = Paragraph::new(vec![
            Line::from(vec![
                Span::styled("🧠 dLinOSS Brain Dynamics Explorer ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
                Span::styled(format!("| Mode: {:?} ", self.mode), Style::default().fg(Color::Yellow)),
                Span::styled(format!("| Frame: {} ", self.frame_count), Style::default().fg(Color::Gray)),
            ])
        ])
        .block(Block::default().borders(Borders::ALL))
        .alignment(Alignment::Center);
        frame.render_widget(title, main_chunks[0]);
        
        // Main visualization area
        let viz_chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(85), Constraint::Percentage(15)])
            .split(main_chunks[1]);
        
        // Render visualization based on mode
        match self.mode {
            VisualizationMode::Oscillators => self.render_oscillators(frame, viz_chunks[0]),
            VisualizationMode::Network => self.render_network(frame, viz_chunks[0]),
            VisualizationMode::BrainWaves => self.render_brain_waves(frame, viz_chunks[0]),
            VisualizationMode::Synchronization => self.render_synchronization(frame, viz_chunks[0]),
        }
        
        // Parameter panel
        self.render_parameters(frame, viz_chunks[1]);
        
        // Controls
        self.render_controls(frame, main_chunks[2]);
        
        // Help overlay
        if self.show_help {
            self.render_help(frame);
        }
    }
    
    fn render_oscillators(&mut self, frame: &mut Frame, area: Rect) {
        // Get data before entering the closure
        let trails = self.visualizer.get_trails();
        let points = self.visualizer.step();
        
        let canvas = Canvas::default()
            .block(Block::default()
                .title("dLinOSS Oscillator Dynamics")
                .borders(Borders::ALL)
                .style(Style::default().fg(Color::Blue)))
            .x_bounds([0.0, self.visualizer.config.canvas_width])
            .y_bounds([0.0, self.visualizer.config.canvas_height])
            .paint(|ctx| {
                // Draw trails
                for trail in &trails {
                    for (x, y, color) in trail {
                        ctx.draw(&ratatui::widgets::canvas::Circle {
                            x: *x, y: *y, radius: 0.3, color: *color,
                        });
                    }
                }
                
                // Draw current positions
                for (x, y, color) in &points {
                    ctx.draw(&ratatui::widgets::canvas::Circle {
                        x: *x, y: *y, radius: 1.5, color: *color,
                    });
                }
            });
        
        frame.render_widget(canvas, area);
    }
    
    fn render_network(&mut self, frame: &mut Frame, area: Rect) {
        // TODO: Implement network connectivity visualization
        let canvas = Canvas::default()
            .block(Block::default()
                .title("Neural Network Topology")
                .borders(Borders::ALL)
                .style(Style::default().fg(Color::Green)))
            .x_bounds([0.0, self.visualizer.config.canvas_width])
            .y_bounds([0.0, self.visualizer.config.canvas_height])
            .paint(|_ctx| {
                // Network visualization implementation
            });
        
        frame.render_widget(canvas, area);
    }
    
    fn render_brain_waves(&mut self, frame: &mut Frame, area: Rect) {
        // TODO: Implement EEG-like wave visualization
        let canvas = Canvas::default()
            .block(Block::default()
                .title("Brain Wave Patterns")
                .borders(Borders::ALL)
                .style(Style::default().fg(Color::Magenta)))
            .x_bounds([0.0, self.visualizer.config.canvas_width])
            .y_bounds([0.0, self.visualizer.config.canvas_height])
            .paint(|_ctx| {
                // Brain wave visualization implementation
            });
        
        frame.render_widget(canvas, area);
    }
    
    fn render_synchronization(&mut self, frame: &mut Frame, area: Rect) {
        // TODO: Implement phase synchronization visualization
        let canvas = Canvas::default()
            .block(Block::default()
                .title("Phase Synchronization")
                .borders(Borders::ALL)
                .style(Style::default().fg(Color::Red)))
            .x_bounds([0.0, self.visualizer.config.canvas_width])
            .y_bounds([0.0, self.visualizer.config.canvas_height])
            .paint(|_ctx| {
                // Synchronization visualization implementation
            });
        
        frame.render_widget(canvas, area);
    }
    
    fn render_parameters(&self, frame: &mut Frame, area: Rect) {
        let param_chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),
                Constraint::Length(3),
                Constraint::Length(3),
                Constraint::Min(1),
            ])
            .split(area);
        
        // Damping gauge
        let damping_gauge = Gauge::default()
            .block(Block::default().title("Damping").borders(Borders::ALL))
            .gauge_style(Style::default().fg(Color::Yellow))
            .ratio(self.damping / 0.5);
        frame.render_widget(damping_gauge, param_chunks[0]);
        
        // Frequency gauge
        let freq_gauge = Gauge::default()
            .block(Block::default().title("Frequency").borders(Borders::ALL))
            .gauge_style(Style::default().fg(Color::Cyan))
            .ratio(self.frequency / 5.0);
        frame.render_widget(freq_gauge, param_chunks[1]);
        
        // Complexity gauge
        let complexity_gauge = Gauge::default()
            .block(Block::default().title("Complexity").borders(Borders::ALL))
            .gauge_style(Style::default().fg(Color::Green))
            .ratio(self.complexity as f64 / 20.0);
        frame.render_widget(complexity_gauge, param_chunks[2]);
    }
    
    fn render_controls(&self, frame: &mut Frame, area: Rect) {
        let controls = Paragraph::new(vec![
            Line::from("🎛️  Real-time Controls:"),
            Line::from("↑↓ Damping  ←→ Frequency  +/- Complexity"),
            Line::from("1-4: Modes  R: Reset  H: Help  Q: Quit"),
        ])
        .block(Block::default()
            .title("Interactive Controls")
            .borders(Borders::ALL)
            .style(Style::default().fg(Color::White)));
        
        frame.render_widget(controls, area);
    }
    
    fn render_help(&self, frame: &mut Frame) {
        let help_area = Rect {
            x: frame.area().width / 4,
            y: frame.area().height / 4,
            width: frame.area().width / 2,
            height: frame.area().height / 2,
        };
        
        frame.render_widget(Clear, help_area);
        
        let help_text = Paragraph::new(vec![
            Line::from("🧠 dLinOSS Brain Dynamics Help"),
            Line::from(""),
            Line::from("Visualization Modes:"),
            Line::from("  1 - Oscillator Dynamics"),
            Line::from("  2 - Neural Network"),
            Line::from("  3 - Brain Waves"),
            Line::from("  4 - Synchronization"),
            Line::from(""),
            Line::from("Parameters:"),
            Line::from("  Damping: Energy dissipation"),
            Line::from("  Frequency: Oscillation speed"),
            Line::from("  Complexity: Number of oscillators"),
            Line::from(""),
            Line::from("Press H again to close help"),
        ])
        .block(Block::default()
            .title("Help")
            .borders(Borders::ALL)
            .style(Style::default().fg(Color::White).bg(Color::Black)));
        
        frame.render_widget(help_text, help_area);
    }
}

/// Run the interactive brain dynamics explorer
pub fn run_brain_dynamics_explorer() -> Result<(), Box<dyn std::error::Error>> {
    // Create a shared flag for graceful shutdown
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    
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
    
    // Create cleanup function
    let cleanup = || {
        let _ = disable_raw_mode();
        let _ = execute!(std::io::stderr(), LeaveAlternateScreen);
        let _ = execute!(std::io::stderr(), cursor::Show);
        // Additional cleanup - restore cursor and clear screen
        print!("\x1b[?25h\x1b[2J\x1b[H"); // Show cursor, clear screen, go to home
        let _ = std::io::stdout().flush();
    };
    
    let result = (|| -> Result<(), Box<dyn std::error::Error>> {
        let mut explorer = BrainDynamicsExplorer::new()?;
        let mut last_time = Instant::now();
        
        while running.load(Ordering::SeqCst) {
            // Handle input (non-blocking)
            if event::poll(Duration::from_millis(16))? {
                if let Event::Key(key) = event::read()? {
                    if !explorer.handle_input(key.code) {
                        break;
                    }
                }
            }
            
            // Render at ~60 FPS
            if last_time.elapsed() >= Duration::from_millis(16) {
                terminal.draw(|frame| {
                    explorer.render(frame);
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
    
    // Always cleanup, regardless of how we exit
    cleanup();
    
    result
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 Interactive dLinOSS Brain Dynamics Explorer");
    println!("==============================================");
    println!("Real-time parameter control and multiple visualization modes");
    println!("Demonstrating dLinOSS as universal building blocks for brain-like dynamics");
    println!("");
    println!("Starting interactive explorer...");
    
    run_brain_dynamics_explorer()?;
    
    println!("Thank you for exploring dLinOSS brain dynamics! 🌟");
    
    Ok(())
}
