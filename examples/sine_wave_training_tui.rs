#![recursion_limit = "256"]

// Sine wave training with visualization for LinOSS
// This demonstrates training with real-time TUI visualization of progress

use std::f32::consts::PI;
use std::time::{Duration, Instant};
use std::collections::VecDeque;
use std::error::Error;
use std::io::Stdout;

// Burn imports
use burn::{
    module::AutodiffModule,
    tensor::{TensorData, Tensor},
    optim::{AdamConfig, Optimizer, GradientsParams},
};

// TUI imports
use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Style},
    symbols,
    widgets::{
        Axis, Block, Borders, Chart, Dataset, Paragraph, Wrap,
    },
    Frame, Terminal,
};

// LinOSS imports
use linoss_rust::linoss::{
    model::{FullLinossModel, FullLinossModelConfig}, 
    block::LinossBlockConfig
};

// Type aliases for convenience based on selected backend
#[cfg(all(feature = "ndarray_backend", not(feature = "wgpu_backend")))]
type MyBackend = burn::backend::ndarray::NdArray<f32>;
#[cfg(all(feature = "ndarray_backend", not(feature = "wgpu_backend")))]
type MyAutodiffBackend = burn::backend::Autodiff<burn::backend::ndarray::NdArray<f32>>;

#[cfg(all(feature = "wgpu_backend", not(feature = "ndarray_backend")))]
type MyBackend = burn::backend::wgpu::Wgpu;
#[cfg(all(feature = "wgpu_backend", not(feature = "ndarray_backend")))]
type MyAutodiffBackend = burn::backend::Autodiff<burn::backend::wgpu::Wgpu>;

#[cfg(all(feature = "wgpu_backend", feature = "ndarray_backend"))]
type MyBackend = burn::backend::wgpu::Wgpu;
#[cfg(all(feature = "wgpu_backend", feature = "ndarray_backend"))]
type MyAutodiffBackend = burn::backend::Autodiff<burn::backend::wgpu::Wgpu>;

#[cfg(not(any(feature = "ndarray_backend", feature = "wgpu_backend")))]
type MyBackend = burn::backend::wgpu::Wgpu;
#[cfg(not(any(feature = "ndarray_backend", feature = "wgpu_backend")))]
type MyAutodiffBackend = burn::backend::Autodiff<burn::backend::wgpu::Wgpu>;

// Constants
const D_INPUT: usize = 1;
const D_MODEL: usize = 32;
const D_OUTPUT: usize = 1;
const N_LAYERS: usize = 2;
const BATCH_SIZE: usize = 4;
const SEQ_LEN: usize = 20;
const NUM_EPOCHS: usize = 200;
const LEARNING_RATE: f64 = 0.001;
const MAX_LOSS_HISTORY: usize = 100;

// Training state and visualization data
#[derive(Debug)]
struct TrainingState {
    epoch: usize,
    loss_history: VecDeque<f64>,
    current_loss: f64,
    best_loss: f64,
    test_input: Vec<f32>,
    test_target: Vec<f32>,
    test_prediction: Vec<f32>,
    training_start_time: Instant,
    paused: bool,
    status_message: String,
    spinner_index: usize,
    update_counter: u64,
    show_help: bool,
}

impl TrainingState {
    fn new() -> Self {
        Self {
            epoch: 0,
            loss_history: VecDeque::with_capacity(MAX_LOSS_HISTORY),
            current_loss: f64::INFINITY,
            best_loss: f64::INFINITY,
            test_input: Vec::new(),
            test_target: Vec::new(),
            test_prediction: Vec::new(),
            training_start_time: Instant::now(),
            paused: false,
            status_message: "Initializing...".to_string(),
            spinner_index: 0,
            update_counter: 0,
            show_help: false,
        }
    }

    fn add_loss(&mut self, loss: f64) {
        self.current_loss = loss;
        if loss < self.best_loss {
            self.best_loss = loss;
        }
        
        if self.loss_history.len() >= MAX_LOSS_HISTORY {
            self.loss_history.pop_front();
        }
        self.loss_history.push_back(loss);
    }

    fn update_test_data(&mut self, input: Vec<f32>, target: Vec<f32>, prediction: Vec<f32>) {
        self.test_input = input;
        self.test_target = target;
        self.test_prediction = prediction;
    }

    fn update_spinner(&mut self) {
        self.update_counter += 1;
        self.spinner_index = (self.spinner_index + 1) % 8;
    }

    fn get_spinner_char(&self) -> &'static str {
        if self.paused {
            "⏸"
        } else {
            match self.spinner_index {
                0 => "⠋",
                1 => "⠙",
                2 => "⠹",
                3 => "⠸",
                4 => "⠼",
                5 => "⠴",
                6 => "⠦",
                7 => "⠧",
                _ => "⠋",
            }
        }
    }

    fn get_activity_indicator(&self) -> String {
        if self.paused {
            " [PAUSED]".to_string()
        } else {
            format!(" {} [TRAINING] Updates: {}", self.get_spinner_char(), self.update_counter)
        }
    }

    fn toggle_help(&mut self) {
        self.show_help = !self.show_help;
    }
}

// Data generation functions
fn generate_sine_wave_sequence(length: usize, time_step: f32, phase: f32, frequency: f32) -> Vec<f32> {
    (0..length)
        .map(|i| (frequency * 2.0 * PI * (i as f32 * time_step + phase)).sin())
        .collect()
}

fn generate_training_batch(
    batch_size: usize, 
    seq_len: usize, 
    device: &<MyAutodiffBackend as burn::tensor::backend::Backend>::Device
) -> (Tensor<MyAutodiffBackend, 3>, Tensor<MyAutodiffBackend, 3>) {
    let mut inputs = Vec::new();
    let mut targets = Vec::new();
    
    for _ in 0..batch_size {
        let phase = (rand::random::<f32>() - 0.5) * 2.0 * PI;
        let frequency = 0.5 + rand::random::<f32>() * 1.5;
        
        let full_sequence = generate_sine_wave_sequence(seq_len + 1, 0.1, phase, frequency);
        let input_seq: Vec<f32> = full_sequence[0..seq_len].to_vec();
        let target_seq: Vec<f32> = full_sequence[1..seq_len + 1].to_vec();
        
        inputs.extend(input_seq);
        targets.extend(target_seq);
    }
    
    let input_tensor = Tensor::<MyAutodiffBackend, 3>::from_data(
        TensorData::new(inputs, [batch_size, seq_len, D_INPUT]), device
    );
    
    let target_tensor = Tensor::<MyAutodiffBackend, 3>::from_data(
        TensorData::new(targets, [batch_size, seq_len, D_OUTPUT]), device
    );
    
    (input_tensor, target_tensor)
}

fn generate_test_predictions(
    model: &FullLinossModel<MyBackend>, 
    device: &<MyBackend as burn::tensor::backend::Backend>::Device
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let test_sine_data = generate_sine_wave_sequence(SEQ_LEN + 5, 0.1, 0.0, 1.0);
    let test_input: Vec<f32> = test_sine_data[0..SEQ_LEN].to_vec();
    let test_target: Vec<f32> = test_sine_data[1..SEQ_LEN + 1].to_vec();

    let test_input_tensor = Tensor::<MyBackend, 3>::from_data(
        TensorData::new(test_input.clone(), [1, SEQ_LEN, D_INPUT]), device
    );

    let test_prediction = model.forward(test_input_tensor);
    let pred_data = test_prediction.into_data().into_vec().unwrap();

    (test_input, test_target, pred_data)
}

// TUI setup functions
fn tui_app_startup() -> Result<Terminal<CrosstermBackend<Stdout>>, Box<dyn Error>> {
    enable_raw_mode()?;
    let mut stdout = std::io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let terminal = Terminal::new(backend)?;
    Ok(terminal)
}

fn tui_app_shutdown(mut terminal: Terminal<CrosstermBackend<Stdout>>) -> Result<(), Box<dyn Error>> {
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    Ok(())
}

// UI rendering functions
fn render_ui(frame: &mut Frame, state: &TrainingState) {
    if state.show_help {
        render_help_screen(frame, state);
    } else {
        render_main_screen(frame, state);
    }
}

fn render_main_screen(frame: &mut Frame, state: &TrainingState) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // Status bar
            Constraint::Min(8),     // Main content
            Constraint::Length(3),  // Controls
        ])
        .split(frame.area());

    // Status bar
    let status_text = format!(
        "LinOSS Training{} | Epoch: {}/{} | Loss: {:.6} | Best: {:.6} | Time: {:.1}s",
        state.get_activity_indicator(),
        state.epoch, NUM_EPOCHS, state.current_loss, state.best_loss,
        state.training_start_time.elapsed().as_secs_f32()
    );
    let status = Paragraph::new(status_text)
        .block(Block::default().borders(Borders::ALL).title("Training Status"))
        .style(Style::default().fg(Color::Yellow));
    frame.render_widget(status, chunks[0]);

    // Main content area
    let main_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(chunks[1]);

    // Loss chart
    render_loss_chart(frame, state, main_chunks[0]);
    
    // Sine wave comparison
    render_sine_wave_chart(frame, state, main_chunks[1]);

    // Controls
    let controls_text = if state.paused {
        "Controls: [Space] Resume | [Q] Quit | [H] Help | Status: ⏸ PAUSED".to_string()
    } else {
        format!("Controls: [Space] Pause | [Q] Quit | [H] Help | Status: {} TRAINING", state.get_spinner_char())
    };
    let controls = Paragraph::new(controls_text)
        .block(Block::default().borders(Borders::ALL).title("Controls"))
        .style(Style::default().fg(Color::Green));
    frame.render_widget(controls, chunks[2]);
}

fn render_loss_chart(frame: &mut Frame, state: &TrainingState, area: Rect) {
    if state.loss_history.is_empty() {
        let empty = Paragraph::new("Waiting for training data...")
            .block(Block::default().borders(Borders::ALL).title("Loss History"));
        frame.render_widget(empty, area);
        return;
    }

    let loss_data: Vec<(f64, f64)> = state.loss_history
        .iter()
        .enumerate()
        .map(|(i, &loss)| (i as f64, loss))
        .collect();

    let min_loss = state.loss_history.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_loss = state.loss_history.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let loss_range = (max_loss - min_loss).max(0.001);

    let datasets = vec![Dataset::default()
        .name("Training Loss")
        .marker(symbols::Marker::Braille)
        .style(Style::default().fg(Color::Red))
        .data(&loss_data)];

    let chart = Chart::new(datasets)
        .block(Block::default().borders(Borders::ALL).title("Loss History"))
        .x_axis(
            Axis::default()
                .title("Iteration")
                .style(Style::default().fg(Color::Gray))
                .bounds([0.0, MAX_LOSS_HISTORY as f64])
        )
        .y_axis(
            Axis::default()
                .title("Loss")
                .style(Style::default().fg(Color::Gray))
                .bounds([min_loss - loss_range * 0.1, max_loss + loss_range * 0.1])
        )
        .legend_position(Some(ratatui::widgets::LegendPosition::TopRight));

    frame.render_widget(chart, area);
}

fn render_sine_wave_chart(frame: &mut Frame, state: &TrainingState, area: Rect) {
    if state.test_input.is_empty() {
        let empty = Paragraph::new("Waiting for test predictions...")
            .block(Block::default().borders(Borders::ALL).title("Sine Wave Prediction"));
        frame.render_widget(empty, area);
        return;
    }

    let target_data: Vec<(f64, f64)> = state.test_target
        .iter()
        .enumerate()
        .map(|(i, &val)| (i as f64, val as f64))
        .collect();

    let prediction_data: Vec<(f64, f64)> = state.test_prediction
        .iter()
        .enumerate()
        .map(|(i, &val)| (i as f64, val as f64))
        .collect();

    let datasets = vec![
        Dataset::default()
            .name("Target")
            .marker(symbols::Marker::Dot)
            .style(Style::default().fg(Color::Blue))
            .data(&target_data),
        Dataset::default()
            .name("Prediction")
            .marker(symbols::Marker::Braille)
            .style(Style::default().fg(Color::Green))
            .data(&prediction_data),
    ];

    let chart = Chart::new(datasets)
        .block(Block::default().borders(Borders::ALL).title("Sine Wave: Target vs Prediction"))
        .x_axis(
            Axis::default()
                .title("Time Step")
                .style(Style::default().fg(Color::Gray))
                .bounds([0.0, SEQ_LEN as f64])
        )
        .y_axis(
            Axis::default()
                .title("Value")
                .style(Style::default().fg(Color::Gray))
                .bounds([-1.5, 1.5])
        )
        .legend_position(Some(ratatui::widgets::LegendPosition::TopLeft));

    frame.render_widget(chart, area);
}

fn render_help_screen(frame: &mut Frame, _state: &TrainingState) {
    let help_text = vec![
        "🧠 DEEP LEARNING CONCEPTS EXPLAINED",
        "",
        "🎯 WHAT IS HAPPENING:",
        "• You're training a LinOSS (Linear Oscillatory State Space) neural network",
        "• The model learns to predict future values of a sine wave pattern",
        "• This is a time series prediction task using sequence-to-sequence learning",
        "",
        "📊 TRAINING PROCESS:",
        "• EPOCH: One complete pass through the training data",
        "• LOSS: Measures how wrong the model's predictions are (lower = better)",
        "• BACKPROPAGATION: The model adjusts its internal parameters after each mistake",
        "• OPTIMIZATION: Adam optimizer gradually improves the model's weights",
        "",
        "🔍 WHAT THE CHARTS SHOW:",
        "• LEFT CHART (Loss History):",
        "  - Red line shows training error over time",
        "  - Should generally trend downward as the model learns",
        "  - Flat lines mean the model has converged (stopped learning)",
        "",
        "• RIGHT CHART (Predictions):",
        "  - Blue dots: Ground truth (what the model should predict)",
        "  - Green line: Model's actual predictions",
        "  - As training progresses, green should match blue better",
        "",
        "🧮 MODEL ARCHITECTURE:",
        "• LinOSS combines linear dynamics with neural networks",
        "• Input dimension: 1 (single sine wave value)",
        "• Hidden dimension: 32 (internal representation size)",
        "• Output dimension: 1 (predicted next value)",
        "• Layers: 2 (depth of the neural network)",
        "",
        "⚡ STATE SPACE MODELS:",
        "• LinOSS is based on state space models from control theory",
        "• It maintains an internal 'state' that evolves over time",
        "• This helps capture long-term dependencies in sequences",
        "• More efficient than traditional RNNs or Transformers for long sequences",
        "",
        "🎮 CONTROLS:",
        "• [Space]: Pause/Resume training to examine current state",
        "• [H]: Toggle this help screen",
        "• [Q]: Quit the application",
        "",
        "💡 LEARNING INDICATORS:",
        "• Spinning animation shows active training",
        "• Update counter shows UI refresh rate",
        "• Best loss tracks the lowest error achieved",
        "",
        "Press [H] to return to training view"
    ];

    let help_content = help_text.join("\n");
    let help_paragraph = Paragraph::new(help_content)
        .block(Block::default().borders(Borders::ALL).title("🎓 Deep Learning Help"))
        .style(Style::default().fg(Color::Cyan))
        .wrap(Wrap { trim: true });
    
    frame.render_widget(help_paragraph, frame.area());
}

// Main training function with TUI
fn run_training_with_tui(
    mut terminal: Terminal<CrosstermBackend<Stdout>>
) -> Result<(), Box<dyn Error>> {
    let device = Default::default();
    let mut state = TrainingState::new();
    
    // Show initialization status
    state.status_message = "🔧 Initializing device and model...".to_string();
    terminal.draw(|frame| {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(1)])
            .split(frame.area());
        let status = Paragraph::new(state.status_message.clone())
            .block(Block::default().borders(Borders::ALL).title("Initialization"))
            .style(Style::default().fg(Color::Yellow));
        frame.render_widget(status, chunks[0]);
    })?;

    // Create model
    state.status_message = "🏗️  Building LinOSS model architecture...".to_string();
    terminal.draw(|frame| {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(1)])
            .split(frame.area());
        let status = Paragraph::new(state.status_message.clone())
            .block(Block::default().borders(Borders::ALL).title("Initialization"))
            .style(Style::default().fg(Color::Yellow));
        frame.render_widget(status, chunks[0]);
    })?;

    let block_config = LinossBlockConfig {
        d_state_m: D_MODEL / 2,
        d_ff: D_MODEL * 2,
        delta_t: 0.1,
        init_std: 0.02,
        enable_d_feedthrough: true,
    };

    let model_config = FullLinossModelConfig {
        d_input: D_INPUT,
        d_model: D_MODEL,
        d_output: D_OUTPUT,
        n_layers: N_LAYERS,
        linoss_block_config: block_config,
    };

    let mut model = model_config.init::<MyAutodiffBackend>(&device);
    
    state.status_message = "🎯 Setting up Adam optimizer...".to_string();
    terminal.draw(|frame| {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(1)])
            .split(frame.area());
        let status = Paragraph::new(state.status_message.clone())
            .block(Block::default().borders(Borders::ALL).title("Initialization"))
            .style(Style::default().fg(Color::Yellow));
        frame.render_widget(status, chunks[0]);
    })?;
    
    let optimizer_config = AdamConfig::new();
    let mut optimizer = optimizer_config.init();

    state.status_message = "✅ Ready! Starting training...".to_string();
    terminal.draw(|frame| {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(1)])
            .split(frame.area());
        let status = Paragraph::new(state.status_message.clone())
            .block(Block::default().borders(Borders::ALL).title("Initialization"))
            .style(Style::default().fg(Color::Green));
        frame.render_widget(status, chunks[0]);
    })?;
    
    // Give user a moment to see the ready message
    std::thread::sleep(std::time::Duration::from_millis(1000));

    // Training loop with TUI
    let mut last_update = Instant::now();
    
    for epoch in 0..NUM_EPOCHS {
        state.epoch = epoch;
        
        if !state.paused {
            let mut epoch_loss = 0.0f32;
            let batches_per_epoch = 5;
            
            for _batch_idx in 0..batches_per_epoch {
                let (input_batch, target_batch) = generate_training_batch(BATCH_SIZE, SEQ_LEN, &device);
                let prediction = model.forward(input_batch);
                let loss = (prediction - target_batch).powf_scalar(2.0).mean();
                let loss_value = loss.clone().into_scalar();
                epoch_loss += loss_value;
                
                let grads = loss.backward();
                let grads = GradientsParams::from_grads(grads, &model);
                model = optimizer.step(LEARNING_RATE, model, grads);
            }
            
            let avg_loss = epoch_loss / batches_per_epoch as f32;
            state.add_loss(avg_loss as f64);

            // Update test predictions every 5 epochs
            if epoch % 5 == 0 {
                let test_model = model.clone().valid();
                let (input, target, prediction) = generate_test_predictions(&test_model, &device);
                state.update_test_data(input, target, prediction);
            }
        }

        // Handle events and update UI
        if last_update.elapsed() >= Duration::from_millis(50) {
            state.update_spinner();
            terminal.draw(|frame| render_ui(frame, &state))?;
            last_update = Instant::now();
            
            // Handle keyboard input
            if event::poll(Duration::from_millis(1))? {
                if let Event::Key(key) = event::read()? {
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Char('Q') => {
                            break;
                        }
                        KeyCode::Char(' ') => {
                            state.paused = !state.paused;
                            state.status_message = if state.paused {
                                "Training paused".to_string()
                            } else {
                                "Training resumed".to_string()
                            };
                        }
                        KeyCode::Char('h') | KeyCode::Char('H') => {
                            state.toggle_help();
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    state.status_message = "Training completed!".to_string();
    terminal.draw(|frame| render_ui(frame, &state))?;
    
    // Wait for user to exit
    loop {
        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                if matches!(key.code, KeyCode::Char('q') | KeyCode::Char('Q')) {
                    break;
                }
            }
        }
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("🚀 Starting LinOSS Sine Wave Training with Visualization...");
    println!("📋 Initializing model configuration...");
    
    // Show initialization progress
    println!("⚙️  Setting up LinOSS model (d_model={}, layers={})...", D_MODEL, N_LAYERS);
    println!("🎯 Training parameters: {} epochs, batch_size={}, seq_len={}", NUM_EPOCHS, BATCH_SIZE, SEQ_LEN);
    println!("📊 Preparing real-time visualization with ratatui...");
    println!("🖥️  Switching to TUI mode in 2 seconds...");
    
    // Give user a moment to read the setup info
    std::thread::sleep(std::time::Duration::from_secs(2));
    
    println!("🎮 Controls: [Space] Pause/Resume | [Q] Quit");
    println!("🔄 Launching interactive training interface...");
    
    let terminal = tui_app_startup()?;
    let result = run_training_with_tui(terminal);
    
    // Cleanup regardless of result
    let cleanup_terminal = tui_app_startup()?; // Get a new terminal for cleanup
    if let Err(e) = tui_app_shutdown(cleanup_terminal) {
        eprintln!("Error during TUI cleanup: {}", e);
    }
    
    match result {
        Ok(_) => {
            println!("Training completed successfully!");
            Ok(())
        }
        Err(e) => {
            eprintln!("Training error: {}", e);
            Err(e)
        }
    }
}
