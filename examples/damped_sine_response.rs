//! Damped Sine Response Training with LinOSS/dLinOSS Toggle
//!
//! This example demonstrates training a neural model to learn the impulse response
//! of a damped sine wave system (like a drum kick or resonant system).
//!
//! ## Model Toggle Feature:
//! - **LinOSS Model**: Standard 2-layer FullLinossModel (32D state, 16D hidden)
//! - **dLinOSS Model**: Dissipative LinOSS with GELU A-matrix parameterization and damping
//!
//! ## Controls:
//! - **[M]**: Toggle between LinOSS and dLinOSS models (resets training)
//! - **[H]**: Toggle help screen with detailed information
//! - **[Q]**: Quit the application
//!
//! ## Training Task:
//! - **Input**: Impulse function (1.0 at t=0, 0.0 elsewhere)
//! - **Target**: Damped sine wave: e^(-0.5*t) * sin(5*t)
//! - **Objective**: Learn the impulse→damped_sine mapping
//!
//! The visualization shows:
//! 1. **Top Panel**: Input impulse (yellow), target damped sine (green), model output (blue)
//! 2. **Middle Panel**: Training loss over epochs
//! 3. **Bottom Panel**: Status with current model type and controls

use burn::{
    optim::{AdamConfig, Optimizer, GradientsParams},
    tensor::{
        backend::AutodiffBackend,
        ElementConversion, Tensor, TensorData, // Removed Element
    },
};
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use linoss_rust::linoss::{
    block::LinossBlockConfig, // Corrected LinossBlockConfig
    model::{FullLinossModel, FullLinossModelConfig}, // Corrected FullLinossModel and FullLinossModelConfig
    dlinoss_layer::{AParameterization, DLinossLayer, DLinossLayerConfig}, // Added dLinOSS support
};
use ratatui::{
    backend::{Backend as RatatuiBackend, CrosstermBackend},
    layout::{Constraint, Direction, Layout},
    style::{Color, Style},
    symbols,
    widgets::{Axis, Block, Borders, Chart, Dataset, GraphType, Paragraph},
    Frame, Terminal,
};
use std::{
    error::Error,
    io::{self, Write},
    fs::OpenOptions,
    time::{Duration, Instant},
};

const D_MODEL: usize = 32;  // Increased from 16
const D_STATE_M: usize = 64; // Increased from 32
const N_LAYERS: usize = 3;   // Increased from 2
const SEQ_LEN: usize = 100;
const BATCH_SIZE: usize = 1;
const LEARNING_RATE: f64 = 1e-3; // Reduced from 5e-3
const N_EPOCHS: usize = 200;     // Doubled

// Define the backend type alias
type MyBackend = burn::backend::NdArray<f32>; // Or Wgpu<f32, i32>
type MyAutodiffBackend = burn::backend::Autodiff<MyBackend>; // Corrected path for Autodiff

/// Simple dLinOSS model for impulse response learning
#[derive(burn::module::Module, Debug)]
pub struct DLinossImpulseModel<B: burn::tensor::backend::Backend> {
    input_linear: burn::nn::Linear<B>,
    dlinoss_core: DLinossLayer<B>,
    output_linear: burn::nn::Linear<B>,
}

impl<B: burn::tensor::backend::Backend> DLinossImpulseModel<B> {
    pub fn new(device: &B::Device) -> Self {
        let config = DLinossLayerConfig {
            d_input: D_MODEL,
            d_model: D_MODEL,
            d_output: D_MODEL,
            delta_t: 0.1,
            init_std: 0.02,
            enable_layer_norm: true,
            enable_damping: true,
            init_damping: 0.1,
            num_damping_scales: 2,
            a_parameterization: AParameterization::GELU, // Smooth activation for damped oscillations
        };
        
        Self {
            input_linear: burn::nn::LinearConfig::new(1, D_MODEL).init(device),
            dlinoss_core: DLinossLayer::new(&config, device),
            output_linear: burn::nn::LinearConfig::new(D_MODEL, 1).init(device),
        }
    }
    
    pub fn forward(&self, input: burn::tensor::Tensor<B, 3>) -> burn::tensor::Tensor<B, 3> {
        // input: [batch, seq_len, 1]
        let processed = self.input_linear.forward(input);
        let dlinoss_out = self.dlinoss_core.forward(processed);
        let output = self.output_linear.forward(dlinoss_out);
        
        // Aggressive scaling to match target amplitude
        // Target range is ~0.74, output range is ~0.09, so scale by ~8x
        output * 8.0
    }
}

/// Enum to represent the two model types
#[derive(Debug, Clone, Copy)]
pub enum ModelType {
    LinOSS,
    DLinOSS,
}

/// Wrapper enum for the two model types
#[derive(Debug, Clone)]
pub enum ModelWrapper<AB: AutodiffBackend> {
    LinOSS(FullLinossModel<AB>),
    DLinOSS(DLinossImpulseModel<AB>),
}

impl<AB: AutodiffBackend> ModelWrapper<AB> 
where
    AB::FloatElem: From<f32> + std::ops::Mul<Output = AB::FloatElem> + Copy + ElementConversion + num_traits::FloatConst,
{
    pub fn forward(&self, input: burn::tensor::Tensor<AB, 3>) -> burn::tensor::Tensor<AB, 3> {
        match self {
            ModelWrapper::LinOSS(model) => model.forward(input),
            ModelWrapper::DLinOSS(model) => model.forward(input),
        }
    }
}

// App state for the visualization
struct DampedSineApp<AB: AutodiffBackend>
where
    AB::FloatElem: From<f32> + std::ops::Mul<Output = AB::FloatElem> + Copy + ElementConversion + num_traits::FloatConst,
{
    input_data: Vec<(f64, f64)>,      // (time, value) for step function input
    output_data: Vec<(f64, f64)>,     // (time, value) for model output
    target_data: Vec<(f64, f64)>,     // (time, value) for target damped sine wave
    xy_scatter_data: Vec<(f64, f64)>, // (input_val, output_val) for scatter plot
    linoss_model: FullLinossModel<AB>,
    dlinoss_model: DLinossImpulseModel<AB>,
    use_dlinoss: bool,                // Toggle between models
    current_epoch: usize,
    total_epochs: usize,
    loss_history: Vec<(f64, f64)>,
    should_quit: bool,
    show_help: bool,                  // Toggle help screen
}

impl<AB: AutodiffBackend> DampedSineApp<AB>
where
    AB::FloatElem: From<f32> + std::ops::Mul<Output = AB::FloatElem> + Copy + ElementConversion + num_traits::FloatConst,
{
    fn new(linoss_model: FullLinossModel<AB>, dlinoss_model: DLinossImpulseModel<AB>) -> Self {
        // Time points
        let time_points: Vec<f64> = (0..SEQ_LEN).map(|i| i as f64 * 0.1).collect();
        
        // Step function input: 1.0 at t=0, 0.0 elsewhere
        // For a damped sine response, we usually excite with an impulse or step.
        // Let's use an impulse at t=0 for simplicity in this example.
        let mut step_input = vec![0.0; SEQ_LEN];
        step_input[0] = 1.0; // The impulse/step at the beginning
        
        // Create damped sine wave target: e^(-at) * sin(bt)
        // These parameters define the shape of the desired output.
        let damping_factor = 0.5; // Controls how quickly the oscillation decays
        let frequency = 5.0;      // Controls the frequency of oscillation
        
        let target_sine: Vec<f64> = time_points.iter()
            .map(|t| {
                if *t <= 0.0 {
                    0.0 // Before t=0, output is 0
                } else {
                    // Original damped sine with proper scaling
                    let raw_value = (-damping_factor * t).exp() * (frequency * t).sin();
                    raw_value * 0.5  // Scale to reasonable range
                }
            })
            .collect();
        
        // Pair time points with input/target values
        let input_data = time_points.iter()
            .zip(step_input.iter())
            .map(|(t, s)| (*t, *s))
            .collect();
        
        let target_data = time_points.iter()
            .zip(target_sine.iter())
            .map(|(t, s)| (*t, *s))
            .collect();

        DampedSineApp {
            input_data,
            output_data: vec![(0.0, 0.0); SEQ_LEN], // Initialize with zeros
            target_data,
            xy_scatter_data: vec![(0.0, 0.0); SEQ_LEN], // Initialize with zeros
            linoss_model,
            dlinoss_model,
            use_dlinoss: true, // Start with dLinOSS model instead
            current_epoch: 0,
            total_epochs: N_EPOCHS,
            loss_history: Vec::new(),
            should_quit: false,
            show_help: false, // Start with help hidden
        }
    }

    fn update_model_output(&mut self, backend_device: &AB::Device) {
        // Extract input values
        let input_tensor_data: Vec<f32> = self.input_data.iter()
            .map(|(_, val)| *val as f32)
            .collect();
        
        // Create input tensor
        let input_tensor = Tensor::<AB, 2>::from_data(
            TensorData::new(input_tensor_data, [BATCH_SIZE, SEQ_LEN]), // Removed redundant .convert::<f32>()
            backend_device,
        )
        .reshape([BATCH_SIZE, SEQ_LEN, 1]);

        // Forward pass through the appropriate model
        let output_tensor = if self.use_dlinoss {
            self.dlinoss_model.forward(input_tensor.clone())
        } else {
            self.linoss_model.forward(input_tensor.clone())
        };
        
        // Convert output tensor back to Vec for visualization
        let output_vec: Vec<f32> = output_tensor
            .into_data()
            .convert::<f32>()
            .into_vec()
            .unwrap();

        // Update output data with time/value pairs
        self.output_data = self.input_data.iter()
            .zip(output_vec.iter())
            .map(|((t, _), o_val)| (*t, *o_val as f64))
            .collect();

        // Update scatter plot data (input vs output)
        self.xy_scatter_data = self.input_data.iter()
            .zip(self.output_data.iter())
            .map(|((_,i_val), (_,o_val))| (*i_val, *o_val))
            .collect();
    }
    
    fn train_step<O: Optimizer<FullLinossModel<AB>, AB>, O2: Optimizer<DLinossImpulseModel<AB>, AB>>(&mut self, linoss_optim: &mut O, dlinoss_optim: &mut O2, backend_device: &AB::Device) { 
        // Debug: Log training step start (only every 10 steps to reduce log size)
        if self.current_epoch % 10 == 0 {
            debug_log(&format!("=== TRAIN STEP {} (Model: {}) ===", 
                self.current_epoch, 
                if self.use_dlinoss { "dLinOSS" } else { "LinOSS" }
            ));
        }
        
        // Prepare input tensor
        let input_values: Vec<f32> = self.input_data.iter()
            .map(|(_, val)| *val as f32)
            .collect();
        
        // Debug: Log input statistics (only every 10 steps)
        if self.current_epoch % 10 == 0 {
            let input_sum: f32 = input_values.iter().sum();
            let input_max = input_values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let input_min = input_values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            debug_log(&format!("Input: sum={:.6}, min={:.6}, max={:.6}", input_sum, input_min, input_max));
        }
        
        let input_tensor = Tensor::<AB, 2>::from_data(
            TensorData::new(input_values, [BATCH_SIZE, SEQ_LEN]), // Removed redundant .convert::<f32>()
            backend_device,
        ).reshape([BATCH_SIZE, SEQ_LEN, 1]);

        // Prepare target tensor (damped sine wave)
        let target_values: Vec<f32> = self.target_data.iter()
            .map(|(_, val)| *val as f32)
            .collect();
        
        // Debug: Log target statistics
        let target_sum: f32 = target_values.iter().sum();
        let target_max = target_values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let target_min = target_values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        debug_log(&format!("Target: sum={:.6}, min={:.6}, max={:.6}", target_sum, target_min, target_max));
        
        let target_tensor = Tensor::<AB, 2>::from_data(
            TensorData::new(target_values, [BATCH_SIZE, SEQ_LEN]), // Removed redundant .convert::<f32>()
            backend_device,
        ).reshape([BATCH_SIZE, SEQ_LEN, 1]);

        // Forward pass, loss calculation, and optimization for the active model
        if self.use_dlinoss {
            let model_output = self.dlinoss_model.forward(input_tensor.clone());
            
            let loss = (model_output.clone() - target_tensor.clone()).powf_scalar(2.0).mean();
            let loss_val = loss.clone().into_scalar().elem::<f64>();
            
            // Debug: Log model output statistics
            let output_data: Vec<f32> = model_output.clone().into_data().convert::<f32>().into_vec().unwrap();
            let output_sum: f32 = output_data.iter().sum();
            let output_max = output_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let output_min = output_data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            debug_log(&format!("E{}: Out sum={:.3}, range=[{:.3}, {:.3}], Loss={:.6}", 
                self.current_epoch, output_sum, output_min, output_max, loss_val));
            debug_log(&format!("dLinOSS Loss: {:.8}", loss_val));
            
            // Backward pass and optimization for dLinOSS model
            let grads_raw = loss.backward();
            let grads = GradientsParams::from_grads(grads_raw, &self.dlinoss_model);
            self.dlinoss_model = dlinoss_optim.step(LEARNING_RATE, self.dlinoss_model.clone(), grads);
            
            // Update loss history
            self.loss_history.push((self.current_epoch as f64, loss_val));
        } else {
            let model_output_raw = self.linoss_model.forward(input_tensor.clone());
            // Center and scale LinOSS output to match target distribution
            // LinOSS seems biased positive, so center around 0 and scale appropriately
            let model_output = (model_output_raw - 0.5) * 2.0;
            
            let loss = (model_output.clone() - target_tensor.clone()).powf_scalar(2.0).mean();
            let loss_val = loss.clone().into_scalar().elem::<f64>();
            
            // Debug: Log model output statistics  
            let output_data: Vec<f32> = model_output.clone().into_data().convert::<f32>().into_vec().unwrap();
            let output_sum: f32 = output_data.iter().sum();
            let output_max = output_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let output_min = output_data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            debug_log(&format!("E{}: Out sum={:.3}, range=[{:.3}, {:.3}], Loss={:.6}", 
                self.current_epoch, output_sum, output_min, output_max, loss_val));
            debug_log(&format!("LinOSS Loss: {:.8}", loss_val));
            
            // Backward pass and optimization for LinOSS model
            let grads_raw = loss.backward();
            let grads = GradientsParams::from_grads(grads_raw, &self.linoss_model);
            self.linoss_model = linoss_optim.step(LEARNING_RATE, self.linoss_model.clone(), grads);
            
            // Update loss history
            self.loss_history.push((self.current_epoch as f64, loss_val));
        }
        
        // Update visualizations
        self.update_model_output(backend_device);
        
        if self.loss_history.len() > SEQ_LEN { 
            self.loss_history.remove(0);
        }
        
        // Update epoch counter
        self.current_epoch += 1;
        // Don't auto-quit when training completes - let user quit manually
        // if self.current_epoch >= self.total_epochs {
        //     self.should_quit = true; 
        // }
    }
    
    fn toggle_model(&mut self) {
        self.use_dlinoss = !self.use_dlinoss;
        // Reset training when switching models
        self.current_epoch = 0;
        self.loss_history.clear();
    }
    
    fn toggle_help(&mut self) {
        self.show_help = !self.show_help;
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    // Clear previous debug log
    clear_debug_log();
    debug_log("=== DAMPED SINE RESPONSE TRAINING START ===");
    
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Initialize the backend device
    let device = <MyAutodiffBackend as burn::tensor::backend::Backend>::Device::default();

    // Create LinossBlock configuration
    let linoss_block_config = LinossBlockConfig {
        d_state_m: D_STATE_M,
        d_ff: D_MODEL, // d_ff is often related to d_model
        delta_t: 0.1,  // Time step for the Linoss layer
        init_std: 0.02,
        enable_d_feedthrough: false, // Or true, depending on the model variant
    };

    // Create FullLinossModel configuration
    let model_config = FullLinossModelConfig {
        d_input: 1, // Single input value at each time step
        d_model: D_MODEL,
        d_output: 1, // Single output value at each time step
        n_layers: N_LAYERS,
        linoss_block_config, // Embed the block config
    };

    // Initialize the LinOSS model
    debug_log(&format!("Initializing LinOSS model - D_MODEL: {}, N_LAYERS: {}", D_MODEL, N_LAYERS));
    let linoss_model: FullLinossModel<MyAutodiffBackend> = model_config.init(&device);
    debug_log("LinOSS model initialized");
    
    // Initialize the dLinOSS model
    debug_log("Initializing dLinOSS model");
    let dlinoss_model: DLinossImpulseModel<MyAutodiffBackend> = DLinossImpulseModel::new(&device);
    debug_log("dLinOSS model initialized");

    // Create the app state
    debug_log("Creating app state and computing initial output");
    let mut app = DampedSineApp::new(linoss_model, dlinoss_model);
    app.update_model_output(&device); // Initial output before training
    debug_log("Initial model output computed");

    // Initialize optimizers for both models
    let mut linoss_optim = AdamConfig::new().init();
    let mut dlinoss_optim = AdamConfig::new().init();

    // Run the application
    let res = run_app(&mut terminal, &mut app, &mut linoss_optim, &mut dlinoss_optim, &device);

    // Restore terminal - do this even if there was an error
    let _cleanup_result = (|| -> Result<(), Box<dyn Error>> {
        // Clear the screen first to prevent garbled output
        terminal.clear()?;
        disable_raw_mode()?;
        execute!(
            terminal.backend_mut(),
            LeaveAlternateScreen,
            DisableMouseCapture
        )?;
        terminal.show_cursor()?;
        Ok(())
    })();

    // Clear any remaining input buffer
    while crossterm::event::poll(Duration::from_millis(1))? {
        let _ = crossterm::event::read();
    }

    if let Err(err) = res {
        println!("Application error: {:?}", err);
    }

    println!("Damped Sine Response Learning completed. Thanks for using LinOSS!");

    Ok(())
}

// RB for RatatuiBackend, AB for AutodiffBackend
fn run_app<RB: RatatuiBackend, AB: AutodiffBackend, O1: Optimizer<FullLinossModel<AB>, AB>, O2: Optimizer<DLinossImpulseModel<AB>, AB>>( // Made optimizer generic
    terminal: &mut Terminal<RB>,
    app: &mut DampedSineApp<AB>, 
    linoss_optimizer: &mut O1, // LinOSS optimizer
    dlinoss_optimizer: &mut O2, // dLinOSS optimizer
    backend_device: &AB::Device,
) -> io::Result<()> 
where
    AB::FloatElem: From<f32> + std::ops::Mul<Output = AB::FloatElem> + Copy + ElementConversion + num_traits::FloatConst,
{
    let tick_rate = Duration::from_millis(100); 
    let mut last_tick = Instant::now();

    loop {
        // Check if we should quit before drawing
        if app.should_quit {
            return Ok(());
        }

        terminal.draw(|f| ui(f, app))?;

        let timeout = tick_rate
            .checked_sub(last_tick.elapsed())
            .unwrap_or_else(|| Duration::from_secs(0));

        if crossterm::event::poll(timeout)? {
            if let Event::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Char('q') => {
                        app.should_quit = true;
                        return Ok(()); // Exit immediately when quit is pressed
                    }
                    KeyCode::Char('m') => {
                        app.toggle_model();
                    }
                    KeyCode::Char('h') | KeyCode::Char('H') => {
                        app.toggle_help();
                    }
                    _ => {}
                }
            }
        }

        if last_tick.elapsed() >= tick_rate {
            // Only train if not completed
            if app.current_epoch < app.total_epochs {
                app.train_step(linoss_optimizer, dlinoss_optimizer, backend_device);
            }
            last_tick = Instant::now();
        }
    }
}

// UI rendering function
fn ui<AB: AutodiffBackend>(f: &mut Frame<'_>, app: &DampedSineApp<AB>) 
where
    AB::FloatElem: From<f32> + std::ops::Mul<Output = AB::FloatElem> + Copy + ElementConversion + num_traits::FloatConst,
{
    // Show help screen if toggled
    if app.show_help {
        draw_help_screen(f);
        return;
    }

    // Normal UI when help is not shown
    draw_main_ui(f, app);
}

// Help screen UI
fn draw_help_screen(f: &mut Frame<'_>) {
    let area = f.area();
    
    let help_text = "\
📖 Damped Sine Response Learning - Help Screen

🎯 OBJECTIVE:
   Learn to map an impulse input to a damped sine wave output
   Target Function: e^(-0.5*t) × sin(5*t)

🤖 MODELS:
   • LinOSS (Standard): 2-layer FullLinossModel (32D state, 16D hidden)
   • dLinOSS (Damped): Single-layer with GELU A-parameterization + damping

📊 VISUALIZATION:
   • Top Panel: Target damped sine (green), model output (blue)
   • Middle Panel: Training loss over epochs
   • Bottom Panel: Model info and controls

🎮 CONTROLS:
   • [M] - Toggle between LinOSS and dLinOSS models
   • [H] - Toggle this help screen
   • [Q] - Quit application

💡 USAGE TIPS:
   • Watch how different models learn the same task
   • dLinOSS often learns damped oscillations more naturally
   • LinOSS may require more epochs but can be more flexible
   • Training resets when you switch models for fair comparison

🔬 TECHNICAL DETAILS:
   • Input: Unit impulse at t=0 (δ function approximation)
   • Output: Decaying sinusoid representing system response
   • Loss: Mean Squared Error between model output and target
   • Optimization: Adam optimizer with learning rate 0.005

Press [H] again to return to main view";

    let help_paragraph = Paragraph::new(help_text)
        .style(Style::default().fg(Color::White))
        .block(Block::default()
            .title("📚 Help - Press [H] to return")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Yellow)));
    
    f.render_widget(help_paragraph, area);
}

// Main application UI
fn draw_main_ui<AB: AutodiffBackend>(f: &mut Frame<'_>, app: &DampedSineApp<AB>) 
where
    AB::FloatElem: From<f32> + std::ops::Mul<Output = AB::FloatElem> + Copy + ElementConversion + num_traits::FloatConst,
{
    let area = f.area(); // Changed from size() to area()
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints(
            [
                Constraint::Percentage(40), // Main plot - smaller
                Constraint::Percentage(35), // Loss plot 
                Constraint::Percentage(25), // Status section - bigger
            ]
            .as_ref(),
        )
        .split(area); // Use area here

    // Split the bottom chunk for additional info and status text
    let bottom_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)].as_ref())
        .split(chunks[2]);
    
    // Main output vs target plot
    let main_datasets = vec![
        Dataset::default()
            .name("Target (Damped Sine)")
            .marker(symbols::Marker::Dot)
            .graph_type(GraphType::Scatter)
            .style(Style::default().fg(Color::Green))
            .data(&app.target_data),
        Dataset::default()
            .name("Model Output")
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(Color::Blue))
            .data(&app.output_data),
        // Removed the impulse input (yellow dots) to avoid interference
    ];
    
    let main_chart = Chart::new(main_datasets)
        .block(Block::default().title("Damped Sine Response Learning").borders(Borders::ALL))
        .x_axis(Axis::default()
            .title("Time")
            .style(Style::default().fg(Color::Gray))
            .bounds([0.0, (SEQ_LEN as f64) * 0.1]))
        .y_axis(Axis::default()
            .title("Amplitude")
            .style(Style::default().fg(Color::Gray))
            .bounds([-1.0, 1.0]));
    f.render_widget(main_chart, chunks[0]);
    
    // Loss history chart - now gets the full middle section
    let loss_datasets = vec![Dataset::default()
        .name("Training Loss")
        .marker(symbols::Marker::Braille)
        .graph_type(GraphType::Line)
        .style(Style::default().fg(Color::Red))
        .data(&app.loss_history)];
    
    // Calculate better bounds for loss visualization
    let max_loss = app.loss_history.iter().map(|(_,y)| *y).fold(0.0, f64::max).max(0.1);
    let min_loss = app.loss_history.iter().map(|(_,y)| *y).fold(f64::INFINITY, f64::min).min(max_loss * 0.9);
    
    let loss_chart = Chart::new(loss_datasets)
        .block(Block::default()
            .title(format!("Training Loss - Current: {:.6}", 
                app.loss_history.last().map(|(_, loss)| *loss).unwrap_or(0.0)))
            .borders(Borders::ALL))
        .x_axis(Axis::default()
            .title("Epoch")
            .style(Style::default().fg(Color::Gray))
            .bounds([0.0, app.total_epochs as f64]))
        .y_axis(Axis::default()
            .title("Loss")
            .style(Style::default().fg(Color::Gray))
            .bounds([min_loss.max(0.0), max_loss]));
    f.render_widget(loss_chart, chunks[1]); // Full middle section

    // Model comparison info (left side of bottom)
    let current_loss = app.loss_history.last().map(|(_, loss)| *loss).unwrap_or(0.0);
    let model_info = format!(
        "Model: {}\n\
        Loss: {:.6}\n\
        Epochs: {}/{}",
        if app.use_dlinoss { "dLinOSS" } else { "LinOSS" },
        current_loss,
        app.current_epoch,
        app.total_epochs
    );
    
    let info_paragraph = Paragraph::new(model_info)
        .style(Style::default().fg(Color::Cyan))
        .block(Block::default().title("Model").borders(Borders::ALL));
    f.render_widget(info_paragraph, bottom_chunks[0]);

    // Status and controls (right side of bottom)
    let model_type_str = if app.use_dlinoss { "dLinOSS" } else { "LinOSS" };
    let training_status = if app.current_epoch >= app.total_epochs { 
        "Complete - Press [Q] to quit" 
    } else { 
        "Active" 
    };
    let status_text = format!(
        "[M] Toggle Model\n\
        [H] Help Screen\n\
        [Q] Quit\n\n\
        Model: {}\n\
        Status: {}",
        model_type_str,
        training_status
    );
    
    let status_paragraph = Paragraph::new(status_text)
        .style(Style::default().fg(Color::Green))
        .block(Block::default().title("Controls").borders(Borders::ALL));
    f.render_widget(status_paragraph, bottom_chunks[1]);
}

// Debug logging function with size limit
fn debug_log(message: &str) {
    let log_path = "logs/damped_sine_debug.log";
    
    if let Ok(metadata) = std::fs::metadata(log_path) {
        // If log file is bigger than 30MB, rotate it
        if metadata.len() > 30_000_000 {
            let _ = std::fs::rename(log_path, "logs/damped_sine_debug.log.old");
        }
    }
    
    if let Ok(mut file) = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)
    {
        let _ = writeln!(file, "[{}] {}", 
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(), 
            message);
    }
}

// Clear debug log at start
fn clear_debug_log() {
    let _ = std::fs::remove_file("logs/damped_sine_debug.log");
}
