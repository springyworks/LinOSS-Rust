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
    io,
    time::{Duration, Instant},
};

const D_MODEL: usize = 16;
const D_STATE_M: usize = 32; // Increased state dimension for more complex dynamics
const N_LAYERS: usize = 2;
const SEQ_LEN: usize = 100;
const BATCH_SIZE: usize = 1;
const LEARNING_RATE: f64 = 5e-3; // Slightly increased learning rate
const N_EPOCHS: usize = 100;

// Define the backend type alias
type MyBackend = burn::backend::NdArray<f32>; // Or Wgpu<f32, i32>
type MyAutodiffBackend = burn::backend::Autodiff<MyBackend>; // Corrected path for Autodiff

// App state for the visualization
struct DampedSineApp<AB: AutodiffBackend>
where
    AB::FloatElem: From<f32> + std::ops::Mul<Output = AB::FloatElem> + Copy + ElementConversion + num_traits::FloatConst,
{
    input_data: Vec<(f64, f64)>,      // (time, value) for step function input
    output_data: Vec<(f64, f64)>,     // (time, value) for model output
    target_data: Vec<(f64, f64)>,     // (time, value) for target damped sine wave
    xy_scatter_data: Vec<(f64, f64)>, // (input_val, output_val) for scatter plot
    model: FullLinossModel<AB>,
    current_epoch: usize,
    total_epochs: usize,
    loss_history: Vec<(f64, f64)>,
    should_quit: bool,
}

impl<AB: AutodiffBackend> DampedSineApp<AB>
where
    AB::FloatElem: From<f32> + std::ops::Mul<Output = AB::FloatElem> + Copy + ElementConversion + num_traits::FloatConst,
{
    fn new(model: FullLinossModel<AB>) -> Self {
        // Time points
        let time_points: Vec<f64> = (0..SEQ_LEN).map(|i| i as f64 * 0.1).collect();
        
        // Step function input: 1.0 at t=0, 0 elsewhere
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
                    (-damping_factor * t).exp() * (frequency * t).sin()
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
            model,
            current_epoch: 0,
            total_epochs: N_EPOCHS,
            loss_history: Vec::new(),
            should_quit: false,
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

        // Forward pass through the model
        let output_tensor = self.model.forward(input_tensor.clone());
        
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
    
    fn train_step<O: Optimizer<FullLinossModel<AB>, AB>>(&mut self, optim: &mut O, backend_device: &AB::Device) { 
        // Prepare input tensor
        let input_values: Vec<f32> = self.input_data.iter()
            .map(|(_, val)| *val as f32)
            .collect();
        
        let input_tensor = Tensor::<AB, 2>::from_data(
            TensorData::new(input_values, [BATCH_SIZE, SEQ_LEN]), // Removed redundant .convert::<f32>()
            backend_device,
        ).reshape([BATCH_SIZE, SEQ_LEN, 1]);

        // Prepare target tensor (damped sine wave)
        let target_values: Vec<f32> = self.target_data.iter()
            .map(|(_, val)| *val as f32)
            .collect();
        
        let target_tensor = Tensor::<AB, 2>::from_data(
            TensorData::new(target_values, [BATCH_SIZE, SEQ_LEN]), // Removed redundant .convert::<f32>()
            backend_device,
        ).reshape([BATCH_SIZE, SEQ_LEN, 1]);

        // Forward pass and loss calculation
        let model_output = self.model.forward(input_tensor.clone());
        let loss = (model_output.clone() - target_tensor.clone()).powf_scalar(2.0).mean();
        
        // Backward pass and optimization
        let grads_raw = loss.backward();
        let grads = GradientsParams::from_grads(grads_raw, &self.model);
        self.model = optim.step(LEARNING_RATE, self.model.clone(), grads);
        
        // Update visualizations
        self.update_model_output(backend_device);
        
        // Update loss history
        self.loss_history.push((self.current_epoch as f64, loss.into_scalar().elem::<f64>()));
        if self.loss_history.len() > SEQ_LEN { 
            self.loss_history.remove(0);
        }
        
        // Update epoch counter
        self.current_epoch += 1;
        if self.current_epoch >= self.total_epochs {
            self.should_quit = true; 
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
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

    // Initialize the model
    let model: FullLinossModel<MyAutodiffBackend> = model_config.init(&device);

    // Create the app state
    let mut app = DampedSineApp::new(model);
    app.update_model_output(&device); // Initial output before training

    // Initialize the optimizer
    let mut optim = AdamConfig::new().init();

    // Run the application
    let res = run_app(&mut terminal, &mut app, &mut optim, &device);

    // Restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    if let Err(err) = res {
        println!("{:?}", err)
    }

    Ok(())
}

// RB for RatatuiBackend, AB for AutodiffBackend
fn run_app<RB: RatatuiBackend, AB: AutodiffBackend, O: Optimizer<FullLinossModel<AB>, AB>>( // Made optimizer generic
    terminal: &mut Terminal<RB>,
    app: &mut DampedSineApp<AB>, 
    optimizer: &mut O, // Use generic optimizer type O
    backend_device: &AB::Device,
) -> io::Result<()> 
where
    AB::FloatElem: From<f32> + std::ops::Mul<Output = AB::FloatElem> + Copy + ElementConversion + num_traits::FloatConst,
{
    let tick_rate = Duration::from_millis(100); 
    let mut last_tick = Instant::now();

    loop {
        if app.should_quit {
            return Ok(());
        }

        terminal.draw(|f| ui(f, app))?;

        let timeout = tick_rate
            .checked_sub(last_tick.elapsed())
            .unwrap_or_else(|| Duration::from_secs(0));

        if crossterm::event::poll(timeout)? {
            if let Event::Key(key) = event::read()? {
                if key.code == KeyCode::Char('q') {
                    app.should_quit = true;
                }
            }
        }

        if last_tick.elapsed() >= tick_rate {
            app.train_step(optimizer, backend_device);
            last_tick = Instant::now();
        }
    }
}

// UI rendering function
fn ui<AB: AutodiffBackend>(f: &mut Frame<'_>, app: &DampedSineApp<AB>) 
where
    AB::FloatElem: From<f32> + std::ops::Mul<Output = AB::FloatElem> + Copy + ElementConversion + num_traits::FloatConst,
{
    let area = f.area(); // Changed from size() to area()
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints(
            [
                Constraint::Percentage(50), // For output and target plots
                Constraint::Percentage(30), // For loss plot
                Constraint::Percentage(20), // For scatter plot and status
            ]
            .as_ref(),
        )
        .split(area); // Use area here

    // Split the bottom chunk for scatter plot and status text
    let bottom_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(70), Constraint::Percentage(30)].as_ref())
        .split(chunks[2]);
    
    // Loss history chart
    let loss_datasets = vec![Dataset::default()
        .name("Training Loss")
        .marker(symbols::Marker::Braille)
        .graph_type(GraphType::Line)
        .style(Style::default().fg(Color::Red))
        .data(&app.loss_history)];
    
    let loss_chart = Chart::new(loss_datasets)
        .block(Block::default().title("Training Loss").borders(Borders::ALL))
        .x_axis(Axis::default()
            .title("Epoch")
            .style(Style::default().fg(Color::Gray))
            .bounds([0.0, app.total_epochs as f64]))
        .y_axis(Axis::default()
            .title("Loss")
            .style(Style::default().fg(Color::Gray))
            .bounds([0.0, app.loss_history.iter().map(|(_,y)| *y).fold(0.0, f64::max).max(0.1)]));
    f.render_widget(loss_chart, bottom_chunks[0]);

    // Status text
    let loss_text = if let Some(last_loss) = app.loss_history.last() {
        format!("Epoch: {}/{}\nLoss: {:.6}", app.current_epoch, app.total_epochs, last_loss.1)
    } else {
        format!("Epoch: {}/{}\nLoss: N/A", app.current_epoch, app.total_epochs)
    };
    
    let status_paragraph = Paragraph::new(loss_text)
        .style(Style::default().fg(Color::Green))
        .block(Block::default().title("Training Status").borders(Borders::ALL));
    f.render_widget(status_paragraph, bottom_chunks[1]);
}
