use burn::{
    backend::Autodiff, // Add explicit Autodiff import
    tensor::Element, // Add explicit Element import
    prelude::*,
};
use linoss_rust::linoss::{
    layer::{LinossLayer, LinossLayerConfig},
    LinossOutput,
};
use ratatui::{
    prelude::{Constraint, CrosstermBackend, Direction, Layout, Frame},
    style::{Color, Style},
    symbols,
    widgets::{Block as TuiBlock, Borders, Chart, Dataset, Paragraph},
    Terminal,
};
use std::{error::Error, io::{self, Stdout}, time::Duration, collections::VecDeque};

use crossterm::{
    event::{self, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};

// --- Backend Selection ---
#[cfg(feature = "wgpu_backend")]
mod gpu_backend {
    use burn::backend::wgpu::{Wgpu, WgpuDevice};
    pub type SelectedBackend = Wgpu<f32, i32>;
    pub fn get_device() -> WgpuDevice { WgpuDevice::default() }
}

#[cfg(all(feature = "ndarray_backend", not(feature = "wgpu_backend")))]
mod cpu_backend {
    use burn::backend::{NdArray, ndarray::NdArrayDevice};
    pub type SelectedBackend = NdArray<f32>;
    pub fn get_device() -> NdArrayDevice { NdArrayDevice::default() }
}

#[cfg(not(any(feature = "wgpu_backend", feature = "ndarray_backend")))]
mod cpu_backend {
    use burn::backend::{NdArray, ndarray::NdArrayDevice};
    pub type SelectedBackend = NdArray<f32>;
    pub fn get_device() -> NdArrayDevice { NdArrayDevice::default() }
}

#[cfg(feature = "wgpu_backend")]
use gpu_backend as chosen_backend;
#[cfg(all(feature = "ndarray_backend", not(feature = "wgpu_backend")))]
use cpu_backend as chosen_backend;
#[cfg(not(any(feature = "wgpu_backend", feature = "ndarray_backend")))]
use cpu_backend as chosen_backend;
// --- End Backend Selection ---

const CANVAS_X_BOUNDS: [f64; 2] = [-3.0, 3.0];
const CANVAS_Y_BOUNDS: [f64; 2] = [-3.0, 3.0];

const MAX_TRAJECTORY_POINTS: usize = 100;

#[derive(Debug)]
struct App<B: Backend> {
    model: LinossLayer<B>,
    current_input_x: Tensor<B, 2>,
    current_hidden_state_h: Tensor<B, 2>,
    trajectory_data: Vec<(f64, f64)>, // Store data directly for chart
    trajectory_deque: VecDeque<(f64, f64)>, // Keep VecDeque for push/pop logic
    x_min: f64,
    x_max: f64,
    y_min: f64,
    y_max: f64,
    paused: bool,
    status_message: String,
    data_status: String, // New field for data status
}

impl<B: Backend> App<B>
where
    B::FloatElem: Element + From<f32> + ElementConversion + std::ops::Mul<Output = B::FloatElem> + Copy + rand::distributions::uniform::SampleUniform + PartialOrd + std::fmt::Debug + 'static,
    B::IntElem: Element + From<i32> + ElementConversion + Copy + std::fmt::Debug + 'static,
{
    fn new(device: &B::Device) -> Self {
        // --- Experiment Parameters ---
        let delta_t = 0.1; // Original: 0.05. Try values like 0.01, 0.1, 0.2
        let init_std = 0.1; // Original: 0.02. Try values like 0.01, 0.05, 0.1, 0.5
        let initial_x_val = 0.5; // Original: 0.1. Try values like 0.0, 0.5, 1.0
        let initial_y_val = 0.5; // Original: 0.1. Try values like 0.0, 0.5, 1.0
        // --- End Experiment Parameters ---

        let linoss_state_size = 64;
        let input_dim = 2;

        let layer_config = LinossLayerConfig {
            d_state_m: linoss_state_size,
            d_input_p: input_dim,
            d_output_q: input_dim, // Corrected: Output dimension should match input dimension for feedback
            delta_t,
            enable_d_feedthrough: false,
            init_std,
        };

        let model: LinossLayer<B> = layer_config.init(device);
        let initial_state_h = Tensor::<B, 2>::zeros([1, linoss_state_size], device);
        let initial_input_x = Tensor::<B, 2>::from_data(
            TensorData::new(vec![initial_x_val as f32, initial_y_val as f32], [1, input_dim]).convert::<B::FloatElem>(),
            device,
        );

        let trajectory_deque = VecDeque::with_capacity(MAX_TRAJECTORY_POINTS);
        Self {
            model,
            current_input_x: initial_input_x,
            current_hidden_state_h: initial_state_h,
            trajectory_data: Vec::new(),
            trajectory_deque,
            x_min: f64::MAX, x_max: f64::MIN, y_min: f64::MAX, y_max: f64::MIN,
            paused: false,
            status_message: String::from("Running..."),
            data_status: String::from("Initializing..."), // Initialize new field
        }
    }

    fn on_tick(&mut self) {
        if self.paused {
            // self.data_status = String::from("Paused."); // Status message will indicate pause
            return;
        }

        let linoss_output: LinossOutput<B, 2> = self.model.forward_step(
            self.current_input_x.clone(),
            Some(self.current_hidden_state_h.clone())
        );

        self.current_input_x = linoss_output.output;
        self.current_hidden_state_h = linoss_output.hidden_state.unwrap();

        let h_state_for_plot = self.current_hidden_state_h.clone();

        // Flatten the tensor to 1D
        let h_state_flat: Tensor<B, 1> = h_state_for_plot.flatten(0,1);

        // Convert Tensor into TensorData
        let tensor_data: TensorData = h_state_flat.into_data();

        // Attempt to convert TensorData into Vec<f32> first, then to Vec<f64>
        match tensor_data.into_vec::<f32>() { // Explicitly request Vec<f32>
            Ok(state_vec_f32) => {
                // Now convert Vec<f32> to Vec<f64> for plotting
                let state_data_vec_f64: Vec<f64> = state_vec_f32.iter()
                    .map(|&val_f32| val_f32 as f64) // Cast f32 to f64
                    .collect();

                if state_data_vec_f64.len() >= 2 {
                    let point = (state_data_vec_f64[0], state_data_vec_f64[1]);

                    self.trajectory_deque.push_back(point);
                    if self.trajectory_deque.len() > MAX_TRAJECTORY_POINTS {
                        self.trajectory_deque.pop_front();
                    }
                    self.trajectory_data = self.trajectory_deque.iter().cloned().collect();

                    self.x_min = self.x_min.min(point.0);
                    self.x_max = self.x_max.max(point.0);
                    self.y_min = self.y_min.min(point.1);
                    self.y_max = self.y_max.max(point.1);

                    self.data_status = format!(
                        "Last pt: ({:.1e},{:.1e}). X: [{:.1e},{:.1e}], Y: [{:.1e},{:.1e}]. Len: {}",
                        point.0, point.1,
                        self.x_min, self.x_max, self.y_min, self.y_max,
                        self.trajectory_data.len()
                    );

                    self.status_message = format!("Running... Last point: ({:.2}, {:.2})", point.0, point.1);

                } else {
                    self.data_status = format!(
                        "DataErr: State < 2. F64Len: {}, F32Len: {}",
                        state_data_vec_f64.len(),
                        state_vec_f32.len()
                    );
                }
            },
            Err(e) => {
                self.data_status = format!("TensorErr: {:?}", e);
            }
        }
    }
}

// Helper function for TUI setup
fn tui_app_startup() -> Result<Terminal<CrosstermBackend<Stdout>>, Box<dyn Error>> { // Use CrosstermBackend
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout); // Use CrosstermBackend
    let terminal = Terminal::new(backend)?;
    Ok(terminal)
}

// Helper function for TUI shutdown
fn tui_app_shutdown(mut terminal: Terminal<CrosstermBackend<Stdout>>) -> Result<(), Box<dyn Error>> { // Use CrosstermBackend
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
    )?;
    terminal.show_cursor()?;
    Ok(())
}


#[cfg(feature = "wgpu_backend")]
mod wgpu_specific {
    use super::*;

    pub fn run() -> Result<(), Box<dyn Error>> {
        let device = chosen_backend::get_device();
        let app = App::<Autodiff<chosen_backend::SelectedBackend>>::new(&device);
        let mut terminal = tui_app_startup()?;
        let res = run_chaotic_linoss_tui(&mut terminal, app, Duration::from_millis(100));
        tui_app_shutdown(terminal)?;
        res
    }
}

#[cfg(feature = "ndarray_backend")]
mod ndarray_specific {
    use super::*;

    #[allow(dead_code)] // Used conditionally based on feature flags
    pub fn run() -> Result<(), Box<dyn Error>> {
        let device = chosen_backend::get_device();
        let app = App::<Autodiff<chosen_backend::SelectedBackend>>::new(&device);
        let mut terminal = tui_app_startup()?;
        let res = run_chaotic_linoss_tui(&mut terminal, app, Duration::from_millis(100));
        tui_app_shutdown(terminal)?;
        res
    }
}

fn run_chaotic_linoss_tui<B: Backend>(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>, // Use CrosstermBackend
    mut app: App<B>,
    tick_rate: Duration,
) -> Result<(), Box<dyn Error>>
where
    B::FloatElem: Element + From<f32> + ElementConversion + std::ops::Mul<Output = B::FloatElem> + Copy + rand::distributions::uniform::SampleUniform + PartialOrd + std::fmt::Debug,
    B::IntElem: Element + From<i32> + ElementConversion + Copy + std::fmt::Debug,
{
    let mut last_tick = std::time::Instant::now();
    loop {
        terminal.draw(|f| ui(f, &app))?;

        let timeout = tick_rate
            .checked_sub(last_tick.elapsed())
            .unwrap_or_else(|| Duration::from_secs(0));

        if crossterm::event::poll(timeout)? {
            if let event::Event::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Char('q') => return Ok(()),
                    KeyCode::Char(' ') => app.paused = !app.paused,
                    _ => {}
                }
                app.status_message = if app.paused { String::from("Paused. Press Space to resume, Q to quit.") } else { String::from("Running... Press Space to pause, Q to quit.") };
            }
        }

        if last_tick.elapsed() >= tick_rate {
            app.on_tick();
            last_tick = std::time::Instant::now();
        }
    }
}

fn ui<B: Backend>(f: &mut Frame, app: &App<B>)
where
    B::FloatElem: Element + From<f32> + ElementConversion + std::ops::Mul<Output = B::FloatElem> + Copy + rand::distributions::uniform::SampleUniform + PartialOrd + std::fmt::Debug,
    B::IntElem: Element + From<i32> + ElementConversion + Copy + std::fmt::Debug,
{
    let main_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(0), // Chart takes remaining space
            Constraint::Length(3), // Status area takes 3 lines (Controls title, Control status, Data status)
        ].as_ref())
        .split(f.area());

    let chart_area = main_chunks[0];
    let status_area = main_chunks[1];

    let x_bounds = if app.x_min == f64::MAX || app.x_min == app.x_max { CANVAS_X_BOUNDS } else { [app.x_min, app.x_max] };
    let y_bounds = if app.y_min == f64::MAX || app.y_min == app.y_max { CANVAS_Y_BOUNDS } else { [app.y_min, app.y_max] };

    let datasets = vec![
        Dataset::default()
            .name("trajectory")
            .marker(symbols::Marker::Dot)
            .style(Style::default().fg(Color::Cyan))
            .data(app.trajectory_data.as_slice()),
    ];

    let chart = Chart::new(datasets)
        .block(
            TuiBlock::default()
                .title("LinOSS Chaotic System")
                .borders(Borders::ALL),
        )
        .x_axis(
            ratatui::widgets::Axis::default()
                .title("X")
                .style(Style::default().fg(Color::Gray))
                .bounds(x_bounds),
        )
        .y_axis(
            ratatui::widgets::Axis::default()
                .title("Y")
                .style(Style::default().fg(Color::Gray))
                .bounds(y_bounds),
        );
    f.render_widget(chart, chart_area);

    // Create a block for the entire status area first
    let status_block = TuiBlock::default().borders(Borders::TOP).title("Controls & Status");
    let inner_status_area = status_block.inner(status_area); // Get area inside the block's borders
    f.render_widget(status_block, status_area); // Render the block itself

    // Then define layout for content within that block
    let status_content_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // Control status
            Constraint::Length(1), // Data status
        ].as_ref())
        .split(inner_status_area); // Use the inner area for content

    let control_status_paragraph = Paragraph::new(app.status_message.as_str());
    f.render_widget(control_status_paragraph, status_content_chunks[0]);

    let data_status_paragraph = Paragraph::new(app.data_status.as_str());
    f.render_widget(data_status_paragraph, status_content_chunks[1]);
}


fn main() -> Result<(), Box<dyn Error>> {
    #[cfg(feature = "wgpu_backend")]
    wgpu_specific::run()?;

    #[cfg(all(feature = "ndarray_backend", not(feature = "wgpu_backend")))]
    ndarray_specific::run()?;

    #[cfg(not(any(feature = "wgpu_backend", feature = "ndarray_backend")))]
    {
        println!("Please enable a backend feature (wgpu_backend or ndarray_backend) to run this example.");
    }

    Ok(())
}

