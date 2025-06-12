// Constants
const INPUT_SIZE: usize = 1;
const HIDDEN_SIZE: usize = 50; // This will be d_state_m and d_output_q for LinossLayer
const OUTPUT_SIZE: usize = 1; // Final output of the model
const BATCH_SIZE: usize = 1;
const SEQ_LEN: usize = 100;
const DELTA_T: f32 = 0.05; // Time step for LinOSS

// Standard library imports
use std::io::{self, Stdout};
// use std::time::Duration as StdDuration; // Ensure this is used or remove
use std::fmt::Display;
use std::error::Error; // Add Error trait

// Burn imports
use burn::{
    backend::{
        // Ensure WgpuDevice is imported if used, AutoGraphicsApi might be part of WgpuConfig
        wgpu::{WgpuDevice, WgpuConfig, AutoGraphicsApi}, 
        Autodiff as AutodiffBackendStruct,
        NdArray, Wgpu,
    },
    // Explicitly import traits that might not be covered by prelude alone if issues persist
    backend::{Backend, AutodiffBackend as BurnAutodiffBackendTrait, LinalgBackend},
    module::{Module},
    nn::loss::{MseLoss, Reduction},
    // Ensure Adam, AdamConfig, OptimizerAdaptor, GradientsParams are correctly scoped
    optim::{Adam, AdamConfig, GradientsParams, adaptor::OptimizerAdaptor}, 
    prelude::*, // Keep prelude for common items
    record::{CompactRecorder},
    // Ensure TensorOps, Distribution, Shape are correctly scoped
    tensor::{Element, TensorData, ElementConversion, ops::TensorOps, Distribution, Shape}, 
};

// Assuming Model and ModelConfig are directly under linoss_rust::linoss module
// If they are in a sub-module like `model`, the path should be `linoss_rust::linoss::model::{Model, ModelConfig}`
use linoss_rust::linoss::{Model, ModelConfig}; 

use rand::distributions::uniform::SampleUniform;
use rand::Rng;

use crossterm::{
    event::{self, Event, KeyCode, KeyEvent, KeyEventKind},
    execute,
    terminal::{enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Alignment, Constraint, Direction, Layout},
    style::{Color, Style, Stylize},
    symbols,
    text::Span,
    widgets::{Axis, Block, Borders, Chart, Dataset, Paragraph},
    Terminal,
    Frame,
};

// Type aliases for the backends
type CurrentBackend = NdArray<f32>;
type CurrentAutodiffBackend = AutodiffBackendStruct<CurrentBackend>; // Use renamed Autodiff

#[cfg(feature = "wgpu_backend")]
type CurrentWgpuBackend = Wgpu<f32, i32>;
#[cfg(feature = "wgpu_backend")]
type CurrentAutodiffWgpuBackend = AutodiffBackendStruct<CurrentWgpuBackend>; // Use renamed Autodiff


struct AppConfig {
    num_epochs: usize,
    lr: f64,
    log_interval: usize,
    // checkpoint_interval: usize, // Not used directly in AppState logic, consider removing if not needed elsewhere
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    sequence_length: usize,
    num_predictions_plot: usize,
    model_dir: String,
    save_interval: usize,
}

struct AppState<AB: BurnAutodiffBackendTrait> // Used aliased AutodiffBackend
where
    AB::InnerBackend: Backend<FloatElem = f32, IntElem = i32> + LinalgBackend + 'static,
    AB::FloatElem: From<f32> + Copy + SampleUniform + PartialOrd + std::fmt::Debug + Display + Element,
    AB::IntElem: From<i32> + Copy + PartialOrd + std::fmt::Debug + Element,
{
    model: Model<AB>,
    optim: OptimizerAdaptor<Adam, Model<AB>, AB>, // Corrected Adam generic
    recurrent_state: Option<Tensor<AB, 2>>,
    input_data_for_plot: Vec<(f64, f64)>,
    target_data: Vec<(f64, f64)>,
    predictions: Vec<(f64, f64)>,
    current_epoch: usize,
    config: AppConfig,
    train_input: Vec<f32>,
    train_target: Vec<f32>,
    val_input: Vec<f32>,
    val_target: Vec<f32>,
    losses: Vec<f32>,
    is_running: bool,
    status_message: String,
    device: AB::Device, // Store the device for the AutodiffBackend
}

impl<AB: BurnAutodiffBackendTrait> AppState<AB> // Used aliased AutodiffBackend
where
    AB::InnerBackend: Backend<FloatElem = f32, IntElem = i32> + LinalgBackend + 'static,
    AB::FloatElem: From<f32> + Copy + SampleUniform + PartialOrd + std::fmt::Debug + Display + Element,
    AB::IntElem: From<i32> + Copy + PartialOrd + std::fmt::Debug + Element,
    <AB::InnerBackend as Backend>::FloatElem: From<f32> + Copy + SampleUniform + PartialOrd + std::fmt::Debug + Display + Element,
    <AB::InnerBackend as Backend>::IntElem: From<i32> + Copy + PartialOrd + std::fmt::Debug + Element,
{
    fn new(config: AppConfig, device: AB::Device) -> Self {
        let device_inner = AB::InnerBackend::Device::default();

        let model_inner: Model<AB::InnerBackend> = ModelConfig::new(
            config.input_size,
            config.hidden_size,
            config.output_size,
        )
        .init(&device_inner);

        let model_ad: Model<AB> = model_inner.fork(&device);

        let optim: OptimizerAdaptor<Adam, Model<AB>, AB> = // Corrected Adam generic
            OptimizerAdaptor::new(AdamConfig::new());

        // Corrected function name and since it returns tensors, we handle it differently for plot data
        // For initial plotting data, we might need a separate, simpler data generation or use parts of a batch
        // Here, we'll generate a batch and take some for plotting.
        // This part needs careful consideration on how initial plot data is sourced.
        // For simplicity, let's assume train_input/val_input are populated first.

        let (train_input_vec, train_target_vec, val_input_vec, val_target_vec) =
            generate_sine_wave_data_vecs(1000, 200); // Assuming a new helper for Vec<f32>

        let temp_input_tensor_inner = Tensor::<AB::InnerBackend, 1>::from_data(
            TensorData::new(train_input_vec.clone(), [train_input_vec.len()]).convert(),
            &device_inner,
        );
        let input_data_for_plot = temp_input_tensor_inner
            .into_data()
            .convert::<f32>()
            .into_vec()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(i, &y)| (i as f64, y as f64))
            .collect();

        AppState {
            model: model_ad,
            optim,
            recurrent_state: None,
            input_data_for_plot,
            target_data: val_target_vec // Use the vec directly
                .iter()
                .enumerate()
                .map(|(i, &y)| (i as f64, y as f64))
                .collect(),
            predictions: Vec::new(),
            current_epoch: 0,
            config,
            train_input: train_input_vec, // Store vecs
            train_target: train_target_vec,
            val_input: val_input_vec,
            val_target: val_target_vec,
            losses: Vec::new(),
            is_running: false,
            status_message: "Idle".to_string(),
            device,
        }
    }

    fn reset_model_and_optimizer(&mut self) {
        let device_inner = AB::InnerBackend::Device::default();
        let model_inner: Model<AB::InnerBackend> = ModelConfig::new(
            self.config.input_size,
            self.config.hidden_size,
            self.config.output_size,
        )
        .init(&device_inner);
        self.model = model_inner.fork(&self.device); // Fork to the AutodiffBackend's device
        self.optim = OptimizerAdaptor::new(AdamConfig::new());
        self.current_epoch = 0;
        self.losses = Vec::new();
        self.recurrent_state = None; // Reset recurrent state
        self.status_message = "Model reset".to_string();
    }

    fn train_epoch(&mut self) {
        self.model.train();
        let num_batches = self.train_input.len() / self.config.sequence_length;
        let device_inner = AB::InnerBackend::Device::default(); // Device for InnerBackend tensors

        for i in 0..num_batches {
            let start = i * self.config.sequence_length;
            let end = start + self.config.sequence_length;
            let input_data_batch: Vec<f32> = self.train_input[start..end].to_vec();
            let target_data_batch: Vec<f32> = self.train_target[start..end].to_vec();

            let input_tensor_inner = Tensor::<AB::InnerBackend, 2>::from_data(
                TensorData::new(input_data_batch, [self.config.sequence_length, self.config.input_size]).convert(),
                &device_inner,
            );
            let target_tensor_inner = Tensor::<AB::InnerBackend, 2>::from_data(
                TensorData::new(target_data_batch, [self.config.sequence_length, self.config.output_size]).convert(),
                &device_inner,
            );

            let input_tensor_ad: Tensor<AB, 2> = input_tensor_inner.fork(&self.device);
            let target_tensor_ad: Tensor<AB, 2> = target_tensor_inner.fork(&self.device);

            let mut current_batch_loss = 0.0;
            // Use the stored recurrent_state, detaching it for the start of a new batch/sequence if needed
            // For simplicity here, we re-initialize, but for longer sequences, passing state is better.
            // Or, ensure recurrent_state is on the correct device and detached if it's carried over.
            let mut hidden_state_ad: Option<Tensor<AB, 2>> = self.recurrent_state.clone();


            for t in 0..self.config.sequence_length {
                let input_step_ad = input_tensor_ad.clone().slice([t..t + 1, 0..self.config.input_size]);
                let target_step_ad = target_tensor_ad.clone().slice([t..t + 1, 0..self.config.output_size]);

                let (output_step_ad, next_hidden_state_ad) =
                    self.model.forward_step(input_step_ad, hidden_state_ad.clone());
                hidden_state_ad = Some(next_hidden_state_ad);

                let loss_step = MseLoss::new().forward(output_step_ad, target_step_ad, Reduction::Mean);
                current_batch_loss += loss_step.clone().into_scalar().elem::<f32>();

                let grads = loss_step.backward();
                self.model = self.optim.step(self.config.lr, self.model.clone(), grads);
            }
            // Detach the final hidden state of the batch before storing
            self.recurrent_state = hidden_state_ad.map(|t| t.detach());
            self.losses.push(current_batch_loss / self.config.sequence_length as f32);
        }
        self.current_epoch += 1;
        self.status_message = format!("Epoch {} complete. Avg Loss: {:.4}", self.current_epoch, self.losses.last().copied().unwrap_or(0.0));
    }

    fn get_predictions_for_plotting(&mut self) { // Changed to &mut self to update self.predictions
        self.model.eval();
        let model_for_plot: Model<AB::InnerBackend> = self.model.clone().detach(); // Detach model to InnerBackend
        let device_inner = AB::InnerBackend::Device::default();

        let mut predictions_vec = Vec::new();
        // let mut actual_values_vec = Vec::new(); // Not strictly needed if self.val_target is used

        // Detach recurrent state to InnerBackend for inference
        let mut hidden_state_inner: Option<Tensor<AB::InnerBackend, 2>> =
            self.recurrent_state.as_ref().map(|t_ad| t_ad.clone().detach().into_inner());

        for i in 0..self.config.num_predictions_plot {
            let input_val = self.val_input.get(i).copied().unwrap_or(0.0);
            // let target_val = self.val_target.get(i).copied().unwrap_or(0.0); // For actual_values_vec if used

            let input_step_data = TensorData::new(vec![input_val], [1, self.config.input_size]).convert();
            let input_step_inner = Tensor::<AB::InnerBackend, 2>::from_data(input_step_data, &device_inner);

            let (output_step_inner, next_hidden_state_inner) =
                model_for_plot.forward_step(input_step_inner, hidden_state_inner.clone());
            hidden_state_inner = Some(next_hidden_state_inner);

            predictions_vec.push(output_step_inner.into_data().convert::<f32>().into_vec().unwrap()[0]);
            // actualValuesVec.push(target_val);
        }
        self.predictions = predictions_vec.iter().enumerate().map(|(i, &y)| (i as f64, y as f64)).collect();
        // self.target_data could remain as is if it's already from val_target for plotting
        self.status_message = "Predictions updated".to_string();
    }

    fn save_checkpoint(&self) {
        if self.current_epoch > 0 && self.current_epoch % self.config.save_interval == 0 {
            let model_path = format!("{}/model_epoch_{}.bin", self.config.model_dir, self.current_epoch);
            std::fs::create_dir_all(&self.config.model_dir).expect("Failed to create model directory");
            let model_to_save: Model<AB::InnerBackend> = self.model.clone().detach();
            model_to_save
                .save_file(&model_path, &CompactRecorder::new())
                .expect("Failed to save model");
            // self.status_message = format!("Model saved to {}", model_path); // Cannot change status_message with &self
        }
    }
}

// Placeholder for a function that returns Vec<f32>
// Replace this with your actual data generation logic that produces Vecs
fn generate_sine_wave_data_vecs(num_train: usize, num_val: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut train_i = Vec::new();
    let mut train_t = Vec::new();
    let mut val_i = Vec::new();
    let mut val_t = Vec::new();
    let mut rng = rand::thread_rng(); // Rng trait needs to be in scope

    for i in 0..(num_train + num_val) {
        let x = i as f32 * 0.1 + rng.gen::<f32>() * 0.1 - 0.05;
        let y = (x * 3.0).sin() * 0.5 + (x * 0.5).cos() * 0.3;
        if i < num_train {
            train_i.push(x);
            train_t.push(y);
        } else {
            val_i.push(x);
            val_t.push(y);
        }
    }
    let mut final_train_input = Vec::new();
    let mut final_train_target = Vec::new();
    for i in 0..num_train {
        final_train_input.push(i as f32 / num_train as f32 + rng.gen::<f32>() * 0.01);
        final_train_target.push(train_t[i]);
    }

    let mut final_val_input = Vec::new();
    let mut final_val_target = Vec::new();
    for i in 0..num_val {
        final_val_input.push((num_train + i) as f32 / (num_train + num_val) as f32 + rng.gen::<f32>() * 0.01);
        final_val_target.push(val_t[i]);
    }

    (final_train_input, final_train_target, final_val_input, final_val_target)
}


fn run_tui_app<AB: BurnAutodiffBackendTrait>(
    app_config: AppConfig,
    app_device: AB::Device,
) -> Result<(), Box<dyn std::error::Error>> // Explicitly use std::error::Error
where
    AB::InnerBackend: Backend<FloatElem = f32, IntElem = i32> + LinalgBackend + 'static,
    AB::FloatElem: From<f32> + Copy + SampleUniform + PartialOrd + std::fmt::Debug + Display + Element,
    AB::IntElem: From<i32> + Copy + PartialOrd + std::fmt::Debug + Element,
    <AB::InnerBackend as Backend>::FloatElem: From<f32> + Copy + SampleUniform + PartialOrd + std::fmt::Debug + Display + Element,
    <AB::InnerBackend as Backend>::IntElem: From<i32> + Copy + PartialOrd + std::fmt::Debug + Element,
{
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?; // execute! should now be in scope
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?; // terminal is now defined

    let mut app_state = AppState::new(app_config, app_device.clone());
    std::fs::create_dir_all(&app_state.config.model_dir).expect("Failed to create model directory");
    app_state.get_predictions_for_plotting();

    loop {
        terminal.draw(|f| ui(f, &mut app_state))?;

        if crossterm::event::poll(std::time::Duration::from_millis(250))? { // event::poll should be fine
            if let Event::Key(key) = event::read()? { // event::read and Event should be fine
                if key.kind == KeyEventKind::Press { // KeyEventKind should be fine
                    handle_input(key, &mut app_state)?;
                }
            }
        }

        if app_state.is_running && app_state.current_epoch < app_state.config.num_epochs {
            app_state.train_epoch();
            app_state.get_predictions_for_plotting();
            app_state.save_checkpoint();
        } else if app_state.is_running && app_state.current_epoch >= app_state.config.num_epochs {
            app_state.is_running = false;
            app_state.status_message = format!("Training complete: {} epochs reached.", app_state.config.num_epochs);
        }

        if !app_state.is_running && app_state.status_message == "Quitting..." {
            break;
        }
    }

    disable_raw_mode()?; // Cleanup raw mode
    execute!(terminal.backend_mut(), LeaveAlternateScreen,)?; // Cleanup alternate screen
    terminal.show_cursor()?;

    if app_state.current_epoch > 0 {
        let final_model_path = format!("{}/model_final.bin", app_state.config.model_dir);
        let model_to_save: Model<AB::InnerBackend> = app_state.model.clone().detach();
        model_to_save
            .save_file(&final_model_path, &CompactRecorder::new())
            .expect("Failed to save final model");
    }
    Ok(())
}

fn handle_input<AB: BurnAutodiffBackendTrait>(key: KeyEvent, app: &mut AppState<AB>) -> Result<(), Box<dyn std::error::Error>> { // Explicitly use std::error::Error
    match key.code { // KeyCode should be fine
        KeyCode::Char('q') => {
            app.is_running = false; // Stop training if running
            app.status_message = "Quitting...".to_string(); // Set status for main loop to break
        }
        KeyCode::Char('s') => {
            if !app.is_running {
                app.is_running = true;
                app.status_message = "Training started...".to_string();
            }
        }
        KeyCode::Char('p') => {
            app.is_running = false;
            app.status_message = "Training paused.".to_string();
        }
        KeyCode::Char('r') => {
            app.reset_model_and_optimizer();
            app.get_predictions_for_plotting(); // Update plot after reset
            app.status_message = "Model reset. Press 's' to start training.".to_string();
        }
        _ => {}
    }
    Ok(())
}

fn ui<AB: BurnAutodiffBackendTrait>(f: &mut Frame, app: &mut AppState<AB>) { // Used aliased AutodiffBackend
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(2)
        .constraints(
            [
                Constraint::Length(3), // For status
                Constraint::Percentage(45), // For loss plot
                Constraint::Percentage(45), // For prediction plot
                Constraint::Length(3), // For instructions
            ]
            .as_ref(),
        )
        .split(f.size());

    let status_text = format!(
        "Epoch: {}/{}, Loss: {:.4}, Status: {}",
        app.current_epoch,
        app.config.num_epochs,
        app.losses.last().copied().unwrap_or(0.0),
        app.status_message
    );
    let status_paragraph = Paragraph::new(status_text)
        .style(Style::default().fg(Color::Yellow))
        .alignment(Alignment::Center);
    f.render_widget(status_paragraph, chunks[0]);

    let losses_data: Vec<(f64, f64)> = app
        .losses
        .iter()
        .enumerate()
        .map(|(i, &l)| (i as f64, l as f64))
        .collect();
    let loss_dataset = Dataset::default()
        .name("Loss")
        .marker(symbols::Marker::Dot)
        .style(Style::default().fg(Color::Cyan))
        .data(&losses_data);

    let x_axis_loss = Axis::default()
        .title("Epoch")
        .style(Style::default().fg(Color::Gray))
        .bounds([0.0, app.config.num_epochs as f64]); // Use app.config
    let y_axis_loss = Axis::default()
        .title("Loss")
        .style(Style::default().fg(Color::Gray))
        .bounds([0.0, app.losses.iter().fold(0.0, |acc:f32, &x| acc.max(x)) as f64 + 0.1]); // Dynamic Y axis for loss


    let chart_loss = Chart::new(vec![loss_dataset])
        .block(
            Block::default()
                .title("Training Loss")
                .borders(Borders::ALL),
        )
        .x_axis(x_axis_loss)
        .y_axis(y_axis_loss);
    f.render_widget(chart_loss, chunks[1]);


    let actual_dataset = Dataset::default()
        .name("Actual")
        .marker(symbols::Marker::Dot)
        .style(Style::default().fg(Color::Green))
        .data(&app.target_data); // Use app.target_data which is already Vec<(f64,f64)>

    let predicted_dataset = Dataset::default()
        .name("Predicted")
        .marker(symbols::Marker::Braille)
        .style(Style::default().fg(Color::Magenta))
        .data(&app.predictions); // Use app.predictions which is already Vec<(f64,f64)>

    let x_max_pred = app.config.num_predictions_plot as f64;
    let y_min_pred = app.target_data.iter().map(|&(_, y)| y).fold(f64::INFINITY, |a, b| a.min(b)) - 0.1;
    let y_max_pred = app.target_data.iter().map(|&(_, y)| y).fold(f64::NEG_INFINITY, |a, b| a.max(b)) + 0.1;


    let chart_pred = Chart::new(vec![actual_dataset, predicted_dataset])
        .block(
            Block::default()
                .title("Sine Wave Prediction")
                .borders(Borders::ALL),
        )
        .x_axis(
            Axis::default()
                .title("Time Step")
                .style(Style::default().fg(Color::Gray))
                .bounds([0.0, x_max_pred]),
        )
        .y_axis(
            Axis::default()
                .title("Value")
                .style(Style::default().fg(Color::Gray))
                .bounds([y_min_pred, y_max_pred]) // Dynamic Y axis
                .labels(vec![
                    Span::styled(format!("{:.1}", y_min_pred), Style::default().fg(Color::Gray)),
                    Span::styled(format!("{:.1}", (y_min_pred + y_max_pred) / 2.0), Style::default().fg(Color::Gray)),
                    Span::styled(format!("{:.1}", y_max_pred), Style::default().fg(Color::Gray)),
                ]),
        );
    f.render_widget(chart_pred, chunks[2]);


    let instructions = "Controls: (q)uit, (s)tart training, (p)ause training, (r)eset model";
    let instructions_paragraph = Paragraph::new(instructions)
        .style(Style::default().fg(Color::White))
        .alignment(Alignment::Center);
    f.render_widget(instructions_paragraph, chunks[3]);
}

fn main() -> Result<(), Box<dyn std::error::Error>> { // Explicitly use std::error::Error
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Determine backend based on feature flags
    #[cfg(not(feature = "wgpu_backend"))]
    let device_main = <CurrentAutodiffBackend as Backend>::Device::default(); // Renamed
    #[cfg(feature = "wgpu_backend")]
    let device_main = <CurrentAutodiffWgpuBackend as Backend>::Device::default(); // Renamed

    // Create a directory for models if it doesn't exist
    let model_dir_main = "models_sine_wave_vis".to_string(); // Renamed
    std::fs::create_dir_all(&model_dir_main).expect("Failed to create model directory");

    let config_main = AppConfig { // Renamed
        num_epochs: 100,
        lr: 0.005,
        log_interval: 1,
        input_size: 1,
        hidden_size: 20,
        output_size: 1,
        sequence_length: 50,
        num_predictions_plot: 200,
        model_dir: model_dir_main, // Use renamed var
        save_interval: 10,
    };

    #[cfg(not(feature = "wgpu_backend"))]
    run_tui_app::<CurrentAutodiffBackend>(config_main, device_main)?;

    #[cfg(feature = "wgpu_backend")]
    run_tui_app::<CurrentAutodiffWgpuBackend>(config_main, device_main)?;

    Ok(())
}

