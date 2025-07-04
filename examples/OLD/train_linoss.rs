// examples/train_linoss.rs
// Basic training loop example for FullLinossModel.

use std::error::Error;
use std::io::{self};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration};

// Burn imports
use burn::{
    optim::{AdamConfig, Optimizer, GradientsParams},
    tensor::{
        backend::Backend,
        Distribution,
        Tensor,
    },
    grad_clipping::GradientClippingConfig,
};
use burn::backend::Autodiff;
use burn::backend::NdArray as BurnNdArray;

use burn::nn::loss::{MseLoss, Reduction};


// Crossterm imports
use crossterm::{
    event::{DisableMouseCapture, EnableMouseCapture, Event as CEvent, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};

// LinOSS imports - Corrected names
use linoss_rust::linoss::{
    model::{FullLinossModel, FullLinossModelConfig},
    block::LinossBlockConfig,
};

// Ratatui imports
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Style},
    symbols,
    text::{Span, Line, Text},
    widgets::{Axis, Block, Borders, Chart, Dataset, GraphType, Paragraph},
    Terminal,
    Frame,
};


type MyBaseBackend = BurnNdArray<f32>;
type MyBackend = Autodiff<MyBaseBackend>;

struct TrainingApp {
    loss_data: Vec<(f64, f64)>,
    log_buffer: Vec<String>,
    should_quit: bool,
    current_epoch: u32,
    max_epochs: u32,
    is_training_complete: bool,
    final_loss: Option<f32>,
}

impl TrainingApp {
    fn new(max_epochs: u32) -> Self {
        Self {
            loss_data: Vec::new(),
            log_buffer: Vec::new(),
            should_quit: false,
            current_epoch: 0,
            max_epochs,
            is_training_complete: false,
            final_loss: None,
        }
    }

    fn on_tick(&mut self, loss_opt: Option<f32>) {
        if let Some(loss) = loss_opt {
            if !loss.is_nan() && !loss.is_infinite() {
                 self.loss_data.push((self.current_epoch as f64, loss as f64));
            } else {
                self.log_buffer.push(format!("Epoch {}: Invalid loss value: {}", self.current_epoch, loss));
            }
        }
    }

    fn ui(f: &mut Frame, app: &TrainingApp) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .margin(1)
            .constraints(
                [
                    Constraint::Percentage(70), // For chart
                    Constraint::Percentage(30), // For logs
                ]
                .as_ref(),
            )
            .split(f.area());

        // Loss Chart
        let loss_values_f64: Vec<(f64, f64)> = app.loss_data.to_vec();
        let datasets = vec![Dataset::default()
            .name("Loss")
            .marker(symbols::Marker::Dot)
            .style(Style::default().fg(Color::Cyan))
            .graph_type(GraphType::Line)
            .data(&loss_values_f64)];

        let x_labels: Vec<Span> = app
            .loss_data
            .iter()
            .map(|(x, _)| Span::from(format!("{:.0}", x)))
            .collect::<Vec<Span>>();

        let y_max = app.loss_data.iter().map(|(_,y)| *y).fold(f64::NEG_INFINITY, f64::max);
        let y_min = app.loss_data.iter().map(|(_,y)| *y).fold(f64::INFINITY, f64::min);
        let y_diff = (y_max - y_min).abs();
        let y_padding = if y_diff < 1e-9 { 0.1 } else { y_diff * 0.1 };
        let y_bounds = if app.loss_data.is_empty() {
            [0.0, 1.0]
        } else {
            [y_min - y_padding, y_max + y_padding]
        };


        let chart = Chart::new(datasets)
            .block(
                Block::default()
                    .title("Training Loss")
                    .borders(Borders::ALL),
            )
            .x_axis(
                Axis::default()
                    .title("Epoch")
                    .style(Style::default().fg(Color::Gray))
                    .labels(x_labels)
                    .bounds([
                        app.loss_data.first().map_or(0.0, |(x, _)| *x),
                        app.loss_data.last().map_or(app.max_epochs as f64, |(x, _)| x.max(app.current_epoch as f64)),
                    ]),
            )
            .y_axis(
                Axis::default()
                    .title("Loss")
                    .style(Style::default().fg(Color::Gray))
                    .labels(
                        (0..=10)
                            .map(|i| {
                                let val = y_bounds[0] + (y_bounds[1] - y_bounds[0]) * (i as f64 / 10.0);
                                Span::from(format!("{:.2e}", val))
                            })
                            .collect::<Vec<Span>>(),
                    )
                    .bounds(y_bounds),
            );
        f.render_widget(chart, chunks[0]);

        // Log Panel
        let log_lines: Vec<Line> = app.log_buffer.iter().map(|s| Line::from(s.as_str())).collect();
        let log_text = Text::from(log_lines);
        let log_paragraph = Paragraph::new(log_text)
            .block(Block::default().title("Logs").borders(Borders::ALL))
            .wrap(ratatui::widgets::Wrap { trim: true });
        f.render_widget(log_paragraph, chunks[1]);
    }
}


fn main() -> Result<(), Box<dyn Error>> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let tui_backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(tui_backend)?;

    let max_epochs = 100;
    let mut app = TrainingApp::new(max_epochs);

    app.log_buffer.push("TUI Initialized. Starting training setup...".to_string());

    let (tx_event, rx_event) = mpsc::channel::<(u32, f32, Option<String>)>();
    let (tx_model_result, rx_model_result) = mpsc::channel::<(FullLinossModel<MyBackend>, f32)>();


    let device: <MyBaseBackend as Backend>::Device = Default::default();

    // Model configuration
    let d_input_model = 16;
    let d_model_global = 32;
    let d_output_model = 8;
    let n_layers_model = 2;

    let block_config = LinossBlockConfig {
        d_state_m: d_model_global / 2,
        d_ff: d_model_global * 2,
        delta_t: 0.1,
        init_std: 0.02,
        enable_d_feedthrough: true, // Renamed to snake_case
    };

    let model_config = FullLinossModelConfig {
        d_input: d_input_model,
        d_model: d_model_global,
        d_output: d_output_model,
        n_layers: n_layers_model,
        linoss_block_config: block_config,
    };

    let initial_model: FullLinossModel<MyBackend> = model_config.init(&device);
    let mut optimizer = AdamConfig::new()
        .with_grad_clipping(Some(GradientClippingConfig::Value(1.0)))
        .init();

    app.log_buffer.push(format!("Model Config: {:?}", model_config));
    app.log_buffer.push("Optimizer Initialized.".to_string());

    let training_thread_tx_event = tx_event.clone();
    let training_thread_tx_model_result = tx_model_result.clone();
    let training_thread_model = initial_model.clone();
    let training_thread_device = device;

    let training_handle = thread::spawn(move || {
        let mut model = training_thread_model;
        let device_clone = training_thread_device;

        for epoch in 0..max_epochs {
            let batch_size_train = 4;
            let seq_len_train = 20;

            let input_train = Tensor::<MyBackend, 3>::random(
                [batch_size_train, seq_len_train, d_input_model],
                Distribution::Uniform(0.0, 1.0),
                &device_clone,
            );
            let target_train = Tensor::<MyBackend, 3>::random(
                [batch_size_train, seq_len_train, d_output_model],
                Distribution::Uniform(0.0, 1.0),
                &device_clone,
            );

            let output_train = model.forward(input_train);
            let loss_train = MseLoss::new().forward(output_train, target_train, Reduction::Mean);
            let loss_val: f32 = loss_train.clone().into_scalar(); // Corrected: into_scalar() returns the host type directly

            if loss_val.is_nan() || loss_val.is_infinite() {
                let log_msg = format!("Epoch {}: Loss is NaN or Inf. Stopping training.", epoch);
                training_thread_tx_event.send((epoch, loss_val, Some(log_msg))).unwrap();
                return (model, loss_val);
            }

            let grads = loss_train.backward();
            let grads_params = GradientsParams::from_grads(grads, &model);
            model = optimizer.step(1e-3, model, grads_params);

            training_thread_tx_event.send((epoch, loss_val, None)).unwrap();
            thread::sleep(Duration::from_millis(50));
        }
        let final_loss_val_tensor = model.forward(Tensor::<MyBackend, 3>::random([1,1,d_input_model], Distribution::Uniform(0.0,1.0),&device_clone)).mean();
        let final_loss_val: f32 = final_loss_val_tensor.into_scalar(); // Corrected
        training_thread_tx_model_result.send((model.clone(), final_loss_val)).unwrap();
        (model, final_loss_val)
    });

    // TUI event loop
    loop {
        if app.should_quit {
            break;
        }

        // Handle events from training thread
        if let Ok((epoch, loss, log_msg_opt)) = rx_event.try_recv() {
            app.current_epoch = epoch;
            app.on_tick(Some(loss));
            if let Some(log_msg) = log_msg_opt {
                app.log_buffer.push(log_msg);
            }
            app.log_buffer.push(format!("Epoch {}: Loss = {:.4e}", epoch, loss));
            if app.log_buffer.len() > 20 { // Keep buffer size manageable
                app.log_buffer.remove(0);
            }
        }
        
        // Check if training is complete (from model result channel)
        if !app.is_training_complete {
            if let Ok((_final_model, final_loss_val)) = rx_model_result.try_recv() {
                app.is_training_complete = true;
                app.final_loss = Some(final_loss_val);
                app.log_buffer.push(format!("Training complete. Final loss: {:.4e}", final_loss_val));
                // Optionally, save the model or do other post-training tasks here
                // For this example, we just log completion.
            }
        }

        terminal.draw(|f| TrainingApp::ui(f, &app))?;

        if crossterm::event::poll(Duration::from_millis(100))? {
            if let CEvent::Key(key) = crossterm::event::read()? {
                if key.code == KeyCode::Char('q') {
                    app.should_quit = true;
                }
            }
        }
    }

    // Wait for training thread to finish if it hasn't already (e.g. if TUI quit early)
    let (_final_model_from_thread, _final_loss_from_thread) = training_handle.join().expect("Training thread panicked");
    // You might want to use/log _final_model_from_thread here if needed.

    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    if let Some(final_loss) = app.final_loss {
        println!("Training finished. Final loss: {:.4e}", final_loss);
    } else {
        println!("Training interrupted or did not complete fully.");
    }

    Ok(())
}
