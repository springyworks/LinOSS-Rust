// examples/flyLinoss.rs

use linoss_rust::linoss::vis_utils::{render_tensor_table, TensorVisData};
use std::{
    io,
    time::{Duration, Instant},
};
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event as CEvent, KeyCode, poll},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    Terminal,
};
use log::{info}; // Removed unused error import

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    info!("Starting flyLinoss example...");

    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    terminal.clear()?;

    // Sample data for visualization
    let sample_tensor_a: TensorVisData = vec![
        vec![1.0, 2.34, 0.55, -1.8],
        vec![-0.5, 1.1, 2.0, 0.0],
        vec![0.75, -1.5, 0.2, 1.23],
    ];
    let sample_tensor_b: TensorVisData = vec![
        vec![10.0, 20.0],
        vec![30.0, 40.0],
    ];

    let tick_rate = Duration::from_millis(250);
    let mut last_tick = Instant::now();

    loop {
        terminal.draw(|frame| {
            let size = frame.area(); // Changed from frame.size() to frame.area()
            if size.width == 0 || size.height == 0 {
                // Terminal too small or not yet initialized
                return;
            }
            
            // Create a layout with two horizontal chunks
            let chunks = Layout::default()
                .direction(Direction::Horizontal)
                .margin(1)
                .constraints(
                    [
                        Constraint::Percentage(50), // Left chunk for Tensor A
                        Constraint::Percentage(50), // Right chunk for Tensor B
                    ]
                    .as_ref(),
                )
                .split(size);

            // Render Tensor A in the left chunk
            if chunks[0].width > 0 && chunks[0].height > 0 {
                 render_tensor_table(frame, &sample_tensor_a, "Tensor A (Sample)", chunks[0], 10);
            }

            // Render Tensor B in the right chunk
            if chunks[1].width > 0 && chunks[1].height > 0 {
                render_tensor_table(frame, &sample_tensor_b, "Tensor B (Sample)", chunks[1], 10);
            }
        })?;

        let timeout = tick_rate
            .checked_sub(last_tick.elapsed())
            .unwrap_or_else(|| Duration::from_secs(0));

        if poll(timeout)? {
            if let CEvent::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => {
                        info!("Quitting flyLinoss example...");
                        break;
                    }
                    _ => {}
                }
            }
        }

        if last_tick.elapsed() >= tick_rate {
            // Here you could update tensor data if it were dynamic
            last_tick = Instant::now();
        }
    }

    // Restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;
    info!("flyLinoss example finished.");
    Ok(())
}
