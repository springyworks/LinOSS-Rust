// src/linoss/vis_utils.rs

use ratatui::{
    Frame,
    layout::{Constraint, Rect},
    style::{Color, Style},
    text::Span,
    widgets::{Block, Borders, Cell, Row, Table},
};

/// Represents data that can be visualized as a 2D grid.
/// Outer Vec is rows, inner Vec is columns.
pub type TensorVisData = Vec<Vec<f32>>;

const DISTINCT_COLORS: [Color; 16] = [
    Color::Blue,
    Color::Green,
    Color::Rgb(160, 82, 45),
    Color::Red,
    Color::Magenta,
    Color::Cyan,
    Color::Yellow,
    Color::DarkGray,
    Color::LightRed,
    Color::LightGreen,
    Color::LightBlue,
    Color::LightMagenta,
    Color::LightCyan,
    Color::Gray,
    Color::Rgb(210, 105, 30),
    Color::Rgb(85, 107, 47),
];

/// Creates a Ratatui Table widget to visualize tensor data.
///
/// # Arguments
/// * `data` - The 2D tensor data to visualize.
/// * `title` - The title for the block surrounding the table.
/// * `cell_width` - The fixed width for each cell in the table.
pub fn create_tensor_table<'a>(
    data: &'a TensorVisData,
    title: &'a str,
    cell_width: u16,
) -> Table<'a> {
    let block = Block::default().title(title).borders(Borders::ALL);

    let rows: Vec<Row> = data
        .iter()
        .map(|row_data| {
            let cells: Vec<Cell> = row_data
                .iter()
                .map(|&val| {
                    // Basic coloring based on value magnitude for demonstration
                    // More sophisticated coloring can be added.
                    let color_val = (val.abs() * 10.0) as usize % DISTINCT_COLORS.len();
                    let color = DISTINCT_COLORS[color_val];
                    let span = Span::raw(format!(
                        "[{:>width$.2}]",
                        val,
                        width = (cell_width as usize).saturating_sub(2)
                    )); // [ val ]
                    Cell::from(span).style(Style::default().bg(color))
                })
                .collect();
            Row::new(cells).height(1) // Each row in the table takes 1 character height
        })
        .collect();

    let num_cols = if data.is_empty() || data[0].is_empty() {
        0
    } else {
        data[0].len()
    };
    let constraints = vec![Constraint::Length(cell_width); num_cols];

    Table::new(rows, &constraints)
        .block(block)
        .column_spacing(0) // No extra spacing between columns
}

/// A helper function to render a tensor table in a given area of the frame.
pub fn render_tensor_table(
    frame: &mut Frame, // Removed <B: Backend> and changed Frame<B> to Frame
    data: &TensorVisData,
    title: &str,
    area: Rect,
    cell_width: u16,
) {
    if data.is_empty() || data[0].is_empty() {
        let placeholder_block = Block::default().title(title).borders(Borders::ALL);
        frame.render_widget(placeholder_block, area);
        // Optionally render a "No data" message inside
        return;
    }

    let table = create_tensor_table(data, title, cell_width);
    frame.render_widget(table, area);
}

// Future additions could include:
// - Functions to draw multiple tensors.
// - Functions to draw model architecture.
// - Helpers for layout management.
// - More sophisticated coloring or cell formatting.

// --- Tokio-based TUI Application Runner ---

use crossterm::{
    event::{self as crossterm_event, Event as CrosstermEvent, KeyEvent, poll}, // Added poll
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{
    Terminal,
    prelude::CrosstermBackend, // Added
};
use std::{io, time::Duration as StdDuration};
use tokio::{sync::mpsc, time::interval};

/// Events that the TUI application loop will handle.
#[derive(Debug)]
pub enum AppEvent {
    Input(KeyEvent),
    Tick,
    Resize(u16, u16),
}

/// Trait for the application logic in a Tokio-based TUI.
pub trait TokioTuiApp {
    /// Draw the UI.
    fn draw_ui(&mut self, frame: &mut Frame); // Changed Frame<B> to Frame

    /// Update state based on an event. Return true to continue, false to quit.
    fn update(&mut self, event: AppEvent) -> bool;

    /// Optional: handle terminal resize explicitly if needed beyond AppEvent::Resize.
    /// This is called once initially and on every AppEvent::Resize.
    fn on_resize_internal(&mut self, width: u16, height: u16);
}

/// Runs a TUI application with a Tokio-based event loop.
///
/// # Arguments
/// * `app` - An instance of a type implementing `TokioTuiApp`.
/// * `tick_rate` - The duration between `AppEvent::Tick` events.
/// * `input_poll_rate` - How often to poll for input events.
pub async fn run_tokio_tui_app<A: TokioTuiApp + Send + 'static>(
    mut app: A,
    tick_rate: StdDuration,
    input_poll_rate: StdDuration,
) -> io::Result<()> {
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    terminal.clear()?;

    let (event_tx, mut event_rx) = mpsc::channel(100); // Channel for app events

    // Initial resize
    let initial_size = terminal.size()?;
    app.on_resize_internal(initial_size.width, initial_size.height);
    // Send initial resize event for consistency if app wants to handle it in update
    let _ = event_tx
        .send(AppEvent::Resize(initial_size.width, initial_size.height))
        .await;

    // Input handling task
    let input_event_tx = event_tx.clone();
    tokio::spawn(async move {
        loop {
            if poll(input_poll_rate).unwrap_or(false) {
                match crossterm_event::read() {
                    Ok(CrosstermEvent::Key(key_event)) => {
                        if input_event_tx
                            .send(AppEvent::Input(key_event))
                            .await
                            .is_err()
                        {
                            break;
                        }
                    }
                    Ok(CrosstermEvent::Resize(width, height)) => {
                        if input_event_tx
                            .send(AppEvent::Resize(width, height))
                            .await
                            .is_err()
                        {
                            break;
                        }
                    }
                    Err(_) => { /* Handle read error if necessary */ }
                    _ => {} // Ignore other events like mouse, focus, etc. for now
                }
            }
            // Yield to allow other tasks to run, especially if poll rate is very low
            tokio::task::yield_now().await;
        }
    });

    // Tick generation task
    let tick_event_tx = event_tx;
    tokio::spawn(async move {
        let mut ticker = interval(tick_rate);
        loop {
            ticker.tick().await;
            if tick_event_tx.send(AppEvent::Tick).await.is_err() {
                break;
            }
        }
    });

    // Main application loop
    loop {
        // Draw UI before waiting for an event
        terminal.draw(|f: &mut Frame| app.draw_ui(f))?; // Added type annotation for f

        match event_rx.recv().await {
            Some(app_event) => {
                if let AppEvent::Resize(width, height) = app_event {
                    terminal.resize(Rect::new(0, 0, width, height))?; // Ensure terminal itself knows about resize
                    app.on_resize_internal(width, height);
                    // The event is also passed to app.update for general handling
                }
                if !app.update(app_event) {
                    break; // App requested quit
                }
            }
            None => break, // Channel closed
        }
    }

    // Restore terminal
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    Ok(())
}
