use ndarray::Array2;
use ratatui::{
    backend::CrosstermBackend,
    widgets::{Block, Borders, Table, Row, Paragraph, Clear, Cell},
    layout::{Layout, Constraint, Direction, Rect},
    style::{Style, Color},
    text::Span,
    Terminal,
};
use crossterm::{
    event::{self, Event, MouseEvent, MouseEventKind, KeyCode, EnableMouseCapture, DisableMouseCapture, poll},
    execute,
    terminal::{self, EnterAlternateScreen, LeaveAlternateScreen},
};
use std::io;
use rand::prelude::*; // Provides Rng, Distribution, etc.
use rand::distributions::Uniform; // Fixed import path
use std::time::{Duration, Instant}; // For timed updates

const NUM_MATRICES: usize = 5;
const MATRIX_ROWS: usize = 5;
const MATRIX_COLS: usize = 5;
const SPACING_BETWEEN_MATRIX_ROWS: u16 = 1; // For wrapped rows

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut matrices: Vec<Array2<i32>> = Vec::new();
    for i in 0..NUM_MATRICES {
        let start_val = (i * MATRIX_ROWS * MATRIX_COLS) as i32 + 1;
        let end_val = start_val + (MATRIX_ROWS * MATRIX_COLS) as i32 - 1;
        let values: Vec<i32> = (start_val..=end_val).collect();
        matrices.push(Array2::from_shape_vec((MATRIX_ROWS, MATRIX_COLS), values)?);
    }

    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    terminal::enable_raw_mode()?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    terminal.clear()?;

    let mut popup_message: Option<String> = None;
    let mut last_matrix_areas: Vec<Option<Rect>> = vec![None; NUM_MATRICES];

    let mut last_update_time = Instant::now();
    let update_interval = Duration::from_secs(2); // Update every 2 seconds
    let mut quit_app = false; // Declare quit_app here

    loop {
        let mut current_matrix_areas_for_draw: Vec<Option<Rect>> = vec![None; NUM_MATRICES];

        terminal.draw(|frame| {
            let size = frame.area(); 
            
            const DISTINCT_COLORS: [Color; 16] = [
                Color::Blue, Color::Green, Color::Rgb(160, 82, 45), Color::Red,
                Color::Magenta, Color::Cyan, Color::Yellow, Color::DarkGray,
                Color::LightRed, Color::LightGreen, Color::LightBlue, Color::LightMagenta,
                Color::LightCyan, Color::Gray, Color::Rgb(210, 105, 30), Color::Rgb(85, 107, 47),  
            ];

            fn create_matrix_table<'a>(matrix_data: &Array2<i32>, title: &'a str) -> Table<'a> {
                let matrix_block = Block::default().title(title).borders(Borders::ALL);
                let rows = matrix_data.rows().into_iter().map(|row_data| {
                    let cells = row_data.iter().map(|&val| {
                        let color_index = ((val - 1).abs() as usize) % DISTINCT_COLORS.len();
                        let color = DISTINCT_COLORS[color_index];
                        let span = Span::raw(format!("[{:>3}]", val));
                        Cell::from(span).style(Style::default().bg(color))
                    }).collect::<Vec<_>>();
                    Row::new(cells).height(1)
                });
                Table::new(rows, &[Constraint::Length(5); MATRIX_COLS])
                    .block(matrix_block)
                    .column_spacing(0)
            }

            let matrix_content_width = (MATRIX_COLS * 5) as u16; 
            let matrix_block_width = matrix_content_width + 2; 
            let matrix_height = MATRIX_ROWS as u16 + 2;      
            let spacing_between_matrices_horizontally = 1u16;

            let available_width_for_matrices = size.width;
            let matrices_per_row = ((available_width_for_matrices.saturating_add(spacing_between_matrices_horizontally)) 
                / (matrix_block_width.saturating_add(spacing_between_matrices_horizontally)))
                .max(1) as usize;
            
            let num_matrix_rows_needed = ((NUM_MATRICES as f32 / matrices_per_row as f32).ceil() as usize).max(1);

            let mut vertical_constraints = Vec::new();
            let constraints_per_visual_row = 1 + (if SPACING_BETWEEN_MATRIX_ROWS > 0 {1} else {0});

            for i in 0..num_matrix_rows_needed {
                vertical_constraints.push(Constraint::Length(matrix_height));
                if i < num_matrix_rows_needed - 1 {
                    vertical_constraints.push(Constraint::Length(SPACING_BETWEEN_MATRIX_ROWS));
                }
            }
            vertical_constraints.push(Constraint::Min(0)); 

            let main_vertical_chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints(vertical_constraints)
                .split(size);

            let mut matrix_render_idx = 0;
            for row_num in 0..num_matrix_rows_needed {
                if matrix_render_idx >= NUM_MATRICES {
                    break;
                }

                let current_row_rect_idx = row_num * constraints_per_visual_row; 
                if current_row_rect_idx >= main_vertical_chunks.len() {
                    break; 
                }
                let current_row_render_area = main_vertical_chunks[current_row_rect_idx];

                let matrices_in_this_row = matrices_per_row.min(NUM_MATRICES - matrix_render_idx);
                if matrices_in_this_row == 0 { 
                    continue;
                }

                let mut horizontal_constraints_for_row = Vec::new();
                for i in 0..matrices_in_this_row {
                    horizontal_constraints_for_row.push(Constraint::Length(matrix_block_width));
                    if i < matrices_in_this_row - 1 { 
                        horizontal_constraints_for_row.push(Constraint::Length(spacing_between_matrices_horizontally));
                    }
                }

                let required_width_for_this_row = (matrix_block_width * matrices_in_this_row as u16) 
                    + (spacing_between_matrices_horizontally * (matrices_in_this_row.saturating_sub(1)) as u16);

                if required_width_for_this_row < current_row_render_area.width {
                    horizontal_constraints_for_row.push(Constraint::Min(0)); 
                }
                
                let horizontal_row_chunks = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints(horizontal_constraints_for_row) 
                    .split(current_row_render_area);

                let mut current_chunk_idx_in_row = 0; 

                for _col_num_in_row in 0..matrices_in_this_row { 
                    if matrix_render_idx >= NUM_MATRICES || current_chunk_idx_in_row >= horizontal_row_chunks.len() {
                        break;
                    }
                    let matrix_rect = horizontal_row_chunks[current_chunk_idx_in_row];
                    current_matrix_areas_for_draw[matrix_render_idx] = Some(matrix_rect);
                    let title_string = format!("Matrix {}", matrix_render_idx + 1);

                    if matrix_rect.width < matrix_block_width || matrix_rect.height < matrix_height {
                        let placeholder_block = Block::default().title(title_string).borders(Borders::ALL);
                        frame.render_widget(placeholder_block.clone(), matrix_rect); 
                        let inner_area = placeholder_block.inner(matrix_rect);
                        if inner_area.width > 0 && inner_area.height > 0 {
                           let message = Paragraph::new("Too small")
                               .alignment(ratatui::layout::Alignment::Center)
                               .wrap(ratatui::widgets::Wrap { trim: true });
                           frame.render_widget(message, inner_area);
                        }
                    } else {
                        let table = create_matrix_table(&matrices[matrix_render_idx], &title_string);
                        frame.render_widget(table, matrix_rect);
                    }
                    
                    matrix_render_idx += 1;
                    current_chunk_idx_in_row += 2; 
                }
            }
            
            if let Some(ref msg) = popup_message {
                let area = frame.area(); 
                let popup_layout = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([
                        Constraint::Percentage(30), Constraint::Percentage(20), Constraint::Percentage(50),
                    ].as_ref()).split(area);
                let popup_area = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints([
                        Constraint::Percentage(30), Constraint::Percentage(40), Constraint::Percentage(30),
                    ].as_ref()).split(popup_layout[1])[1];
                let popup_block = Block::default().title("Popup!").borders(Borders::ALL).style(Style::default().bg(Color::DarkGray));
                let popup_text_widget = Paragraph::new(msg.clone()).block(popup_block).wrap(ratatui::widgets::Wrap { trim: true });
                frame.render_widget(Clear, popup_area); 
                frame.render_widget(popup_text_widget, popup_area);
            }
        })?;

        for i in 0..NUM_MATRICES {
            if let Some(area) = current_matrix_areas_for_draw[i] {
                last_matrix_areas[i] = Some(area);
            }
        }

        if last_update_time.elapsed() >= update_interval {
            if !matrices.is_empty() {
                // Example: Randomly pick a matrix and change one of its values
                let mut rng = rand::thread_rng(); // Fixed: use thread_rng()
                let matrix_idx_to_change = rng.gen_range(0..matrices.len());
                let row_idx = rng.gen_range(0..MATRIX_ROWS);
                let col_idx = rng.gen_range(0..MATRIX_COLS);
                matrices[matrix_idx_to_change][(row_idx, col_idx)] = rng.gen_range(0..100);
                last_update_time = Instant::now(); 
            }
        }

        if poll(Duration::from_millis(100))? { 
            let event_read = event::read()?; 

            match event_read {
                Event::Mouse(MouseEvent { kind, column, row, .. }) => {
                    if let MouseEventKind::Down(_button) = kind {
                        if popup_message.is_some() {
                            popup_message = None; 
                        } else {
                            let mut clicked_on_matrix: Option<(usize, usize, usize, i32)> = None; 

                            for matrix_idx in 0..NUM_MATRICES {
                                if let Some(matrix_area_for_click) = last_matrix_areas[matrix_idx] {
                                    if column >= matrix_area_for_click.x && column < matrix_area_for_click.x + matrix_area_for_click.width &&
                                       row >= matrix_area_for_click.y && row < matrix_area_for_click.y + matrix_area_for_click.height {
                                        
                                        let table_content_start_col_in_area = 1u16;
                                        let table_content_start_row_in_area = 1u16;
                                        let relative_col_to_matrix_area = column - matrix_area_for_click.x;
                                        let relative_row_to_matrix_area = row - matrix_area_for_click.y;

                                        if relative_col_to_matrix_area >= table_content_start_col_in_area && 
                                           relative_row_to_matrix_area >= table_content_start_row_in_area {

                                            let col_inside_content = relative_col_to_matrix_area - table_content_start_col_in_area;
                                            let row_inside_content = relative_row_to_matrix_area - table_content_start_row_in_area;
                                            
                                            let cell_width_actual = 5u16; 
                                            let cell_height_actual = 1u16;

                                            let num_cols_in_matrix = MATRIX_COLS as u16;
                                            let num_rows_in_matrix = MATRIX_ROWS as u16;

                                            let table_content_width_actual = num_cols_in_matrix * cell_width_actual;
                                            let table_content_height_actual = num_rows_in_matrix * cell_height_actual;

                                            if col_inside_content < table_content_width_actual && row_inside_content < table_content_height_actual {
                                                let cell_x = (col_inside_content / cell_width_actual) as usize;
                                                let cell_y = (row_inside_content / cell_height_actual) as usize;
                                                
                                                if cell_x < MATRIX_COLS && cell_y < MATRIX_ROWS {
                                                    let value = matrices[matrix_idx][[cell_y, cell_x]];
                                                    clicked_on_matrix = Some((matrix_idx + 1, cell_y, cell_x, value)); 
                                                    break; 
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            if let Some((matrix_display_idx, c_y, c_x, val)) = clicked_on_matrix {
                                let msg_content = format!("Matrix {}, Cell ({}, {}) clicked!\nValue: {}", matrix_display_idx, c_y, c_x, val);
                                popup_message = Some(msg_content);
                            } else {
                                if popup_message.is_some() { popup_message = None; }
                            }
                        }
                    }
                }
                Event::Key(key) => {
                    if key.code == KeyCode::Esc || key.code == KeyCode::Char('q') {
                        if popup_message.is_some() {
                            popup_message = None;
                        } else {
                            quit_app = true;
                        }
                    }

                    if key.code == KeyCode::Char('u') {
                        if !matrices.is_empty() {
                            let mut rng = rand::thread_rng(); // Fixed: use thread_rng()
                            let value_dist = Uniform::new(0, 100); // Fixed: remove unwrap()
                            for r_idx in 0..MATRIX_ROWS {
                                for c_idx in 0..MATRIX_COLS {
                                    matrices[0][(r_idx, c_idx)] = rng.sample(value_dist);
                                }
                            }
                            popup_message = Some("Matrix 0 manually updated!".to_string());
                        }
                    }

                    if popup_message.is_some() && (key.code != KeyCode::Char('u') || quit_app) {
                        popup_message = None;
                        if quit_app {
                            break;
                        }
                    } else if quit_app {
                        break;
                    }
                }
                _ => {}
            }
        }
    }

    terminal::disable_raw_mode()?;
    execute!(io::stdout(), LeaveAlternateScreen, DisableMouseCapture)?;
    Ok(())
}