use crossterm::{
    event::{KeyCode, KeyEventKind}, // Removed Event, poll
    // execute, // Will be handled by vis_utils
    // terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen}, // Handled by vis_utils
};
use ratatui::{
    // backend::{Backend, CrosstermBackend}, // Backend handling moved
    style::{Color, Style},
    symbols,
    widgets::{canvas::Canvas, Block, Borders, Paragraph},
    // Terminal, // Terminal handling moved
    layout::Rect,
    Frame, // Added for TokioTuiApp trait
};
use std::{
    io::{self},
    time::{Duration, Instant},
};
use rand::Rng; // Added back for random number generation

// Import the new Tokio TUI runner and related items
use linoss_rust::linoss::vis_utils::{run_tokio_tui_app, AppEvent, TokioTuiApp};


const MAX_LINE_RADIUS_DOTS: f64 = 20.0; // Renamed from LINE_LENGTH_CELLS
const MIN_LINE_RADIUS_DOTS: f64 = 4.0;  // Minimum radius for the line in dots
const MIN_VISUAL_HORIZONTAL_DOT_EXTENT: f64 = 2.0; // For vertical line visibility (1 Braille cell width)

const BLINK_DURATION: Duration = Duration::from_millis(750);
const BLINK_VISIBILITY_INTERVAL_MS: u128 = 150; // Controls blink speed
const OVERFLOW_CHANGE_EPSILON: f64 = 0.1; // Min change in overflow to trigger blink
const HORIZONTAL_TARGET_UPDATE_INTERVAL: Duration = Duration::from_secs(3); // How often to pick a new horizontal target
const HORIZONTAL_MOVEMENT_SPEED: f64 = 0.02; // Proportion of distance to target to move each frame

struct AppState {
    center_x: f64, // In cell coordinates
    center_y: f64, // In cell coordinates
    angle_degrees: f64,
    vertical_direction: i16,
    last_animation_update: Instant, // Renamed from last_update for clarity
    terminal_width: u16, // In cells
    terminal_height: u16, // In cells

    // Dynamic line properties
    current_line_radius_dots: f64,

    // Horizontal movement
    horizontal_target_x: f64,
    last_horizontal_target_update: Instant,

    // For clipping indicators and blinking
    app_start_time: Instant,
    last_overflow_top: f64,
    last_overflow_bottom: f64,
    last_overflow_left: f64,
    last_overflow_right: f64,
    blink_top_until: Option<Instant>,
    blink_bottom_until: Option<Instant>,
    blink_left_until: Option<Instant>,
    blink_right_until: Option<Instant>,
}

impl AppState {
    fn new() -> Self { // Removed width, height args, will be set by on_resize_internal
        let now = Instant::now();
        Self {
            center_x: 0.0, // Initialized by on_resize_internal
            center_y: 0.0, // Initialized by on_resize_internal
            angle_degrees: 0.0,
            vertical_direction: 1,
            last_animation_update: now,
            terminal_width: 0, // Initialized by on_resize_internal
            terminal_height: 0, // Initialized by on_resize_internal
            current_line_radius_dots: MAX_LINE_RADIUS_DOTS, // Initial value, updated in on_resize
            horizontal_target_x: 0.0, // Will be set in on_resize_internal
            last_horizontal_target_update: now,
            app_start_time: now, // Keep app_start_time here
            last_overflow_top: 0.0,
            last_overflow_bottom: 0.0,
            last_overflow_left: 0.0,
            last_overflow_right: 0.0,
            blink_top_until: None,
            blink_bottom_until: None,
            blink_left_until: None,
            blink_right_until: None,
        }
    }

    fn update_dynamic_line_properties(&mut self) {
        if self.terminal_width == 0 || self.terminal_height == 0 {
            self.current_line_radius_dots = MIN_LINE_RADIUS_DOTS;
            return;
        }
        // Estimate canvas size (terminal size minus borders)
        let canvas_width_cells_est = (self.terminal_width.saturating_sub(2)).max(1) as f64;
        let canvas_height_cells_est = (self.terminal_height.saturating_sub(2)).max(1) as f64;

        let canvas_width_dots_est = canvas_width_cells_est * 2.0;
        let canvas_height_dots_est = canvas_height_cells_est * 4.0;

        self.current_line_radius_dots = MAX_LINE_RADIUS_DOTS
            .min(canvas_width_dots_est * 0.45) // Try to fit within 90% of canvas dim (radius is 0.45 * dim)
            .min(canvas_height_dots_est * 0.45)
            .max(MIN_LINE_RADIUS_DOTS);
    }

    fn update_line_animation(&mut self) {
        self.update_dynamic_line_properties(); // Update line size based on current terminal dims

        let safe_margin_for_center_x = self.current_line_radius_dots / 2.0 + 1.0; // Max x-extent of line in cells + buffer
        let safe_margin_for_center_y = self.current_line_radius_dots / 4.0 + 1.0; // Max y-extent of line in cells + buffer

        if self.last_animation_update.elapsed() >= Duration::from_millis(50) {
            self.angle_degrees = (self.angle_degrees + 5.0) % 360.0;

            // Vertical movement
            let vertical_speed = 0.5; // cells per update
            if self.terminal_height > 0 { // Avoid issues if not yet initialized
                // Pin to center if the available movement space is less than one step,
                // or if the terminal is too small to even define clear top/bottom margins for the line.
                let min_height_for_movement = 2.0 * safe_margin_for_center_y + vertical_speed;

                if (self.terminal_height as f64) < min_height_for_movement {
                    self.center_y = self.terminal_height as f64 / 2.0;
                    // When pinned, ensure vertical_direction doesn't cause an immediate attempt to move
                    // if the size slightly changes later. A simple way is to set it based on current position
                    // relative to the (new) center, or just let it be. Pinning is the key.
                } else {
                    self.center_y += vertical_speed * self.vertical_direction as f64;
                    // Ensure center_y stays within bounds defined by safe_margin_for_center_y
                    if self.center_y >= self.terminal_height as f64 - safe_margin_for_center_y {
                        self.center_y = self.terminal_height as f64 - safe_margin_for_center_y;
                        self.vertical_direction = -1;
                    } else if self.center_y <= safe_margin_for_center_y {
                        self.center_y = safe_margin_for_center_y;
                        self.vertical_direction = 1;
                    }
                }
            }

            // Horizontal movement
            if self.terminal_width > 0 { // Avoid issues if not yet initialized
                if self.last_horizontal_target_update.elapsed() >= HORIZONTAL_TARGET_UPDATE_INTERVAL {
                    let mut rng = rand::thread_rng(); // Fixed: use thread_rng()
                    // Target range for center_x, ensuring it respects the safe_margins
                    let min_target_x = safe_margin_for_center_x.max(self.terminal_width as f64 * 0.25);
                    let max_target_x = (self.terminal_width as f64 - safe_margin_for_center_x).min(self.terminal_width as f64 * 0.75);
                    if min_target_x < max_target_x { // Ensure valid range
                        self.horizontal_target_x = rng.gen_range(min_target_x..max_target_x); // Fixed: use gen_range
                    }
                    self.last_horizontal_target_update = Instant::now();
                }

                // Smoothly move towards the target
                let dx = self.horizontal_target_x - self.center_x;
                self.center_x += dx * HORIZONTAL_MOVEMENT_SPEED;

                // Clamp center_x to prevent it from going too close to edges
                self.center_x = self.center_x
                    .max(safe_margin_for_center_x)
                    .min(self.terminal_width as f64 - safe_margin_for_center_x);
            }

            self.last_animation_update = Instant::now();
        }
    }

    // New method to update clipping status and trigger blinks
    fn update_clipping_state_and_blinks(
        &mut self,
        x1_draw: f64, y1_draw: f64, x2_draw: f64, y2_draw: f64, // Line endpoints in canvas draw coordinates
        x_bound_max: f64, y_bound_max: f64, // Canvas bounds (0 to max)
    ) -> (bool, bool, bool, bool) { // Returns current is_clipped status for each side
    
        let current_overflow_left = (0.0 - x1_draw).max(0.0 - x2_draw).max(0.0);
        let current_overflow_right = (x1_draw - x_bound_max).max(x2_draw - x_bound_max).max(0.0);
        let current_overflow_bottom = (0.0 - y1_draw).max(0.0 - y2_draw).max(0.0);
        let current_overflow_top = (y1_draw - y_bound_max).max(y2_draw - y_bound_max).max(0.0);

        if current_overflow_top > 0.0 && 
           (self.last_overflow_top == 0.0 || (current_overflow_top - self.last_overflow_top).abs() > OVERFLOW_CHANGE_EPSILON) {
            self.blink_top_until = Some(Instant::now() + BLINK_DURATION);
        }
        if current_overflow_bottom > 0.0 &&
           (self.last_overflow_bottom == 0.0 || (current_overflow_bottom - self.last_overflow_bottom).abs() > OVERFLOW_CHANGE_EPSILON) {
            self.blink_bottom_until = Some(Instant::now() + BLINK_DURATION);
        }
        if current_overflow_left > 0.0 &&
           (self.last_overflow_left == 0.0 || (current_overflow_left - self.last_overflow_left).abs() > OVERFLOW_CHANGE_EPSILON) {
            self.blink_left_until = Some(Instant::now() + BLINK_DURATION);
        }
        if current_overflow_right > 0.0 &&
           (self.last_overflow_right == 0.0 || (current_overflow_right - self.last_overflow_right).abs() > OVERFLOW_CHANGE_EPSILON) {
            self.blink_right_until = Some(Instant::now() + BLINK_DURATION);
        }

        self.last_overflow_top = current_overflow_top;
        self.last_overflow_bottom = current_overflow_bottom;
        self.last_overflow_left = current_overflow_left;
        self.last_overflow_right = current_overflow_right;

        (
            current_overflow_top > 0.0,
            current_overflow_bottom > 0.0,
            current_overflow_left > 0.0,
            current_overflow_right > 0.0,
        )
    }
}

// Implement TokioTuiApp for AppState
impl TokioTuiApp for AppState {
    fn on_resize_internal(&mut self, width: u16, height: u16) {
        let old_width = self.terminal_width;
        self.terminal_width = width;
        self.terminal_height = height;

        self.update_dynamic_line_properties(); // Update line size first
        
        let safe_margin_for_center_x = self.current_line_radius_dots / 2.0 + 1.0;
        let safe_margin_for_center_y = self.current_line_radius_dots / 4.0 + 1.0;
        
        // Recalculate vertical center and clamp
        if (height as f64) < 2.0 * safe_margin_for_center_y {
            self.center_y = height as f64 / 2.0;
        } else {
            if self.center_y == 0.0 { // First time initialization or coming from a very small height
                self.center_y = height as f64 / 2.0;
            } else {
                self.center_y = self.center_y
                    .max(safe_margin_for_center_y)
                    .min(height as f64 - safe_margin_for_center_y);
            }
        }

        // Recalculate horizontal center and target
        if old_width == 0 { // First time initialization
            self.center_x = width as f64 / 2.0;
            let mut rng = rand::thread_rng(); // Fixed: use thread_rng()
            let min_target_x = safe_margin_for_center_x.max(width as f64 * 0.25);
            let max_target_x = (width as f64 - safe_margin_for_center_x).min(width as f64 * 0.75);
            if min_target_x < max_target_x {
                 self.horizontal_target_x = rng.gen_range(min_target_x..max_target_x); // Fixed: use gen_range
            } else {
                self.horizontal_target_x = width as f64 / 2.0; // Default to center if range is invalid
            }
        } else {
            // Adjust existing center_x and horizontal_target_x proportionally to the new width
            let width_ratio = width as f64 / old_width as f64;
            self.center_x *= width_ratio;
            self.horizontal_target_x *= width_ratio;
        }
        // Clamp center_x and horizontal_target_x to new bounds
        self.center_x = self.center_x
            .max(safe_margin_for_center_x)
            .min(width as f64 - safe_margin_for_center_x);
        
        let min_target_x_clamped = safe_margin_for_center_x.max(width as f64 * 0.15); // Wider range for target after resize
        let max_target_x_clamped = (width as f64 - safe_margin_for_center_x).min(width as f64 * 0.85);
        
        if min_target_x_clamped < max_target_x_clamped {
            self.horizontal_target_x = self.horizontal_target_x
                .max(min_target_x_clamped)
                .min(max_target_x_clamped);
        } else {
             self.horizontal_target_x = width as f64 / 2.0; // Default if range invalid
        }
        self.last_horizontal_target_update = Instant::now(); // Reset target update timer
    }

    fn update(&mut self, event: AppEvent) -> bool {
        match event {
            AppEvent::Input(key_event) => {
                if key_event.kind == KeyEventKind::Press {
                    match key_event.code {
                        KeyCode::Char('q') | KeyCode::Esc => return false, // Request quit
                        _ => {}
                    }
                }
            }
            AppEvent::Tick => {
                self.update_line_animation();
            }
            AppEvent::Resize(_, _) => {
                // Resize is handled by on_resize_internal, but AppEvent::Resize is still passed
                // here if any additional general logic is needed on resize.
                // For this app, on_resize_internal covers it.
            }
        }
        true // Continue running
    }

    fn draw_ui(&mut self, frame: &mut Frame) { // Changed Frame<B> to Frame
        let size = frame.area();
        if size.width == 0 || size.height == 0 {
            return;
        }

        let canvas_block = Block::default().title("Async Animated Line (Fixed Length)").borders(Borders::ALL);
        frame.render_widget(canvas_block, size);
        let canvas_area = Block::default().borders(Borders::ALL).inner(size);

        // Calculate line properties and canvas bounds regardless of canvas_area size
        let angle_rad = self.angle_degrees.to_radians();
        let half_len_dots = self.current_line_radius_dots; 

        let x_offset_dots = half_len_dots * angle_rad.cos();
        let y_offset_dots = half_len_dots * angle_rad.sin();

        let mut x_offset_dots_adjusted = x_offset_dots;
        if x_offset_dots.abs() < MIN_VISUAL_HORIZONTAL_DOT_EXTENT {
            if x_offset_dots == 0.0 {
                x_offset_dots_adjusted = MIN_VISUAL_HORIZONTAL_DOT_EXTENT;
            } else {
                x_offset_dots_adjusted = x_offset_dots.signum() * MIN_VISUAL_HORIZONTAL_DOT_EXTENT.max(x_offset_dots.abs());
            }
        }

        let center_x_dots_global = self.center_x * 2.0; 
        let center_y_dots_global = self.center_y * 4.0;

        let x_start_dots_global = center_x_dots_global - x_offset_dots_adjusted;
        let y_start_dots_global = center_y_dots_global - y_offset_dots;
        let x_end_dots_global = center_x_dots_global + x_offset_dots_adjusted;
        let y_end_dots_global = center_y_dots_global + y_offset_dots;
        
        let line_color = Color::Cyan;
        let indicator_color = Color::Red;

        // These bounds can be 0.0 if canvas_area is 0-width or 0-height
        let x_bounds_max_braille = canvas_area.width as f64 * 2.0;
        let y_bounds_max_braille = canvas_area.height as f64 * 4.0;

        let canvas_origin_x_dots = canvas_area.x as f64 * 2.0;
        let canvas_origin_y_dots = canvas_area.y as f64 * 4.0;

        let x1_rel_dots = x_start_dots_global - canvas_origin_x_dots;
        let y1_rel_dots_from_top = y_start_dots_global - canvas_origin_y_dots;
        let x2_rel_dots = x_end_dots_global - canvas_origin_x_dots;
        let y2_rel_dots_from_top = y_end_dots_global - canvas_origin_y_dots;

        let x1_draw = x1_rel_dots;
        let y1_draw = y_bounds_max_braille - y1_rel_dots_from_top;
        let x2_draw = x2_rel_dots;
        let y2_draw = y_bounds_max_braille - y2_rel_dots_from_top;

        // Update clipping state based on calculated line endpoints and canvas bounds
        let (is_clipped_top, is_clipped_bottom, is_clipped_left, is_clipped_right) = 
            self.update_clipping_state_and_blinks(
                x1_draw, y1_draw, x2_draw, y2_draw,
                x_bounds_max_braille, y_bounds_max_braille
            );

        // Only render the canvas widget (the line) if the canvas_area is drawable
        if canvas_area.width > 0 && canvas_area.height > 0 {
            let canvas_widget = Canvas::default()
                .block(Block::default()) 
                .marker(symbols::Marker::Braille) 
                .paint(move |ctx| {
                    ctx.draw(&ratatui::widgets::canvas::Line {
                        x1: x1_draw, 
                        y1: y1_draw,    
                        x2: x2_draw,
                        y2: y2_draw,
                        color: line_color,
                    });
                })
                .x_bounds([0.0, x_bounds_max_braille])
                .y_bounds([0.0, y_bounds_max_braille]);

            frame.render_widget(canvas_widget, canvas_area);
        }
        // If canvas_area is 0x0, the line is not drawn, but indicators below will still be processed.

        let now = Instant::now();
        let time_since_app_start_ms = now.duration_since(self.app_start_time).as_millis();

        // Changed condition to allow indicators on 1-cell wide/high terminals
        if size.width >= 1 && size.height >= 1 {
            if is_clipped_top {
                let mut show_arrow = true;
                if let Some(blink_end_time) = self.blink_top_until {
                    if now < blink_end_time {
                        if (time_since_app_start_ms / BLINK_VISIBILITY_INTERVAL_MS) % 2 != 0 { show_arrow = false; }
                    } else {
                        self.blink_top_until = None;
                    }
                }
                if show_arrow {
                    let indicator_area = Rect { x: size.x + size.width / 2, y: size.y, width: 1, height: 1 };
                    if indicator_area.x < size.x + size.width {
                        frame.render_widget(Paragraph::new("▲").style(Style::default().fg(indicator_color)), indicator_area);
                    }
                }
            }
            if is_clipped_bottom {
                let mut show_arrow = true;
                if let Some(blink_end_time) = self.blink_bottom_until {
                    if now < blink_end_time {
                        if (time_since_app_start_ms / BLINK_VISIBILITY_INTERVAL_MS) % 2 != 0 { show_arrow = false; }
                    } else {
                        self.blink_bottom_until = None;
                    }
                }
                if show_arrow {
                    let indicator_area = Rect { x: size.x + size.width / 2, y: size.y + size.height - 1, width: 1, height: 1 };
                    if indicator_area.x < size.x + size.width && indicator_area.y < size.y + size.height {
                        frame.render_widget(Paragraph::new("▼").style(Style::default().fg(indicator_color)), indicator_area);
                    }
                }
            }
            if is_clipped_left {
                let mut show_arrow = true;
                if let Some(blink_end_time) = self.blink_left_until {
                    if now < blink_end_time {
                        if (time_since_app_start_ms / BLINK_VISIBILITY_INTERVAL_MS) % 2 != 0 { show_arrow = false; }
                    } else {
                        self.blink_left_until = None;
                    }
                }
                if show_arrow {
                    let indicator_area = Rect { x: size.x, y: size.y + size.height / 2, width: 1, height: 1 };
                    if indicator_area.y < size.y + size.height {
                        frame.render_widget(Paragraph::new("◄").style(Style::default().fg(indicator_color)), indicator_area);
                    }
                }
            }
            if is_clipped_right {
                let mut show_arrow = true;
                if let Some(blink_end_time) = self.blink_right_until {
                    if now < blink_end_time {
                        if (time_since_app_start_ms / BLINK_VISIBILITY_INTERVAL_MS) % 2 != 0 { show_arrow = false; }
                    } else {
                        self.blink_right_until = None;
                    }
                }
                if show_arrow {
                    let indicator_area = Rect { x: size.x + size.width - 1, y: size.y + size.height / 2, width: 1, height: 1 };
                    if indicator_area.x < size.x + size.width && indicator_area.y < size.y + size.height {
                        frame.render_widget(Paragraph::new("►").style(Style::default().fg(indicator_color)), indicator_area);
                    }
                }
            }
        }
    }
}


#[tokio::main]
async fn main() -> io::Result<()> {
    let app = AppState::new();
    let tick_rate = Duration::from_millis(50); // For AppEvent::Tick
    let input_poll_rate = Duration::from_millis(10); // How often to check for crossterm events

    run_tokio_tui_app(app, tick_rate, input_poll_rate).await
}

