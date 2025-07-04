//! Multiple Lorenz Attractors for Brain Dynamics Modeling
//! 
//! Inspired by Prof. Carhart-Harris's work on consciousness and chaotic brain dynamics.
//! This example creates multiple interacting Lorenz attractors using dLinOSS to model
//! complex neural dynamics that might represent different brain regions or neural circuits.

use burn::{
    prelude::*,
};
use linoss_rust::linoss::{
    production_dlinoss::{ProductionDLinossModel, ProductionDLinossConfig, create_production_dlinoss_classifier},
};
use ratatui::{
    prelude::{Constraint, CrosstermBackend, Direction, Layout, Frame},
    style::{Color, Style},
    symbols,
    widgets::{Block as TuiBlock, Borders, Chart, Dataset, Paragraph, List, ListItem},
    Terminal,
};
use std::{
    error::Error, 
    io::{self, Stdout}, 
    time::{Duration, Instant}, 
    collections::VecDeque,
};

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

type B = chosen_backend::SelectedBackend;

// Constants for the brain dynamics simulation
const MAX_TRAJECTORY_POINTS: usize = 2000;
const SIMULATION_SPEED: u64 = 30; // Faster for better responsiveness
const CANVAS_X_BOUNDS: [f64; 2] = [-50.0, 50.0];
const CANVAS_Y_BOUNDS: [f64; 2] = [-50.0, 50.0];
const DT: f64 = 0.005; // Smaller time step to reduce numerical damping

// Lorenz attractor parameters (classic chaotic system)
#[derive(Clone)]
struct LorenzParams {
    sigma: f64,   // Prandtl number
    rho: f64,     // Rayleigh number
    beta: f64,    // Geometric parameter
    coupling: f64, // Coupling strength with other attractors
}

impl Default for LorenzParams {
    fn default() -> Self {
        Self {
            sigma: 10.0,
            rho: 28.0,
            beta: 8.0/3.0,
            coupling: 0.1,
        }
    }
}

// Brain region representation
#[derive(Clone)]
struct BrainRegion {
    name: String,
    params: LorenzParams,
    position: (f64, f64, f64), // Current Lorenz state (x, y, z)
    trajectory: VecDeque<(f64, f64)>, // 2D projection for visualization
    color: Color,
    dlinoss_model: Option<ProductionDLinossModel<B>>,
    dlinoss_input: Option<Tensor<B, 3>>,
}

impl BrainRegion {
    fn new(name: String, initial_pos: (f64, f64, f64), params: LorenzParams, color: Color) -> Self {
        Self {
            name,
            params,
            position: initial_pos,
            trajectory: VecDeque::with_capacity(MAX_TRAJECTORY_POINTS),
            color,
            dlinoss_model: None,
            dlinoss_input: None,
        }
    }

    fn init_dlinoss(&mut self, device: &<B as Backend>::Device) {
        let layer_config = DLinossLayerConfig::new_dlinoss(3, 8, 3); // Input=3 (x,y,z), Hidden=8, Output=3 (x,y,z)

        self.dlinoss_model = Some(DLinossLayer::new(&layer_config, device));
        self.dlinoss_input = Some(Tensor::<B, 2>::from_floats(
            [[self.position.0 as f32, self.position.1 as f32, self.position.2 as f32]], 
            device,
        ));
    }

    // Classic Lorenz equations with coupling
    fn lorenz_step(&mut self, dt: f64, coupling_input: (f64, f64, f64)) {
        let (x, y, z) = self.position;
        let (cx, cy, cz) = coupling_input;
        
        let dx = self.params.sigma * (y - x) + self.params.coupling * cx;
        let dy = x * (self.params.rho - z) - y + self.params.coupling * cy;
        let dz = x * y - self.params.beta * z + self.params.coupling * cz;
        
        self.position.0 += dx * dt;
        self.position.1 += dy * dt;
        self.position.2 += dz * dt;
    }

    // Update using dLinOSS (neural dynamics on top of chaotic attractor)
    fn dlinoss_step(&mut self) {
        if let (Some(model), Some(input)) = (&self.dlinoss_model, &self.dlinoss_input) {
            // Create a sequence of length 1 for the current state
            let sequence_input = input.clone().unsqueeze_dim(1); // [1, 1, 3]
            
            // Forward through dLinOSS
            let output = model.forward(sequence_input); // [1, 1, 3]
            let output_squeezed = output.squeeze::<2>(1); // [1, 3]
            
            // Extract output and influence the Lorenz system
            let output_data = output_squeezed.into_data();
            let output_values = output_data.to_vec::<f32>().unwrap();
            
            // Apply dLinOSS output as perturbations to the Lorenz system
            self.position.0 += output_values[0] as f64 * 0.002; // Even smaller influence factor
            self.position.1 += output_values[1] as f64 * 0.002;
            self.position.2 += output_values[2] as f64 * 0.002;

            // Update dLinOSS input with current position
            self.dlinoss_input = Some(Tensor::<B, 2>::from_floats(
                [[self.position.0 as f32, self.position.1 as f32, self.position.2 as f32]], 
                &input.device(),
            ));
        }
    }

    fn update_trajectory(&mut self) {
        // Project 3D Lorenz attractor to 2D for visualization (x-y plane)
        let point = (self.position.0, self.position.1);
        
        if self.trajectory.len() >= MAX_TRAJECTORY_POINTS {
            self.trajectory.pop_front();
        }
        self.trajectory.push_back(point);
    }

    fn get_trajectory_data(&self) -> Vec<(f64, f64)> {
        self.trajectory.iter().cloned().collect()
    }
}

struct BrainDynamicsApp {
    regions: Vec<BrainRegion>,
    paused: bool,
    status_message: String,
    simulation_time: f64,
    coupling_strength: f64,
    use_linoss: bool,
    x_min: f64, x_max: f64, y_min: f64, y_max: f64,
}

impl BrainDynamicsApp {
    fn new() -> Self {
        let device = chosen_backend::get_device();
        
        // Create different brain regions with different parameters
        let mut regions = vec![
            BrainRegion::new(
                "Prefrontal Cortex".to_string(),
                (1.0, 1.0, 1.0),
                LorenzParams { sigma: 10.0, rho: 28.0, beta: 8.0/3.0, coupling: 0.05 },
                Color::Cyan
            ),
            BrainRegion::new(
                "Default Mode Network".to_string(),
                (-1.0, -1.0, -1.0),
                LorenzParams { sigma: 16.0, rho: 45.6, beta: 4.0, coupling: 0.08 },
                Color::Yellow
            ),
            BrainRegion::new(
                "Thalamus".to_string(),
                (0.5, -0.5, 1.5),
                LorenzParams { sigma: 12.0, rho: 35.0, beta: 3.0, coupling: 0.12 },
                Color::Magenta
            ),
        ];

        // Initialize dLinOSS for each region
        for region in &mut regions {
            region.init_dlinoss(&device);
        }

        Self {
            regions,
            paused: false,
            status_message: "Brain Dynamics Running - Press 'p' to pause, 'l' to toggle dLinOSS (Damped), 'q' to quit".to_string(),
            simulation_time: 0.0,
            coupling_strength: 0.1,
            use_linoss: true,
            x_min: f64::MAX, x_max: f64::MIN, y_min: f64::MAX, y_max: f64::MIN,
        }
    }

    fn on_tick(&mut self) {
        if self.paused {
            return;
        }

        self.simulation_time += DT;

        // Calculate coupling between regions
        let coupling_matrix = self.calculate_coupling();

        // Update each brain region
        for (i, region) in self.regions.iter_mut().enumerate() {
            // Apply Lorenz dynamics with coupling from other regions
            region.lorenz_step(DT, coupling_matrix[i]);
            
            // Apply dLinOSS neural dynamics if enabled
            if self.use_linoss {
                region.dlinoss_step();
            }
            
            // Update trajectory for visualization
            region.update_trajectory();
            
            // Update bounds for auto-scaling
            let (x, y) = (region.position.0, region.position.1);
            self.x_min = self.x_min.min(x);
            self.x_max = self.x_max.max(x);
            self.y_min = self.y_min.min(y);
            self.y_max = self.y_max.max(y);
        }
    }

    fn calculate_coupling(&self) -> Vec<(f64, f64, f64)> {
        let mut coupling = vec![(0.0, 0.0, 0.0); self.regions.len()];
        
        for (i, coupling_ref) in coupling.iter_mut().enumerate().take(self.regions.len()) {
            let mut total_coupling = (0.0, 0.0, 0.0);
            
            for j in 0..self.regions.len() {
                if i != j {
                    let weight = self.coupling_strength * self.regions[j].params.coupling;
                    total_coupling.0 += weight * self.regions[j].position.0;
                    total_coupling.1 += weight * self.regions[j].position.1;
                    total_coupling.2 += weight * self.regions[j].position.2;
                }
            }
            
            *coupling_ref = total_coupling;
        }
        
        coupling
    }

    fn toggle_pause(&mut self) {
        self.paused = !self.paused;
        self.status_message = if self.paused {
            "PAUSED - Press 'p' to resume".to_string()
        } else {
            "Brain Dynamics Running - Press 'p' to pause, 'l' to toggle dLinOSS (Damped), 'q' to quit".to_string()
        };
    }

    fn toggle_linoss(&mut self) {
        self.use_linoss = !self.use_linoss;
        self.status_message = format!(
            "dLinOSS (Damped): {} - Press 'p' to pause, 'l' to toggle dLinOSS, 'q' to quit",
            if self.use_linoss { "ON" } else { "OFF" }
        );
    }
}

fn draw_ui(f: &mut Frame, app: &BrainDynamicsApp) {
    let main_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(70), // Chart area
            Constraint::Percentage(20), // Status area
            Constraint::Percentage(10), // Info area
        ].as_ref())
        .split(f.area());

    // Chart area
    let chart_area = main_chunks[0];
    let status_area = main_chunks[1];
    let info_area = main_chunks[2];

    // Auto-scale or use default bounds
    let x_bounds = if app.x_min == f64::MAX || app.x_min == app.x_max { 
        CANVAS_X_BOUNDS 
    } else { 
        [app.x_min * 1.1, app.x_max * 1.1] 
    };
    let y_bounds = if app.y_min == f64::MAX || app.y_min == app.y_max { 
        CANVAS_Y_BOUNDS 
    } else { 
        [app.y_min * 1.1, app.y_max * 1.1] 
    };

    // Collect trajectory data first to ensure proper lifetimes
    let trajectory_data: Vec<Vec<(f64, f64)>> = app.regions.iter()
        .map(|region| region.get_trajectory_data())
        .collect();
    
    // Create datasets for each brain region
    let datasets: Vec<Dataset> = app.regions.iter()
        .enumerate()
        .map(|(i, region)| {
            Dataset::default()
                .name(region.name.clone())
                .marker(symbols::Marker::Dot)
                .style(Style::default().fg(region.color))
                .data(&trajectory_data[i])
        }).collect();

    let chart = Chart::new(datasets)
        .block(
            TuiBlock::default()
                .title("Brain Dynamics: Multiple Lorenz Attractors (Carhart-Harris Model)")
                .borders(Borders::ALL),
        )
        .x_axis(
            ratatui::widgets::Axis::default()
                .title("X (Neural Activity)")
                .style(Style::default().fg(Color::Gray))
                .bounds(x_bounds),
        )
        .y_axis(
            ratatui::widgets::Axis::default()
                .title("Y (Neural Activity)")
                .style(Style::default().fg(Color::Gray))
                .bounds(y_bounds),
        );
    f.render_widget(chart, chart_area);

    // Status area
    let status_block = TuiBlock::default().borders(Borders::ALL).title("System Status");
    let status_inner = status_block.inner(status_area);
    f.render_widget(status_block, status_area);
    let status_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // Control status
            Constraint::Length(1), // Simulation time
            Constraint::Length(1), // LinOSS status
        ].as_ref())
        .split(status_inner);

    let control_paragraph = Paragraph::new(app.status_message.as_str());
    f.render_widget(control_paragraph, status_chunks[0]);

    let time_paragraph = Paragraph::new(format!("Simulation Time: {:.2}s", app.simulation_time));
    f.render_widget(time_paragraph, status_chunks[1]);

    let linoss_paragraph = Paragraph::new(format!(
        "dLinOSS Neural Dynamics: {} | Coupling Strength: {:.3}", 
        if app.use_linoss { "ACTIVE" } else { "DISABLED" },
        app.coupling_strength
    ));
    f.render_widget(linoss_paragraph, status_chunks[2]);

    // Info area - Brain regions info
    let info_block = TuiBlock::default().borders(Borders::ALL).title("Brain Regions");
    let info_inner = info_block.inner(info_area);
    f.render_widget(info_block, info_area);
    let region_info: Vec<ListItem> = app.regions.iter().map(|region| {
        ListItem::new(format!(
            "{}: ({:.2}, {:.2}, {:.2})", 
            region.name, 
            region.position.0, 
            region.position.1, 
            region.position.2
        )).style(Style::default().fg(region.color))
    }).collect();

    let regions_list = List::new(region_info);
    f.render_widget(regions_list, info_inner);
}

fn run_app(terminal: &mut Terminal<CrosstermBackend<Stdout>>) -> Result<(), Box<dyn Error>> {
    let mut app = BrainDynamicsApp::new();
    let mut last_tick = Instant::now();

    loop {
        terminal.draw(|f| draw_ui(f, &app))?;

        let timeout = Duration::from_millis(SIMULATION_SPEED)
            .checked_sub(last_tick.elapsed())
            .unwrap_or_else(|| Duration::from_secs(0));

        if crossterm::event::poll(timeout)? {
            if let event::Event::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Char('q') => return Ok(()),
                    KeyCode::Char('p') => app.toggle_pause(),
                    KeyCode::Char('l') => app.toggle_linoss(),
                    KeyCode::Char('+') => {
                        app.coupling_strength = (app.coupling_strength + 0.01).min(1.0);
                    },
                    KeyCode::Char('-') => {
                        app.coupling_strength = (app.coupling_strength - 0.01).max(0.0);
                    },
                    _ => {}
                }
            }
        }

        if last_tick.elapsed() >= Duration::from_millis(SIMULATION_SPEED) {
            app.on_tick();
            last_tick = Instant::now();
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Run the app
    let res = run_app(&mut terminal);

    // Restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
    )?;
    terminal.show_cursor()?;

    if let Err(err) = res {
        println!("{:?}", err);
    }

    Ok(())
}
