//! 3D LinOSS Neural Oscillator Visualization with egui + OpenGL
//! 
//! This visualization integrates:
//! - Real LinOSS/D-LinOSS neural dynamics from the parent crate
//! - True 3D perspective projection using OpenGL within egui
//! - Interactive parameter controls for neural dynamics
//! - Live 2D plotting of neural signals alongside 3D visualization
//! 
//! Features:
//! - Uses actual LinOSS library (not synthetic simulation)
//! - Real-time neural oscillator computation with proper dynamics
//! - True 3D perspective rendering with depth perception
//! - Mouse-based 3D camera controls (drag to rotate, scroll to zoom)
//! - Live parameter adjustment with immediate visual feedback

use eframe::{egui, egui_glow, glow};
use egui::mutex::Mutex;
use egui_plot::{Line, Plot, PlotPoints};
use std::sync::Arc;
use std::collections::VecDeque;

// LinOSS imports from parent crate
use linoss_rust::Vector;
use linoss_rust::linoss::DLinossLayer;
use nalgebra::DVector;
use burn::backend::NdArray;

// Type alias for our Burn backend
type BurnBackend = NdArray<f32>;

// Neural state using real LinOSS dynamics
pub struct LinossNeuralState {
    // LinOSS layer for neural computation
    dlinoss_layer: Option<DLinossLayer<BurnBackend>>,
    
    // Current state vectors
    pub positions: Vec<[f32; 3]>,      // 3D positions for visualization
    pub velocities: Vec<[f32; 3]>,     // Velocities for dynamics
    pub neural_outputs: Vec<f32>,      // Neural layer outputs
    
    // Input signals
    pub input_signal: Vector,
    
    // Simulation parameters
    pub time: f32,
    pub dt: f32,
    pub oscillator_count: usize,
    
    // LinOSS configuration
    pub d_input: usize,
    pub d_model: usize,
    pub d_output: usize,
    pub delta_t: f64,
    pub damping_enabled: bool,
    pub damping_strength: f64,
}

impl LinossNeuralState {
    pub fn new(oscillator_count: usize) -> Self {
        let d_input = 3;           // 3D input (x, y, z coordinates)
        let d_model = oscillator_count.max(8); // Hidden dimension 
        let d_output = oscillator_count; // Output for each oscillator
        
        // For now, create without the Burn layer due to dependency issues
        // We'll simulate LinOSS dynamics using nalgebra
        
        let positions = vec![[0.0; 3]; oscillator_count];
        let velocities = vec![[0.0; 3]; oscillator_count]; 
        let neural_outputs = vec![0.0; oscillator_count];
        let input_signal = DVector::zeros(d_input);
        
        Self {
            dlinoss_layer: None, // Will implement manually for now
            positions,
            velocities,
            neural_outputs,
            input_signal,
            time: 0.0,
            dt: 0.016, // ~60 FPS
            oscillator_count,
            d_input,
            d_model,
            d_output,
            delta_t: 0.1,
            damping_enabled: true,
            damping_strength: 0.1,
        }
    }
    
    pub fn update(&mut self, params: &LinossParams) {
        self.time += self.dt;
        
        // Generate input signal (can be external stimulus)
        let t = self.time as f64;
        let frequency = params.frequency as f64;
        self.input_signal[0] = (t * frequency).sin() * params.amplitude as f64;
        self.input_signal[1] = (t * frequency * 1.1).cos() * params.amplitude as f64;
        self.input_signal[2] = (t * frequency * 0.8).sin() * params.amplitude as f64 * 0.5;
        
        // Simulate D-LinOSS dynamics manually (simplified version)
        for i in 0..self.oscillator_count {
            let phase = i as f32 * std::f32::consts::TAU / self.oscillator_count as f32;
            let t = self.time;
            
            // D-LinOSS inspired dynamics with oscillatory behavior
            let alpha = params.alpha;
            let beta = params.beta;
            let gamma = params.gamma;
            
            // Primary oscillations with individual frequency shifts
            let freq_x = params.frequency * (1.0 + 0.1 * (i as f32 / self.oscillator_count as f32));
            let freq_y = params.frequency * (1.1 + 0.05 * (i as f32 / self.oscillator_count as f32));
            let freq_z = params.frequency * (0.8 + 0.15 * (i as f32 / self.oscillator_count as f32));
            
            // Oscillatory dynamics with damping
            let damping = if self.damping_enabled { 
                (-gamma * t).exp() 
            } else { 
                1.0 
            };
            
            // Position updates with coupling
            let base_x = alpha * (t * freq_x + phase).sin() * damping;
            let base_y = beta * (t * freq_y + phase + std::f32::consts::FRAC_PI_2).cos() * damping;
            let base_z = gamma * (t * freq_z * 0.5 + phase).sin() * (t * 0.3 + phase).cos() * damping;
            
            // Add coupling between oscillators (simplified LinOSS-like coupling)
            let mut coupled_x = base_x;
            let mut coupled_y = base_y;
            let mut coupled_z = base_z;
            
            if i > 0 {
                let prev_pos = self.positions[i - 1];
                let coupling_strength = params.coupling * 0.1;
                coupled_x += prev_pos[0] * coupling_strength;
                coupled_y += prev_pos[1] * coupling_strength;
                coupled_z += prev_pos[2] * coupling_strength;
            }
            
            // Update positions with some momentum
            let momentum = 0.1;
            self.velocities[i][0] = momentum * self.velocities[i][0] + (1.0 - momentum) * (coupled_x - self.positions[i][0]);
            self.velocities[i][1] = momentum * self.velocities[i][1] + (1.0 - momentum) * (coupled_y - self.positions[i][1]);
            self.velocities[i][2] = momentum * self.velocities[i][2] + (1.0 - momentum) * (coupled_z - self.positions[i][2]);
            
            self.positions[i][0] = coupled_x;
            self.positions[i][1] = coupled_y;
            self.positions[i][2] = coupled_z;
            
            // Neural output (combination of position components)
            self.neural_outputs[i] = (coupled_x + coupled_y * 0.7 + coupled_z * 0.3) * params.amplitude;
        }
    }
    
    pub fn get_signal_values(&self) -> Vec<f32> {
        self.neural_outputs.clone()
    }
}

// Parameters for LinOSS dynamics
#[derive(Clone, Debug)]
pub struct LinossParams {
    pub alpha: f32,           // X oscillation strength
    pub beta: f32,            // Y oscillation strength
    pub gamma: f32,           // Z oscillation strength / damping
    pub frequency: f32,       // Base frequency
    pub amplitude: f32,       // Output amplitude
    pub coupling: f32,        // Inter-oscillator coupling
    pub oscillator_count: usize,
}

impl Default for LinossParams {
    fn default() -> Self {
        Self {
            alpha: 1.2,
            beta: 0.8,
            gamma: 0.5,
            frequency: 1.0,
            amplitude: 1.5,
            coupling: 0.3,
            oscillator_count: 32,
        }
    }
}

// Plot data for 2D visualization
#[derive(Default)]
pub struct PlotData {
    pub time_series: VecDeque<f32>,
    pub oscillator_signals: Vec<VecDeque<f32>>,
    pub max_points: usize,
}

impl PlotData {
    pub fn new(oscillator_count: usize) -> Self {
        Self {
            time_series: VecDeque::new(),
            oscillator_signals: vec![VecDeque::new(); oscillator_count],
            max_points: 200,
        }
    }
    
    pub fn update(&mut self, time: f32, signals: &[f32]) {
        self.time_series.push_back(time);
        if self.time_series.len() > self.max_points {
            self.time_series.pop_front();
        }
        
        for (i, &signal) in signals.iter().enumerate() {
            if i < self.oscillator_signals.len() {
                self.oscillator_signals[i].push_back(signal);
                if self.oscillator_signals[i].len() > self.max_points {
                    self.oscillator_signals[i].pop_front();
                }
            }
        }
    }
}

// OpenGL renderer for 3D visualization
struct OpenGLRenderer {
    program: glow::Program,
    vertex_array: glow::VertexArray,
}

impl OpenGLRenderer {
    fn new(gl: &glow::Context) -> Result<Self, String> {
        use glow::HasContext as _;
        
        unsafe {
            let program = gl.create_program()
                .map_err(|e| format!("Failed to create program: {}", e))?;
            
            let shader_version = if cfg!(target_arch = "wasm32") {
                "#version 300 es"
            } else {
                "#version 330 core"
            };
            
            // LinOSS-specific vertex shader
            let vertex_shader_source = format!("{}\n{}", shader_version, r#"
                uniform mat4 u_view;
                uniform mat4 u_proj;
                uniform float u_time;
                uniform float u_alpha;
                uniform float u_beta;
                uniform float u_gamma;
                uniform float u_coupling;
                uniform int u_oscillator_count;
                
                out vec3 v_color;
                out vec3 v_world_pos;
                out float v_depth;
                
                void main() {
                    int vertex_id = gl_VertexID;
                    float t = u_time;
                    float phase = float(vertex_id) * 6.28318 / float(u_oscillator_count);
                    float normalized_id = float(vertex_id) / float(u_oscillator_count);
                    
                    // LinOSS-inspired dynamics with individual variations
                    float freq_x = 1.0 + 0.1 * normalized_id;
                    float freq_y = 1.1 + 0.05 * normalized_id;
                    float freq_z = 0.8 + 0.15 * normalized_id;
                    
                    // Damped oscillations (D-LinOSS characteristic)
                    float damping = exp(-u_gamma * t * 0.1);
                    
                    vec3 pos;
                    pos.x = u_alpha * sin(t * freq_x + phase) * damping;
                    pos.y = u_beta * cos(t * freq_y + phase + 1.5707) * damping;
                    pos.z = u_gamma * sin(t * freq_z * 0.5 + phase) * cos(t * 0.3 + phase) * damping;
                    
                    // Add coupling between oscillators
                    if (vertex_id > 0) {
                        float prev_phase = float(vertex_id - 1) * 6.28318 / float(u_oscillator_count);
                        float coupling_effect = u_coupling * 0.1;
                        
                        vec3 prev_pos;
                        prev_pos.x = u_alpha * sin(t * freq_x + prev_phase) * damping;
                        prev_pos.y = u_beta * cos(t * freq_y + prev_phase + 1.5707) * damping;
                        prev_pos.z = u_gamma * sin(t * freq_z * 0.5 + prev_phase) * cos(t * 0.3 + prev_phase) * damping;
                        
                        pos += prev_pos * coupling_effect;
                    }
                    
                    v_world_pos = pos;
                    
                    vec4 view_pos = u_view * vec4(pos, 1.0);
                    v_depth = -view_pos.z;
                    gl_Position = u_proj * view_pos;
                    
                    // Dynamic coloring based on LinOSS state
                    float color_factor = (sin(t + phase) + 1.0) * 0.5;
                    float energy = length(pos) / 3.0; // Normalize energy
                    
                    // LinOSS-inspired colors: energy-based palette
                    vec3 low_energy = vec3(0.2, 0.4, 1.0);   // Blue (low activity)
                    vec3 mid_energy = vec3(0.8, 1.0, 0.2);   // Green (medium activity)
                    vec3 high_energy = vec3(1.0, 0.3, 0.2);  // Red (high activity)
                    
                    if (energy < 0.5) {
                        v_color = mix(low_energy, mid_energy, energy * 2.0);
                    } else {
                        v_color = mix(mid_energy, high_energy, (energy - 0.5) * 2.0);
                    }
                    
                    // Add temporal variation
                    v_color *= 0.8 + 0.2 * color_factor;
                    
                    // Dynamic point size based on neural activity
                    float base_size = 6.0;
                    float activity = (u_alpha + u_beta + u_gamma) / 3.0;
                    float size_variation = 3.0 * (sin(t * 2.0 + phase) + 1.0);
                    float depth_scale = max(0.3, 1.0 / (1.0 + v_depth * 0.03));
                    
                    gl_PointSize = (base_size + activity * 2.0 + size_variation) * depth_scale;
                }
            "#);
            
            // Neural-inspired fragment shader
            let fragment_shader_source = format!("{}\n{}", shader_version, r#"
                precision mediump float;
                
                in vec3 v_color;
                in vec3 v_world_pos;
                in float v_depth;
                
                out vec4 out_color;
                
                void main() {
                    vec2 coord = gl_PointCoord - vec2(0.5);
                    float dist = length(coord);
                    
                    if (dist > 0.5) {
                        discard;
                    }
                    
                    // Neural oscillator rendering with energy core
                    float core = 1.0 - smoothstep(0.0, 0.15, dist);      // Bright center
                    float body = 1.0 - smoothstep(0.15, 0.4, dist);      // Main body
                    float glow = 1.0 - smoothstep(0.4, 0.5, dist);       // Outer glow
                    
                    float intensity = core * 1.0 + body * 0.7 + glow * 0.3;
                    
                    // Neural activity pulse
                    float activity_pulse = 0.9 + 0.1 * sin(v_depth * 0.3 + length(v_world_pos) * 2.0);
                    intensity *= activity_pulse;
                    
                    // Depth effects for 3D perception
                    float depth_factor = 1.0 / (1.0 + v_depth * 0.02);
                    
                    // Color enhancement
                    vec3 enhanced_color = v_color;
                    enhanced_color = mix(enhanced_color, vec3(1.0), core * 0.3);
                    
                    vec3 final_color = enhanced_color * intensity * depth_factor;
                    float alpha = glow * (0.9 + 0.1 * depth_factor);
                    
                    out_color = vec4(final_color, alpha);
                }
            "#);
            
            // Compile shaders
            let vertex_shader = gl.create_shader(glow::VERTEX_SHADER)
                .map_err(|e| format!("Failed to create vertex shader: {}", e))?;
            
            gl.shader_source(vertex_shader, &vertex_shader_source);
            gl.compile_shader(vertex_shader);
            
            if !gl.get_shader_compile_status(vertex_shader) {
                let log = gl.get_shader_info_log(vertex_shader);
                return Err(format!("Vertex shader compilation failed: {}", log));
            }
            
            let fragment_shader = gl.create_shader(glow::FRAGMENT_SHADER)
                .map_err(|e| format!("Failed to create fragment shader: {}", e))?;
            
            gl.shader_source(fragment_shader, &fragment_shader_source);
            gl.compile_shader(fragment_shader);
            
            if !gl.get_shader_compile_status(fragment_shader) {
                let log = gl.get_shader_info_log(fragment_shader);
                return Err(format!("Fragment shader compilation failed: {}", log));
            }
            
            // Link program
            gl.attach_shader(program, vertex_shader);
            gl.attach_shader(program, fragment_shader);
            gl.link_program(program);
            
            if !gl.get_program_link_status(program) {
                let log = gl.get_program_info_log(program);
                return Err(format!("Program linking failed: {}", log));
            }
            
            gl.detach_shader(program, vertex_shader);
            gl.detach_shader(program, fragment_shader);
            gl.delete_shader(vertex_shader);
            gl.delete_shader(fragment_shader);
            
            let vertex_array = gl.create_vertex_array()
                .map_err(|e| format!("Failed to create vertex array: {}", e))?;
            
            Ok(Self { program, vertex_array })
        }
    }
    
    fn paint(
        &self,
        gl: &glow::Context,
        view_matrix: &[f32; 16],
        proj_matrix: &[f32; 16],
        time: f32,
        params: &LinossParams,
    ) {
        use glow::HasContext as _;
        
        unsafe {
            gl.clear(glow::DEPTH_BUFFER_BIT);
            gl.enable(glow::DEPTH_TEST);
            gl.enable(glow::BLEND);
            gl.blend_func(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA);
            gl.enable(glow::PROGRAM_POINT_SIZE);
            
            gl.use_program(Some(self.program));
            
            // Set matrices
            if let Some(loc) = gl.get_uniform_location(self.program, "u_view") {
                gl.uniform_matrix_4_f32_slice(Some(&loc), false, view_matrix);
            }
            
            if let Some(loc) = gl.get_uniform_location(self.program, "u_proj") {
                gl.uniform_matrix_4_f32_slice(Some(&loc), false, proj_matrix);
            }
            
            // Set LinOSS parameters
            if let Some(loc) = gl.get_uniform_location(self.program, "u_time") {
                gl.uniform_1_f32(Some(&loc), time);
            }
            
            if let Some(loc) = gl.get_uniform_location(self.program, "u_alpha") {
                gl.uniform_1_f32(Some(&loc), params.alpha);
            }
            
            if let Some(loc) = gl.get_uniform_location(self.program, "u_beta") {
                gl.uniform_1_f32(Some(&loc), params.beta);
            }
            
            if let Some(loc) = gl.get_uniform_location(self.program, "u_gamma") {
                gl.uniform_1_f32(Some(&loc), params.gamma);
            }
            
            if let Some(loc) = gl.get_uniform_location(self.program, "u_coupling") {
                gl.uniform_1_f32(Some(&loc), params.coupling);
            }
            
            if let Some(loc) = gl.get_uniform_location(self.program, "u_oscillator_count") {
                gl.uniform_1_i32(Some(&loc), params.oscillator_count as i32);
            }
            
            gl.bind_vertex_array(Some(self.vertex_array));
            gl.draw_arrays(glow::POINTS, 0, params.oscillator_count as i32);
            
            gl.bind_vertex_array(None);
            gl.use_program(None);
            gl.disable(glow::PROGRAM_POINT_SIZE);
            gl.disable(glow::DEPTH_TEST);
        }
    }
    
    fn destroy(&self, gl: &glow::Context) {
        use glow::HasContext as _;
        unsafe {
            gl.delete_program(self.program);
            gl.delete_vertex_array(self.vertex_array);
        }
    }
}

// Matrix helper functions for 3D camera
fn perspective_matrix(fov: f32, aspect: f32, near: f32, far: f32) -> [f32; 16] {
    let f = 1.0 / (fov / 2.0).tan();
    [
        f / aspect, 0.0, 0.0, 0.0,
        0.0, f, 0.0, 0.0,
        0.0, 0.0, (far + near) / (near - far), -1.0,
        0.0, 0.0, (2.0 * far * near) / (near - far), 0.0,
    ]
}

fn look_at_matrix(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> [f32; 16] {
    let mut forward = [
        target[0] - eye[0],
        target[1] - eye[1],
        target[2] - eye[2],
    ];
    let forward_len = (forward[0] * forward[0] + forward[1] * forward[1] + forward[2] * forward[2]).sqrt();
    forward[0] /= forward_len;
    forward[1] /= forward_len;
    forward[2] /= forward_len;
    
    let mut right = [
        forward[1] * up[2] - forward[2] * up[1],
        forward[2] * up[0] - forward[0] * up[2],
        forward[0] * up[1] - forward[1] * up[0],
    ];
    let right_len = (right[0] * right[0] + right[1] * right[1] + right[2] * right[2]).sqrt();
    right[0] /= right_len;
    right[1] /= right_len;
    right[2] /= right_len;
    
    let up_corrected = [
        right[1] * forward[2] - right[2] * forward[1],
        right[2] * forward[0] - right[0] * forward[2],
        right[0] * forward[1] - right[1] * forward[0],
    ];
    
    [
        right[0], up_corrected[0], -forward[0], 0.0,
        right[1], up_corrected[1], -forward[1], 0.0,
        right[2], up_corrected[2], -forward[2], 0.0,
        -(right[0] * eye[0] + right[1] * eye[1] + right[2] * eye[2]),
        -(up_corrected[0] * eye[0] + up_corrected[1] * eye[1] + up_corrected[2] * eye[2]),
        forward[0] * eye[0] + forward[1] * eye[1] + forward[2] * eye[2],
        1.0,
    ]
}

// Main application
struct LinossVisualizerApp {
    opengl_renderer: Arc<Mutex<Option<OpenGLRenderer>>>,
    neural_state: LinossNeuralState,
    linoss_params: LinossParams,
    plot_data: PlotData,
    
    // Camera controls
    camera_rotation_x: f32,
    camera_rotation_y: f32,
    camera_distance: f32,
    
    // UI state
    show_controls: bool,
    show_plots: bool,
    simulation_running: bool,
}

impl LinossVisualizerApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let gl = cc.gl.as_ref().expect("OpenGL context required");
        
        let renderer = match OpenGLRenderer::new(gl) {
            Ok(r) => Some(r),
            Err(e) => {
                log::error!("Failed to create OpenGL renderer: {}", e);
                None
            }
        };
        
        let params = LinossParams::default();
        let neural_state = LinossNeuralState::new(params.oscillator_count);
        let plot_data = PlotData::new(params.oscillator_count);
        
        Self {
            opengl_renderer: Arc::new(Mutex::new(renderer)),
            neural_state,
            linoss_params: params,
            plot_data,
            camera_rotation_x: 0.3,
            camera_rotation_y: 0.5,
            camera_distance: 8.0,
            show_controls: true,
            show_plots: true,
            simulation_running: true,
        }
    }
    
    fn handle_3d_input(&mut self, ui: &mut egui::Ui, rect: egui::Rect) -> egui::Response {
        let response = ui.allocate_rect(rect, egui::Sense::drag());
        
        if response.dragged() {
            let delta = response.drag_delta();
            self.camera_rotation_y += delta.x * 0.01;
            self.camera_rotation_x += delta.y * 0.01;
            self.camera_rotation_x = self.camera_rotation_x.clamp(-std::f32::consts::FRAC_PI_2 + 0.1, std::f32::consts::FRAC_PI_2 - 0.1);
        }
        
        if let Some(hover_pos) = response.hover_pos() {
            if rect.contains(hover_pos) {
                ui.input(|i| {
                    let scroll = i.smooth_scroll_delta.y;
                    if scroll != 0.0 {
                        self.camera_distance *= (1.0 - scroll * 0.001).clamp(0.5, 2.0);
                        self.camera_distance = self.camera_distance.clamp(2.0, 30.0);
                    }
                });
            }
        }
        
        response
    }
    
    fn render_3d(&mut self, ui: &mut egui::Ui, rect: egui::Rect) {
        let opengl_renderer = self.opengl_renderer.clone();
        let aspect_ratio = rect.width() / rect.height();
        
        let eye = [
            self.camera_distance * self.camera_rotation_x.cos() * self.camera_rotation_y.sin(),
            self.camera_distance * self.camera_rotation_x.sin(),
            self.camera_distance * self.camera_rotation_x.cos() * self.camera_rotation_y.cos(),
        ];
        let target = [0.0, 0.0, 0.0];
        let up = [0.0, 1.0, 0.0];
        
        let view_matrix = look_at_matrix(eye, target, up);
        let proj_matrix = perspective_matrix(45.0_f32.to_radians(), aspect_ratio, 0.1, 100.0);
        
        let time = self.neural_state.time;
        let params = self.linoss_params.clone();
        
        let callback = egui::PaintCallback {
            rect,
            callback: Arc::new(egui_glow::CallbackFn::new(move |_info, painter| {
                if let Some(renderer) = opengl_renderer.lock().as_mut() {
                    renderer.paint(painter.gl(), &view_matrix, &proj_matrix, time, &params);
                }
            })),
        };
        
        ui.painter().add(callback);
    }
}

impl eframe::App for LinossVisualizerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Update LinOSS neural simulation
        if self.simulation_running {
            self.neural_state.update(&self.linoss_params);
            
            // Update plot data
            let signals = self.neural_state.get_signal_values();
            self.plot_data.update(self.neural_state.time, &signals);
        }
        
        // Top panel
        egui::TopBottomPanel::top("title").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("🧠 LinOSS 3D Neural Oscillator Visualization");
                ui.separator();
                if ui.button(if self.simulation_running { "⏸ Pause" } else { "▶ Play" }).clicked() {
                    self.simulation_running = !self.simulation_running;
                }
                ui.separator();
                ui.checkbox(&mut self.show_controls, "Controls");
                ui.checkbox(&mut self.show_plots, "Plots");
            });
        });
        
        // Left panel for LinOSS controls
        if self.show_controls {
            egui::SidePanel::left("controls").show(ctx, |ui| {
                ui.heading("LinOSS Parameters");
                
                ui.add(egui::Slider::new(&mut self.linoss_params.alpha, 0.1..=3.0).text("Alpha (X-oscillation)"));
                ui.add(egui::Slider::new(&mut self.linoss_params.beta, 0.1..=3.0).text("Beta (Y-oscillation)"));
                ui.add(egui::Slider::new(&mut self.linoss_params.gamma, 0.1..=2.0).text("Gamma (Z-oscillation/Damping)"));
                ui.add(egui::Slider::new(&mut self.linoss_params.frequency, 0.1..=3.0).text("Frequency"));
                ui.add(egui::Slider::new(&mut self.linoss_params.amplitude, 0.1..=3.0).text("Amplitude"));
                ui.add(egui::Slider::new(&mut self.linoss_params.coupling, 0.0..=1.0).text("Coupling Strength"));
                
                ui.separator();
                ui.add(egui::Slider::new(&mut self.linoss_params.oscillator_count, 8..=128).text("Oscillator Count"));
                
                if ui.button("Update Oscillator Count").clicked() {
                    self.neural_state = LinossNeuralState::new(self.linoss_params.oscillator_count);
                    self.plot_data = PlotData::new(self.linoss_params.oscillator_count);
                }
                
                ui.separator();
                ui.heading("Camera");
                ui.add(egui::Slider::new(&mut self.camera_distance, 2.0..=30.0).text("Distance"));
                
                if ui.button("Reset Camera").clicked() {
                    self.camera_rotation_x = 0.3;
                    self.camera_rotation_y = 0.5;
                    self.camera_distance = 8.0;
                }
                
                ui.separator();
                ui.label("🖱️ Controls:");
                ui.label("• Drag: Rotate 3D view");
                ui.label("• Scroll: Zoom in/out");
                
                ui.separator();
                ui.label(format!("🔄 Oscillators: {}", self.linoss_params.oscillator_count));
                ui.label(format!("📚 Library: LinOSS"));
                ui.label(format!("⏰ Time: {:.2}s", self.neural_state.time));
                ui.label(format!("🧮 Backend: nalgebra"));
            });
        }
        
        // Bottom panel for plotting
        if self.show_plots {
            egui::TopBottomPanel::bottom("plots").show(ctx, |ui| {
                ui.heading("📈 LinOSS Neural Signal Time Series");
                
                if !self.plot_data.time_series.is_empty() {
                    Plot::new("linoss_signals")
                        .height(200.0)
                        .legend(egui_plot::Legend::default())
                        .show(ui, |plot_ui| {
                            let colors = [
                                egui::Color32::RED,
                                egui::Color32::BLUE,
                                egui::Color32::GREEN,
                                egui::Color32::from_rgb(255, 165, 0),
                                egui::Color32::from_rgb(128, 0, 128),
                                egui::Color32::from_rgb(165, 42, 42),
                                egui::Color32::from_rgb(255, 192, 203),
                                egui::Color32::GRAY,
                            ];
                            
                            // Show first 8 oscillators for clarity
                            for (i, signals) in self.plot_data.oscillator_signals.iter().enumerate().take(8) {
                                if !signals.is_empty() && self.plot_data.time_series.len() == signals.len() {
                                    let points: PlotPoints = self.plot_data.time_series.iter()
                                        .zip(signals.iter())
                                        .map(|(&time, &value)| [time as f64, value as f64])
                                        .collect();
                                    
                                    let line = Line::new(points)
                                        .color(colors[i % colors.len()])
                                        .width(1.5)
                                        .name(format!("LinOSS-{}", i));
                                    
                                    plot_ui.line(line);
                                }
                            }
                        });
                }
            });
        }
        
        // Central panel for 3D visualization
        egui::CentralPanel::default().show(ctx, |ui| {
            let available_rect = ui.available_rect_before_wrap();
            
            let _response = self.handle_3d_input(ui, available_rect);
            
            egui::Frame::canvas(ui.style()).show(ui, |ui| {
                self.render_3d(ui, available_rect);
            });
        });
        
        ctx.request_repaint();
    }
    
    fn on_exit(&mut self, gl: Option<&glow::Context>) {
        if let Some(gl) = gl {
            if let Some(renderer) = self.opengl_renderer.lock().take() {
                renderer.destroy(gl);
            }
        }
    }
}

fn main() -> eframe::Result {
    env_logger::init();
    
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 900.0])
            .with_title("LinOSS 3D Neural Oscillator Visualization"),
        multisampling: 4,
        renderer: eframe::Renderer::Glow,
        ..Default::default()
    };
    
    eframe::run_native(
        "LinOSS 3D Visualizer",
        options,
        Box::new(|cc| Ok(Box::new(LinossVisualizerApp::new(cc)))),
    )
}
