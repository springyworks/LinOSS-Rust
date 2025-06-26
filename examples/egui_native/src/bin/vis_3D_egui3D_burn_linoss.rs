//! 3D Neural Oscillator Visualization with D-LinOSS, OpenGL, and Burn Integration
//! 
//! This combines:
//! - D-LinOSS neural dynamics from the Bevy explorer
//! - True 3D perspective projection using OpenGL (from neural_3d_opengl.rs)
//! - Burn tensor operations for neural computation
//! - egui for parameter controls and 2D plotting
//! 
//! Features:
//! - Real-time neural oscillator simulation using Burn tensors
//! - True 3D perspective rendering with OpenGL shaders
//! - Interactive parameter controls for D-LinOSS dynamics
//! - Live 2D plotting of neural signals
//! - Mouse-based 3D camera controls

use eframe::{egui, egui_glow, glow};
use egui::mutex::Mutex;
use egui_plot::{Line, Plot, PlotPoints};
use std::sync::Arc;
use std::collections::VecDeque;

// Burn imports for neural computation
use burn::tensor::{Tensor, TensorData, backend::Backend};
use burn::backend::NdArray;

// Type alias for our Burn backend
type BurnBackend = NdArray<f32>;
type BurnTensor = Tensor<BurnBackend, 2>;

// D-LinOSS parameters structure
#[derive(Clone, Debug)]
pub struct DLinossParams {
    pub alpha: f32,     // Primary oscillation strength
    pub beta: f32,      // Secondary oscillation strength  
    pub gamma: f32,     // Damping coefficient
    pub delta: f32,     // Coupling strength
    pub frequency: f32, // Base frequency multiplier
    pub amplitude: f32, // Output amplitude scaling
    pub oscillator_count: usize, // Number of neural oscillators
}

impl Default for DLinossParams {
    fn default() -> Self {
        Self {
            alpha: 1.2,
            beta: 0.8,
            gamma: 0.5,
            delta: 0.3,
            frequency: 1.0,
            amplitude: 1.5,
            oscillator_count: 64,
        }
    }
}

// Neural oscillator state using Burn tensors
pub struct NeuralOscillatorState {
    // State tensors [oscillator_count, state_dim]
    pub positions: BurnTensor,      // Current 3D positions
    pub velocities: BurnTensor,     // Current velocities
    pub phases: BurnTensor,         // Oscillator phases
    pub frequencies: BurnTensor,    // Individual frequencies
    
    // Simulation parameters
    pub time: f32,
    pub dt: f32,
    pub device: burn::backend::ndarray::NdArrayDevice,
}

impl NeuralOscillatorState {
    pub fn new(oscillator_count: usize) -> Self {
        let device = burn::backend::ndarray::NdArrayDevice::default();
        
        // Initialize tensors
        let positions = Tensor::zeros([oscillator_count, 3], &device);
        let velocities = Tensor::zeros([oscillator_count, 3], &device);
        
        // Initialize phases with different starting phases
        let phase_data: Vec<f32> = (0..oscillator_count)
            .map(|i| i as f32 * std::f32::consts::TAU / oscillator_count as f32)
            .collect();
        let phases = Tensor::from_data(TensorData::new(phase_data, [oscillator_count, 1]), &device);
        
        // Initialize frequencies with slight variations
        let freq_data: Vec<f32> = (0..oscillator_count)
            .map(|i| 1.0 + (i as f32 * 0.1) / oscillator_count as f32)
            .collect();
        let frequencies = Tensor::from_data(TensorData::new(freq_data, [oscillator_count, 1]), &device);
        
        Self {
            positions,
            velocities,
            phases,
            frequencies,
            time: 0.0,
            dt: 0.016, // ~60 FPS
            device,
        }
    }
    
    pub fn update(&mut self, params: &DLinossParams) {
        self.time += self.dt;
        
        // Update phases based on frequencies and parameters
        let dt_tensor = Tensor::from_data(TensorData::new(vec![self.dt * params.frequency], [1]), &self.device);
        let freq_update = self.frequencies.clone() * dt_tensor;
        self.phases = self.phases.clone() + freq_update;
        
        // D-LinOSS dynamics using Burn tensor operations
        let alpha_tensor = Tensor::from_data(TensorData::new(vec![params.alpha], [1]), &self.device);
        let beta_tensor = Tensor::from_data(TensorData::new(vec![params.beta], [1]), &self.device);
        let gamma_tensor = Tensor::from_data(TensorData::new(vec![params.gamma], [1]), &self.device);
        
        // Primary oscillation: x = alpha * sin(phase)
        let x_positions = (self.phases.clone() * alpha_tensor).sin() * params.amplitude;
        
        // Secondary oscillation: y = beta * cos(phase + pi/2)
        let phase_shift = Tensor::from_data(TensorData::new(vec![std::f32::consts::FRAC_PI_2], [1]), &self.device);
        let y_positions = ((self.phases.clone() + phase_shift) * beta_tensor).cos() * params.amplitude;
        
        // Tertiary oscillation: z = gamma * sin(phase * 0.5)
        let z_positions = (self.phases.clone() * gamma_tensor * 0.5).sin() * params.amplitude * 0.8;
        
        // Combine into 3D positions
        self.positions = Tensor::stack([x_positions, y_positions, z_positions], 1);
        
        // Add coupling between oscillators (simplified)
        if params.delta > 0.0 {
            let coupling_strength = Tensor::from_data(TensorData::new(vec![params.delta * 0.2], [1]), &self.device);
            
            // Simple nearest-neighbor coupling
            let coupled_effect = self.positions.clone() * coupling_strength;
            self.positions = self.positions.clone() + coupled_effect;
        }
    }
    
    pub fn get_positions_as_vec(&self) -> Vec<[f32; 3]> {
        let data = self.positions.to_data();
        let values = data.as_slice::<f32>().unwrap();
        
        let mut positions = Vec::new();
        for i in 0..self.positions.dims()[0] {
            let base_idx = i * 3;
            if base_idx + 2 < values.len() {
                positions.push([
                    values[base_idx],
                    values[base_idx + 1], 
                    values[base_idx + 2]
                ]);
            }
        }
        positions
    }
    
    pub fn get_signal_values(&self) -> Vec<f32> {
        // Extract Y positions as neural signal values
        let data = self.positions.to_data();
        let values = data.as_slice::<f32>().unwrap();
        
        let mut signals = Vec::new();
        for i in 0..self.positions.dims()[0] {
            let y_idx = i * 3 + 1; // Y component
            if y_idx < values.len() {
                signals.push(values[y_idx]);
            }
        }
        signals
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
        // Add time point
        self.time_series.push_back(time);
        if self.time_series.len() > self.max_points {
            self.time_series.pop_front();
        }
        
        // Add signal values
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

// OpenGL renderer for 3D visualization (based on neural_3d_opengl.rs)
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
            
            // Vertex shader for neural oscillators
            let vertex_shader_source = format!("{}\n{}", shader_version, r#"
                uniform mat4 u_view;
                uniform mat4 u_proj;
                uniform float u_time;
                uniform float u_alpha;
                uniform float u_beta;
                uniform float u_gamma;
                uniform int u_oscillator_count;
                
                out vec3 v_color;
                out vec3 v_world_pos;
                out float v_depth;
                
                void main() {
                    int vertex_id = gl_VertexID;
                    float t = u_time;
                    float phase = float(vertex_id) * 0.1;
                    float normalized_id = float(vertex_id) / float(u_oscillator_count);
                    
                    // Enhanced D-LinOSS dynamics with individual oscillator variations
                    float freq_x = 1.0 + 0.3 * normalized_id;
                    float freq_y = 1.1 + 0.2 * normalized_id;
                    float freq_z = 0.8 + 0.4 * normalized_id;
                    
                    vec3 pos;
                    pos.x = u_alpha * sin(t * freq_x + phase) * (1.0 + 0.3 * sin(t * 0.3 + phase));
                    pos.y = u_beta * cos(t * freq_y + phase + 1.57) * (1.0 + 0.2 * cos(t * 0.7 + phase));
                    pos.z = u_gamma * sin(t * freq_z * 0.5 + phase) * cos(t * 0.8 + phase) * 0.8;
                    
                    // Add coupling between oscillators
                    if (vertex_id > 0) {
                        float prev_phase = float(vertex_id - 1) * 0.1;
                        float coupling = 0.2;
                        pos += vec3(
                            cos(t + prev_phase) * coupling,
                            sin(t + prev_phase) * coupling * 0.5,
                            sin(t * 0.5 + prev_phase) * coupling * 0.3
                        );
                    }
                    
                    v_world_pos = pos;
                    
                    vec4 view_pos = u_view * vec4(pos, 1.0);
                    v_depth = -view_pos.z;
                    gl_Position = u_proj * view_pos;
                    
                    // Dynamic coloring based on D-LinOSS parameters and time
                    float color_factor = (sin(u_time + phase) + 1.0) * 0.5;
                    float depth_factor = (pos.z + 3.0) / 6.0;
                    
                    vec3 color1 = vec3(1.0, 0.3, 0.5);  // Neural red
                    vec3 color2 = vec3(0.3, 0.7, 1.0);  // Synapse blue
                    vec3 color3 = vec3(0.7, 1.0, 0.3);  // Activity green
                    
                    if (normalized_id < 0.33) {
                        v_color = mix(color1, color2, color_factor);
                    } else if (normalized_id < 0.66) {
                        v_color = mix(color2, color3, color_factor);
                    } else {
                        v_color = mix(color3, color1, color_factor);
                    }
                    
                    v_color *= 0.7 + 0.3 * depth_factor;
                    
                    // Dynamic point size with D-LinOSS parameter influence
                    float base_size = 6.0;
                    float param_variation = 2.0 * (u_alpha + u_beta) / 4.0;
                    float time_variation = 2.0 * (sin(u_time * 2.0 + phase) + 1.0);
                    float depth_scale = max(0.3, 1.0 / (1.0 + v_depth * 0.05));
                    
                    gl_PointSize = (base_size + param_variation + time_variation) * depth_scale;
                }
            "#);
            
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
                    
                    float center_core = 1.0 - smoothstep(0.0, 0.1, dist);
                    float main_body = 1.0 - smoothstep(0.1, 0.35, dist);
                    float outer_glow = 1.0 - smoothstep(0.35, 0.5, dist);
                    
                    float intensity = center_core * 1.0 + main_body * 0.8 + outer_glow * 0.3;
                    
                    float pulse = 0.9 + 0.1 * sin(v_depth * 0.5 + v_world_pos.x * 2.0);
                    intensity *= pulse;
                    
                    float depth_factor = 1.0 / (1.0 + v_depth * 0.03);
                    float depth_alpha = 0.8 + 0.2 * depth_factor;
                    
                    vec3 enhanced_color = v_color;
                    float highlight = center_core * 0.4;
                    enhanced_color = mix(enhanced_color, vec3(1.0, 1.0, 1.0), highlight);
                    
                    vec3 final_color = enhanced_color * intensity * depth_factor;
                    float alpha = outer_glow * depth_alpha;
                    
                    final_color += v_color * 0.1 * outer_glow;
                    
                    out_color = vec4(final_color, alpha);
                }
            "#);
            
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
        positions: &[[f32; 3]],
        time: f32,
        params: &DLinossParams,
    ) {
        use glow::HasContext as _;
        
        unsafe {
            gl.clear(glow::DEPTH_BUFFER_BIT);
            gl.enable(glow::DEPTH_TEST);
            gl.enable(glow::BLEND);
            gl.blend_func(glow::SRC_ALPHA, glow::ONE_MINUS_SRC_ALPHA);
            gl.enable(glow::PROGRAM_POINT_SIZE);
            
            gl.use_program(Some(self.program));
            
            // Set matrix uniforms
            if let Some(loc) = gl.get_uniform_location(self.program, "u_view") {
                gl.uniform_matrix_4_f32_slice(Some(&loc), false, view_matrix);
            }
            
            if let Some(loc) = gl.get_uniform_location(self.program, "u_proj") {
                gl.uniform_matrix_4_f32_slice(Some(&loc), false, proj_matrix);
            }
            
            // Set D-LinOSS parameters as uniforms
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
            
            if let Some(loc) = gl.get_uniform_location(self.program, "u_oscillator_count") {
                gl.uniform_1_i32(Some(&loc), positions.len() as i32);
            }
            
            gl.bind_vertex_array(Some(self.vertex_array));
            gl.draw_arrays(glow::POINTS, 0, positions.len() as i32);
            
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

// Matrix helper functions (from neural_3d_opengl.rs)
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

// Main application structure
struct VisualApp {
    opengl_renderer: Arc<Mutex<Option<OpenGLRenderer>>>,
    neural_state: NeuralOscillatorState,
    dlinoss_params: DLinossParams,
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

impl VisualApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let gl = cc.gl.as_ref().expect("OpenGL context required");
        
        let renderer = match OpenGLRenderer::new(gl) {
            Ok(r) => Some(r),
            Err(e) => {
                log::error!("Failed to create OpenGL renderer: {}", e);
                None
            }
        };
        
        let params = DLinossParams::default();
        let neural_state = NeuralOscillatorState::new(params.oscillator_count);
        let plot_data = PlotData::new(params.oscillator_count);
        
        Self {
            opengl_renderer: Arc::new(Mutex::new(renderer)),
            neural_state,
            dlinoss_params: params,
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
        
        // Handle mouse rotation
        if response.dragged() {
            let delta = response.drag_delta();
            self.camera_rotation_y += delta.x * 0.01;
            self.camera_rotation_x += delta.y * 0.01;
            self.camera_rotation_x = self.camera_rotation_x.clamp(-std::f32::consts::FRAC_PI_2 + 0.1, std::f32::consts::FRAC_PI_2 - 0.1);
        }
        
        // Handle zoom
        if let Some(hover_pos) = response.hover_pos() {
            if rect.contains(hover_pos) {
                ui.input(|i| {
                    let scroll = i.smooth_scroll_delta.y;
                    if scroll != 0.0 {
                        self.camera_distance *= (1.0 - scroll * 0.001).clamp(0.5, 2.0);
                        self.camera_distance = self.camera_distance.clamp(2.0, 20.0);
                    }
                });
            }
        }
        
        response
    }
    
    fn render_3d(&mut self, ui: &mut egui::Ui, rect: egui::Rect) {
        let opengl_renderer = self.opengl_renderer.clone();
        let aspect_ratio = rect.width() / rect.height();
        
        // Build camera matrices
        let eye = [
            self.camera_distance * self.camera_rotation_x.cos() * self.camera_rotation_y.sin(),
            self.camera_distance * self.camera_rotation_x.sin(),
            self.camera_distance * self.camera_rotation_x.cos() * self.camera_rotation_y.cos(),
        ];
        let target = [0.0, 0.0, 0.0];
        let up = [0.0, 1.0, 0.0];
        
        let view_matrix = look_at_matrix(eye, target, up);
        let proj_matrix = perspective_matrix(45.0_f32.to_radians(), aspect_ratio, 0.1, 100.0);
        
        let positions = self.neural_state.get_positions_as_vec();
        let time = self.neural_state.time;
        let params = self.dlinoss_params.clone();
        
        let callback = egui::PaintCallback {
            rect,
            callback: Arc::new(egui_glow::CallbackFn::new(move |_info, painter| {
                if let Some(renderer) = opengl_renderer.lock().as_mut() {
                    renderer.paint(painter.gl(), &view_matrix, &proj_matrix, &positions, time, &params);
                }
            })),
        };
        
        ui.painter().add(callback);
    }
}

impl eframe::App for VisualApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Update neural simulation using Burn
        if self.simulation_running {
            self.neural_state.update(&self.dlinoss_params);
            
            // Update plot data
            let signals = self.neural_state.get_signal_values();
            self.plot_data.update(self.neural_state.time, &signals);
        }
        
        // Top panel for title and controls
        egui::TopBottomPanel::top("title").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("🧠 3D Neural Oscillator: D-LinOSS + OpenGL + Burn");
                ui.separator();
                if ui.button(if self.simulation_running { "⏸ Pause" } else { "▶ Play" }).clicked() {
                    self.simulation_running = !self.simulation_running;
                }
                ui.separator();
                ui.checkbox(&mut self.show_controls, "Controls");
                ui.checkbox(&mut self.show_plots, "Plots");
            });
        });
        
        // Left panel for D-LinOSS controls
        if self.show_controls {
            egui::SidePanel::left("controls").show(ctx, |ui| {
                ui.heading("D-LinOSS Parameters");
                
                ui.add(egui::Slider::new(&mut self.dlinoss_params.alpha, 0.1..=3.0).text("Alpha"));
                ui.add(egui::Slider::new(&mut self.dlinoss_params.beta, 0.1..=3.0).text("Beta"));
                ui.add(egui::Slider::new(&mut self.dlinoss_params.gamma, 0.1..=2.0).text("Gamma"));
                ui.add(egui::Slider::new(&mut self.dlinoss_params.delta, 0.0..=1.0).text("Delta (Coupling)"));
                ui.add(egui::Slider::new(&mut self.dlinoss_params.frequency, 0.1..=3.0).text("Frequency"));
                ui.add(egui::Slider::new(&mut self.dlinoss_params.amplitude, 0.1..=3.0).text("Amplitude"));
                
                ui.separator();
                ui.heading("Camera");
                ui.add(egui::Slider::new(&mut self.camera_distance, 2.0..=20.0).text("Distance"));
                
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
                ui.label(format!("⚙️ Oscillators: {}", self.dlinoss_params.oscillator_count));
                ui.label(format!("🔥 Backend: Burn NdArray"));
                ui.label(format!("📐 Time: {:.2}s", self.neural_state.time));
            });
        }
        
        // Bottom panel for plotting
        if self.show_plots {
            egui::TopBottomPanel::bottom("plots").show(ctx, |ui| {
                ui.heading("📈 Neural Signal Time Series");
                
                if !self.plot_data.time_series.is_empty() {
                    Plot::new("neural_signals")
                        .height(200.0)
                        .legend(egui_plot::Legend::default())
                        .show(ui, |plot_ui| {
                            // Show first 8 oscillators for clarity
                            let colors = [
                                egui::Color32::RED,
                                egui::Color32::BLUE,
                                egui::Color32::GREEN,
                                egui::Color32::from_rgb(255, 165, 0), // Orange
                                egui::Color32::from_rgb(128, 0, 128), // Purple
                                egui::Color32::from_rgb(165, 42, 42), // Brown
                                egui::Color32::from_rgb(255, 192, 203), // Pink
                                egui::Color32::GRAY,
                            ];
                            
                            for (i, signals) in self.plot_data.oscillator_signals.iter().enumerate().take(8) {
                                if !signals.is_empty() && self.plot_data.time_series.len() == signals.len() {
                                    let points: PlotPoints = self.plot_data.time_series.iter()
                                        .zip(signals.iter())
                                        .map(|(&time, &value)| [time as f64, value as f64])
                                        .collect();
                                    
                                    let line = Line::new(points)
                                        .color(colors[i % colors.len()])
                                        .width(1.5)
                                        .name(format!("Osc {}", i));
                                    
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
            
            // Handle 3D input
            let _response = self.handle_3d_input(ui, available_rect);
            
            // Render 3D scene
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
            .with_inner_size([1200.0, 800.0])
            .with_title("3D Neural Oscillator: D-LinOSS + OpenGL + Burn"),
        multisampling: 4,
        renderer: eframe::Renderer::Glow,
        ..Default::default()
    };
    
    eframe::run_native(
        "3D Neural Visualizer",
        options,
        Box::new(|cc| Ok(Box::new(VisualApp::new(cc)))),
    )
}
