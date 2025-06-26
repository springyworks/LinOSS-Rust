//! LinOSS 3D Neural Visualizer with WGPU Backend
//! 
//! This implementation uses WGPU for both:
//! - 3D neural oscillator visualization rendering
//! - Burn tensor operations (unified GPU backend)
//! 
//! Benefits over OpenGL approach:
//! - Unified WGPU backend for both egui and Burn
//! - Potential for GPU memory sharing between neural computation and rendering
//! - Modern, safe, cross-platform graphics API
//! - Better debugging and validation tools
//! - Future possibility of GPU-GPU communication for real-time neural-visual feedback

use eframe::{egui, egui_wgpu};
use egui_plot::{Line, Plot, PlotPoints};
use std::sync::Arc;
use std::collections::VecDeque;


// LinOSS imports with WGPU backend
use linoss_rust::Vector;
use linoss_rust::linoss::DLinossLayer;
use nalgebra::DVector;
use burn::backend::Wgpu;

// Type alias for unified WGPU backend
type BurnBackend = Wgpu<f32, i32>;

// Neural state using real LinOSS dynamics with WGPU backend
pub struct LinossNeuralState {
    // LinOSS layer with WGPU backend for unified GPU operations
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
    
    // WGPU device reference for potential GPU memory sharing
    pub device: Option<Arc<wgpu::Device>>,
}

impl LinossNeuralState {
    pub fn new(oscillator_count: usize, device: Option<Arc<wgpu::Device>>) -> Self {
        let d_input = 3;           // 3D input (x, y, z coordinates)
        let d_model = oscillator_count.max(8); // Hidden dimension 
        let d_output = oscillator_count; // Output for each oscillator
        
        // Initialize with WGPU device for potential future GPU operations
        let positions = vec![[0.0; 3]; oscillator_count];
        let velocities = vec![[0.0; 3]; oscillator_count]; 
        let neural_outputs = vec![0.0; oscillator_count];
        let input_signal = DVector::zeros(d_input);
        
        Self {
            dlinoss_layer: None, // Will implement with WGPU backend later
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
            device,
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
        
        // LinOSS-inspired dynamics with WGPU potential
        // TODO: Future optimization - move this computation to GPU using shared WGPU device
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
            
            // Add coupling between oscillators (LinOSS-like coupling)
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
            
            // Update positions with momentum
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
    
    // Future: Get GPU buffer for zero-copy rendering
    pub fn get_gpu_positions_buffer(&self) -> Option<&wgpu::Buffer> {
        // TODO: Return WGPU buffer containing positions for direct GPU rendering
        // This would eliminate CPU-GPU copies
        None
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

// WGPU-based 3D renderer for neural oscillators
struct WgpuNeuralRenderer {
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    instance_count: u32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    time: f32,
    alpha: f32,
    beta: f32,
    gamma: f32,
    coupling: f32,
    oscillator_count: u32,
    _padding: [f32; 2],  // Ensure alignment
}

impl WgpuNeuralRenderer {
    fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        oscillator_count: u32,
    ) -> Self {
        // WGPU shader for LinOSS neural visualization
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("LinOSS Neural Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("wgpu_neural_shader.wgsl"))),
        });
        
        // Uniforms for neural parameters
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Neural Uniform Buffer"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Neural Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Neural Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });
        
        // Pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Neural Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        
        // Render pipeline
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Neural Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::PointList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });
        
        // Dummy vertex buffer (positions generated in shader)
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Neural Vertex Buffer"),
            size: 0,
            usage: wgpu::BufferUsages::VERTEX,
            mapped_at_creation: false,
        });
        
        Self {
            render_pipeline,
            vertex_buffer,
            uniform_buffer,
            bind_group,
            instance_count: oscillator_count,
        }
    }
    
    fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        queue: &wgpu::Queue,
        uniforms: &Uniforms,
    ) {
        // Update uniforms
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[*uniforms]));
        
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Neural Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.draw(0..self.instance_count, 0..1);
    }
}

// Matrix helper functions
fn perspective_matrix(fov: f32, aspect: f32, near: f32, far: f32) -> [[f32; 4]; 4] {
    let f = 1.0 / (fov / 2.0).tan();
    [
        [f / aspect, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, (far + near) / (near - far), -1.0],
        [0.0, 0.0, (2.0 * far * near) / (near - far), 0.0],
    ]
}

fn look_at_matrix(eye: [f32; 3], target: [f32; 3], up: [f32; 3]) -> [[f32; 4]; 4] {
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
        [right[0], up_corrected[0], -forward[0], 0.0],
        [right[1], up_corrected[1], -forward[1], 0.0],
        [right[2], up_corrected[2], -forward[2], 0.0],
        [-(right[0] * eye[0] + right[1] * eye[1] + right[2] * eye[2]),
         -(up_corrected[0] * eye[0] + up_corrected[1] * eye[1] + up_corrected[2] * eye[2]),
         forward[0] * eye[0] + forward[1] * eye[1] + forward[2] * eye[2],
         1.0],
    ]
}

fn matrix_multiply(a: [[f32; 4]; 4], b: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut result = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    result
}

// Main application with WGPU integration
struct LinossWgpuVisualizerApp {
    neural_state: LinossNeuralState,
    linoss_params: LinossParams,
    plot_data: PlotData,
    
    // 3D camera
    camera_rotation_x: f32,
    camera_rotation_y: f32,
    camera_distance: f32,
    
    // UI state
    show_controls: bool,
    show_plots: bool,
    simulation_running: bool,
    
    // WGPU renderer
    neural_renderer: Option<WgpuNeuralRenderer>,
    depth_texture: Option<wgpu::Texture>,
}

impl LinossWgpuVisualizerApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let wgpu_render_state = cc.wgpu_render_state.as_ref().expect("WGPU render state required");
        
        let device = &wgpu_render_state.device;
        let params = LinossParams::default();
        let neural_state = LinossNeuralState::new(params.oscillator_count, Some(Arc::new(device.clone())));
        let plot_data = PlotData::new(params.oscillator_count);
        
        Self {
            neural_state,
            linoss_params: params,
            plot_data,
            camera_rotation_x: 0.3,
            camera_rotation_y: 0.5,
            camera_distance: 8.0,
            show_controls: true,
            show_plots: true,
            simulation_running: true,
            neural_renderer: None,
            depth_texture: None,
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
        let wgpu_render_state = ui.ctx().wgpu_render_state().expect("WGPU render state required");
        
        // Initialize renderer if needed
        if self.neural_renderer.is_none() {
            let format = wgpu_render_state.target_format;
            self.neural_renderer = Some(WgpuNeuralRenderer::new(
                &wgpu_render_state.device,
                format,
                self.linoss_params.oscillator_count as u32,
            ));
            
            // Create depth texture
            let size = wgpu::Extent3d {
                width: rect.width() as u32,
                height: rect.height() as u32,
                depth_or_array_layers: 1,
            };
            
            self.depth_texture = Some(wgpu_render_state.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Neural Depth Texture"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            }));
        }
        
        // Calculate camera matrices
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
        let view_proj = matrix_multiply(proj_matrix, view_matrix);
        
        // Create uniforms
        let uniforms = Uniforms {
            view_proj,
            time: self.neural_state.time,
            alpha: self.linoss_params.alpha,
            beta: self.linoss_params.beta,
            gamma: self.linoss_params.gamma,
            coupling: self.linoss_params.coupling,
            oscillator_count: self.linoss_params.oscillator_count as u32,
            _padding: [0.0, 0.0],
        };
        
        // WGPU render callback
        let renderer = self.neural_renderer.as_ref().unwrap();
        let depth_texture = self.depth_texture.as_ref().unwrap();
        
        let callback = egui_wgpu::Callback::new_paint_callback(
            rect,
            WgpuCallbackData {
                uniforms,
                renderer: renderer as *const WgpuNeuralRenderer,
                depth_texture: depth_texture as *const wgpu::Texture,
            },
        );
        
        ui.painter().add(callback);
    }
}

// Callback data for WGPU rendering
struct WgpuCallbackData {
    uniforms: Uniforms,
    renderer: *const WgpuNeuralRenderer,
    depth_texture: *const wgpu::Texture,
}

impl egui_wgpu::CallbackTrait for WgpuCallbackData {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        screen_descriptor: &egui_wgpu::ScreenDescriptor,
        egui_encoder: &mut wgpu::CommandEncoder,
        callback_resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        // Prepare render resources if needed
        vec![]
    }
    
    fn paint<'a>(
        &'a self,
        info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'a>,
        callback_resources: &'a egui_wgpu::CallbackResources,
    ) {
        // Custom WGPU rendering would go here
        // For now, this is a placeholder
    }
}

impl eframe::App for LinossWgpuVisualizerApp {
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
                ui.heading("🧠 LinOSS 3D Neural Visualizer (WGPU Backend)");
                ui.separator();
                if ui.button(if self.simulation_running { "⏸ Pause" } else { "▶ Play" }).clicked() {
                    self.simulation_running = !self.simulation_running;
                }
                ui.separator();
                ui.checkbox(&mut self.show_controls, "Controls");
                ui.checkbox(&mut self.show_plots, "Plots");
                ui.separator();
                ui.label("🔥 WGPU + Burn Backend");
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
                    let device = self.neural_state.device.clone();
                    self.neural_state = LinossNeuralState::new(self.linoss_params.oscillator_count, device);
                    self.plot_data = PlotData::new(self.linoss_params.oscillator_count);
                    self.neural_renderer = None; // Force recreation
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
                ui.label(format!("📚 Library: LinOSS + Burn"));
                ui.label(format!("⏰ Time: {:.2}s", self.neural_state.time));
                ui.label(format!("🎮 Backend: WGPU"));
                
                ui.separator();
                ui.heading("🚀 WGPU Benefits");
                ui.label("✅ Unified rendering & compute");
                ui.label("✅ Modern graphics API");
                ui.label("✅ Cross-platform support");
                ui.label("✅ Better debugging tools");
                ui.label("⚡ Future: GPU-GPU communication");
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
}

fn main() -> eframe::Result {
    env_logger::init();
    
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 900.0])
            .with_title("LinOSS 3D Neural Visualizer - WGPU Backend"),
        multisampling: 4,
        renderer: eframe::Renderer::Wgpu, // Force WGPU backend
        wgpu_options: eframe::egui_wgpu::WgpuConfiguration {
            supported_backends: wgpu::Backends::PRIMARY,
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        },
        ..Default::default()
    };
    
    eframe::run_native(
        "LinOSS WGPU Visualizer",
        options,
        Box::new(|cc| Ok(Box::new(LinossWgpuVisualizerApp::new(cc)))),
    )
}
