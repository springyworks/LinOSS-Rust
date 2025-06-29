use wasm_bindgen::prelude::*;
use std::collections::VecDeque;

// egui imports for web UI
use eframe::wasm_bindgen;
use eframe::web_sys;

// Conditional imports based on features (following copilot.md pattern)
#[cfg(feature = "burn")]
use burn_tensor::{Tensor, TensorData};

#[cfg(feature = "burn")]
use burn_ndarray::{NdArray, NdArrayDevice};

#[cfg(feature = "linoss")]
use linoss_rust::linoss::dlinoss_layer::DLinossLayer;

use egui::{Color32, Pos2, Stroke};

// Type alias for our WASM-compatible backend
#[cfg(feature = "burn")]
type Backend = NdArray<f32>;

// JavaScript-compatible demo class for HTML interfaces
#[wasm_bindgen]
pub struct DLinOSSDemo {
    #[cfg(feature = "burn")]
    device: NdArrayDevice,
    
    #[cfg(feature = "linoss")]
    layer: Option<DLinossLayer<Backend>>,
    
    // Fallback for non-burn builds
    #[cfg(not(feature = "burn"))]
    mock_weights: Vec<f32>,
}

#[wasm_bindgen]
impl DLinOSSDemo {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        #[cfg(feature = "burn")]
        {
            let device = NdArrayDevice::Cpu;
            
            #[cfg(feature = "linoss")]
            {
                use linoss_rust::linoss::dlinoss_layer::DLinossLayerConfig;
                
                web_sys::console::log_1(&"🚀 Creating DLinOSSDemo with real D-LinOSS layer".into());
                
                let config = DLinossLayerConfig::new_dlinoss(
                    10,  // d_input
                    32,  // d_model (hidden dimension)
                    10,  // d_output
                );
                
                let layer = Some(DLinossLayer::new(&config, &device));
                
                Self { device, layer }
            }
            
            #[cfg(not(feature = "linoss"))]
            {
                web_sys::console::log_1(&"🔧 Creating DLinOSSDemo with Burn tensors only".into());
                
                Self { device }
            }
        }
        
        #[cfg(not(feature = "burn"))]
        {
            web_sys::console::log_1(&"📝 Creating DLinOSSDemo with mock implementation".into());
            
            Self {
                mock_weights: vec![0.5, -0.3, 0.8, -0.2, 0.1, 0.6, -0.4, 0.9, -0.1, 0.7],
            }
        }
    }
    
    #[wasm_bindgen]
    pub fn forward(&self, input_data: &[f32]) -> Vec<f32> {
        #[cfg(feature = "linoss")]
        {
            if let Some(ref layer) = self.layer {
                web_sys::console::log_1(&format!("🧠 Running real D-LinOSS forward pass with {} inputs", input_data.len()).into());
                
                // Ensure we have exactly 10 inputs for the layer
                let mut padded_input = input_data.to_vec();
                while padded_input.len() < 10 {
                    padded_input.push(0.0);
                }
                padded_input.truncate(10);
                
                // Create input tensor [1, 1, 10] (batch, sequence, features)
                let input_tensor = Tensor::<Backend, 3>::from_data(
                    TensorData::new(padded_input, [1, 1, 10]),
                    &self.device,
                );
                
                let output_tensor = layer.forward(input_tensor);
                
                // Convert output back to Vec<f32>
                let output_data = output_tensor.to_data();
                let output_slice = output_data.as_slice::<f32>().unwrap();
                
                return output_slice.to_vec();
            } else {
                web_sys::console::log_1(&"⚠️ D-LinOSS layer not initialized, using fallback processing".into());
                
                // Fallback processing when layer is None
                let fallback_weights = vec![0.5, -0.3, 0.8, -0.2, 0.1, 0.6, -0.4, 0.9, -0.1, 0.7];
                let mut result = Vec::new();
                for (i, &input) in input_data.iter().enumerate().take(10) {
                    let weight = fallback_weights.get(i).copied().unwrap_or(0.5);
                    let output = (input * weight).sin() * 0.8 + input * 0.2;
                    result.push(output);
                }
                
                // Pad to 10 outputs if needed
                while result.len() < 10 {
                    result.push(0.0);
                }
                
                return result;
            }
        }
        
        #[cfg(all(feature = "burn", not(feature = "linoss")))]
        {
            web_sys::console::log_1(&"🔧 Running Burn tensor processing".into());
            
            // Simple tensor processing with Burn
            let input_size = input_data.len().min(10);
            let mut processed_input = input_data[..input_size].to_vec();
            while processed_input.len() < 10 {
                processed_input.push(0.0);
            }
            
            let input_tensor = Tensor::<Backend, 1>::from_data(
                TensorData::new(processed_input, [10]),
                &self.device,
            );
            
            // Simple transformation: x * sin(x) + 0.1
            let transformed = input_tensor.clone() * input_tensor.sin() + Tensor::from_data([0.1; 10], &self.device);
            
            let output_data = transformed.to_data();
            let output_slice = output_data.as_slice::<f32>().unwrap();
            
            return output_slice.to_vec();
        }
        
        #[cfg(not(feature = "burn"))]
        {
            web_sys::console::log_1(&"📝 Running mock processing".into());
            
            // Mock implementation: simple mathematical transformation
            let mut result = Vec::new();
            for (i, &input) in input_data.iter().enumerate().take(10) {
                let weight = self.mock_weights.get(i).copied().unwrap_or(0.5);
                let output = (input * weight).sin() * 0.8 + input * 0.2;
                result.push(output);
            }
            
            // Pad to 10 outputs if needed
            while result.len() < 10 {
                result.push(0.0);
            }
            
            return result;
        }
    }
    
    #[wasm_bindgen]
    pub fn get_info(&self) -> String {
        #[cfg(feature = "linoss")]
        {
            return "D-LinOSS Neural Dynamics Layer with real Burn backend running in WASM".to_string();
        }
        
        #[cfg(all(feature = "burn", not(feature = "linoss")))]
        {
            return "Burn Tensor Processing with NdArray backend in WASM".to_string();
        }
        
        #[cfg(not(feature = "burn"))]
        {
            return "Mock D-LinOSS Demo for WASM compatibility testing".to_string();
        }
    }
}

#[wasm_bindgen(start)]
pub fn start() {
    // Prevent double initialization
    static INIT: std::sync::Once = std::sync::Once::new();
    
    INIT.call_once(|| {
        // Add extensive debug logging
        web_sys::console::log_1(&"🚀 Starting D-LinOSS Web Demo...".into());
        
        console_error_panic_hook::set_once();
        web_sys::console::log_1(&"✅ Panic hook set".into());
        
        // Initialize logger
        wasm_logger::init(wasm_logger::Config::default());
        web_sys::console::log_1(&"📝 Logger initialized".into());
        
        let web_options = eframe::WebOptions::default();
        
        web_sys::console::log_1(&"🔧 Creating eframe runner...".into());
        
        wasm_bindgen_futures::spawn_local(async {
            web_sys::console::log_1(&"🔄 Inside async block...".into());
            
            let document = web_sys::window()
                .expect("No window")
                .document()
                .expect("No document");
                
            web_sys::console::log_1(&"📄 Got document".into());
            
            let canvas = document
                .get_element_by_id("linoss_canvas")  // Match HTML canvas ID
                .expect("No canvas found with id 'linoss_canvas'")
                .dyn_into::<web_sys::HtmlCanvasElement>()
                .expect("Element is not a canvas");
                
            web_sys::console::log_1(&"🎨 Got canvas element".into());
            
            let runner = eframe::WebRunner::new();
            web_sys::console::log_1(&"🏃 WebRunner created".into());
            
            let start_result = runner.start(
                canvas,
                web_options,
                Box::new(|cc| {
                    web_sys::console::log_1(&"📦 Inside app creator closure...".into());
                    let app = DLinossApp::new(cc);
                    web_sys::console::log_1(&"✅ App created successfully".into());
                    Ok(Box::new(app))
                }),
            )
            .await;
            
            match start_result {
                Ok(_) => {
                    web_sys::console::log_1(&"🎉 D-LinOSS Web Demo started successfully!".into());
                }
                Err(e) => {
                    web_sys::console::error_1(&format!("❌ Failed to start: {:?}", e).into());
                }
            }
        });
        
        web_sys::console::log_1(&"🧠 D-LinOSS Web Demo initialized!".into());
    });
}

// D-LinOSS egui Application
pub struct DLinossApp {
    #[cfg(feature = "linoss")]
    layer: Option<DLinossLayer<Backend>>,
    #[cfg(feature = "burn")]
    device: NdArrayDevice,
    
    // Simulation state
    time: f32,
    running: bool,
    frequency: f32,
    damping: f32,
    coupling: f32,
    
    // Visualization data
    phase_data: VecDeque<[f64; 2]>,
    signal_data: VecDeque<[f64; 2]>,
    max_points: usize,
    
    // 3D visualization state
    rotation_x: f32,
    rotation_y: f32,
    zoom: f32,
    trajectory_3d: VecDeque<[f32; 3]>, // Store 3D points
}

impl DLinossApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        web_sys::console::log_1(&"🔧 Initializing D-LinOSS components...".into());
        
        #[cfg(feature = "linoss")]
        {
            use linoss_rust::linoss::dlinoss_layer::DLinossLayerConfig;
            
            let device = NdArrayDevice::Cpu;
            web_sys::console::log_1(&"🚀 Creating D-LinOSS layer with real neural dynamics...".into());
            
            let config = DLinossLayerConfig::new_dlinoss(
                2,   // d_input (x, y coordinates)
                16,  // d_model (hidden dimension, should be even for oscillatory pairs)
                2,   // d_output (phase space output)
            );
            
            let layer = Some(DLinossLayer::new(&config, &device));
            
            web_sys::console::log_1(&"✅ D-LinOSS layer created successfully!".into());
            
            Self {
                layer,
                device,
                time: 0.0,
                running: true,
                frequency: 2.0,
                damping: 0.1,
                coupling: 0.5,
                phase_data: VecDeque::new(),
                signal_data: VecDeque::new(),
                max_points: 1000,
                rotation_x: 0.3,
                rotation_y: 0.5,
                zoom: 2.5,  // Start more zoomed in for better initial view
                trajectory_3d: VecDeque::new(),
            }
        }
        
        #[cfg(all(feature = "burn", not(feature = "linoss")))]
        {
            let device = NdArrayDevice::Cpu;
            web_sys::console::log_1(&"🔧 Creating minimal Burn tensor demo...".into());
            
            Self {
                device,
                time: 0.0,
                running: true,
                frequency: 2.0,
                damping: 0.1,
                coupling: 0.5,
                phase_data: VecDeque::new(),
                signal_data: VecDeque::new(),
                max_points: 1000,
                rotation_x: 0.3,
                rotation_y: 0.5,
                zoom: 2.5,  // Start more zoomed in for better initial view
                trajectory_3d: VecDeque::new(),
            }
        }
        
        #[cfg(not(feature = "burn"))]
        {
            web_sys::console::log_1(&"📝 Creating mock demo (no Burn features)...".into());
            
            Self {
                time: 0.0,
                running: true,
                frequency: 2.0,
                damping: 0.1,
                coupling: 0.5,
                phase_data: VecDeque::new(),
                signal_data: VecDeque::new(),
                max_points: 1000,
                rotation_x: 0.3,
                rotation_y: 0.5,
                zoom: 2.5,  // Start more zoomed in for better initial view
                trajectory_3d: VecDeque::new(),
            }
        }
    }
    
    fn update_simulation(&mut self) {
        if !self.running {
            return;
        }
        
        let dt = 0.016; // ~60 FPS
        self.time += dt;
        
        #[cfg(feature = "linoss")]
        {
            // Create input signal with frequency and damping effects
            let damped_amplitude = (-self.damping * self.time).exp();
            let input_x = self.frequency * self.time.cos() * damped_amplitude;
            let input_y = self.frequency * self.time.sin() * 0.7 * damped_amplitude;
            
            // Process through D-LinOSS
            let input_data = vec![input_x, input_y];
            let input_tensor = Tensor::<Backend, 2>::from_data(
                TensorData::new(input_data, [1, 2]),
                &self.device,
            );
            
            if let Some(ref layer) = &self.layer {
                // D-LinOSS expects 3D input: [batch, sequence, features]
                // Our input_tensor is [1, 2], we need to add sequence dimension to get [1, 1, 2]
                let input_3d = input_tensor.unsqueeze_dim::<3>(1); // Add sequence dimension: [1, 1, 2]
                
                let output_3d = layer.forward(input_3d);
                
                // Output is [1, 1, d_output], squeeze to get [d_output]
                let output = output_3d.squeeze::<2>(0).squeeze::<1>(0);
                
                // Convert tensor data for visualization
                let tensor_data = output.to_data();
                let result = tensor_data.as_slice::<f32>().unwrap();
                
                // Apply coupling to the neural output and mix with input
                let coupled_x = result[0] as f64 * self.coupling as f64 + input_x as f64 * (1.0 - self.coupling as f64);
                let coupled_y = result[1] as f64 * self.coupling as f64 + input_y as f64 * (1.0 - self.coupling as f64);
                
                // Store the phase space coordinates
                self.phase_data.push_back([coupled_x, coupled_y]);
                
                // Also store signal data for the time series plot
                self.signal_data.push_back([self.time as f64, coupled_x]);
                
                if self.phase_data.len() > self.max_points {
                    self.phase_data.pop_front();
                }
            }
        }
        
        #[cfg(all(feature = "burn", not(feature = "linoss")))]
        {
            // Simple oscillator simulation using Burn tensors with proper parameter effects
            let damped_amplitude = (-self.damping * self.time).exp();
            let input_data = vec![
                self.frequency * self.time.cos() * damped_amplitude,
                self.frequency * self.time.sin() * damped_amplitude,
            ];
            
            let input_tensor = Tensor::<Backend, 1>::from_data(
                TensorData::new(input_data, [2]),
                &self.device,
            );
            
            // Apply coupling and scaling transformation
            let scaled = input_tensor * Tensor::from_data([self.coupling, self.coupling * 0.8], &self.device);
            let tensor_data = scaled.to_data();
            let result = tensor_data.as_slice::<f32>().unwrap();
            
            self.phase_data.push_back([result[0] as f64, result[1] as f64]);
            self.signal_data.push_back([self.time as f64, result[0] as f64]);
        }
        
        #[cfg(not(feature = "burn"))]
        {
            // Mock oscillator with proper parameter effects
            let damped_amplitude = (-self.damping * self.time).exp();
            let x = self.frequency * self.time.cos() * damped_amplitude * self.coupling;
            let y = self.frequency * self.time.sin() * damped_amplitude * self.coupling * 0.8;
            
            self.phase_data.push_back([x as f64, y as f64]);
            self.signal_data.push_back([self.time as f64, x as f64]);
        }
        
        // Limit data points
        if self.phase_data.len() > self.max_points {
            self.phase_data.pop_front();
        }
        if self.signal_data.len() > self.max_points {
            self.signal_data.pop_front();
        }
        
        // After updating phase_data and signal_data, add 3D trajectory
        if let Some(&[x, y]) = self.phase_data.back() {
            // Create a 3D trajectory using phase space + time dimension with frequency variation
            let z = (self.time * self.frequency * 0.3).sin() * 0.5 * (-self.damping * self.time * 0.5).exp(); // Z varies with frequency and damping
            self.trajectory_3d.push_back([x as f32, y as f32, z]);
            
            if self.trajectory_3d.len() > self.max_points {
                self.trajectory_3d.pop_front();
            }
        }
    }
    
    fn project_3d_to_2d(&self, point: [f32; 3], center: Pos2, scale: f32) -> (Pos2, f32) {
        // Enhanced 3D to 2D projection with rotation, zoom, and viewing distance
        let [x, y, z] = point;
        
        // Apply rotations
        let cos_x = self.rotation_x.cos();
        let sin_x = self.rotation_x.sin();
        let cos_y = self.rotation_y.cos();
        let sin_y = self.rotation_y.sin();
        
        // Rotate around X axis
        let y1 = y * cos_x - z * sin_x;
        let z1 = y * sin_x + z * cos_x;
        
        // Rotate around Y axis
        let x2 = x * cos_y + z1 * sin_y;
        let z2 = -x * sin_y + z1 * cos_y;
        
        // Calculate distance from viewer (camera is positioned at negative Z)
        let camera_distance = 2.0; // Camera distance from origin
        let viewer_distance = camera_distance + z2; // Distance from camera to point
        
        // Perspective projection based on viewer distance
        let perspective = camera_distance / viewer_distance.max(0.1); // Prevent division by zero
        let final_scale = scale * self.zoom * perspective;
        
        let screen_pos = Pos2::new(
            center.x + x2 * final_scale,
            center.y - y1 * final_scale, // Flip Y for screen coordinates
        );
        
        // Return position and depth factor based on actual viewing distance
        // Closer to viewer = larger depth factor (closer to 1.0)
        // Farther from viewer = smaller depth factor (closer to 0.0)
        let depth_factor = (perspective * 2.0).clamp(0.1, 2.0);
        (screen_pos, depth_factor)
    }
    
    fn draw_phase_space_panel(&mut self, ui: &mut egui::Ui, compact: bool) {
        ui.vertical(|ui| {
            ui.heading("Phase Space");
            if !compact {
                ui.separator();
            }
            
            use egui_plot::{Plot, Points, PlotPoints};
            
            // Calculate available space for the plot - use almost all available height
            let available_height = ui.available_height() - 5.0; // Minimal padding
            
            // Use all available space with guaranteed minimum height
            let plot = Plot::new("phase_plot")
                .view_aspect(1.0)
                .data_aspect(1.0)
                .auto_bounds([true, true])
                .height(available_height.max(180.0)); // Increased minimum height
            
            plot.show(ui, |plot_ui| {
                if !self.phase_data.is_empty() {
                    // Draw trajectory
                    let points: PlotPoints = self.phase_data.iter().cloned().collect();
                    plot_ui.points(
                        Points::new(points)
                            .radius(2.0)
                            .color(egui::Color32::from_rgb(100, 150, 250))
                    );
                    
                    // Draw current position
                    if let Some(&last_point) = self.phase_data.back() {
                        let current: PlotPoints = vec![last_point].into();
                        plot_ui.points(
                            Points::new(current)
                                .radius(8.0)
                                .color(egui::Color32::from_rgb(255, 100, 100))
                        );
                    }
                }
            });
        });
    }
    
    fn draw_signal_panel(&mut self, ui: &mut egui::Ui, compact: bool) {
        ui.vertical(|ui| {
            ui.heading("Signal Over Time");
            if !compact {
                ui.separator();
            }
            
            use egui_plot::{Plot, Line, PlotPoints};
            
            // Calculate available space for the plot - use almost all available height
            let available_height = ui.available_height() - 5.0; // Minimal padding
            
            // Use all available space with guaranteed minimum height
            let plot = Plot::new("signal_plot")
                .height(available_height.max(180.0)); // Increased minimum height
            
            plot.show(ui, |plot_ui| {
                if !self.signal_data.is_empty() {
                    let points: PlotPoints = self.signal_data.iter().cloned().collect();
                    plot_ui.line(
                        Line::new(points)
                            .color(egui::Color32::from_rgb(150, 255, 150))
                            .width(2.0)
                    );
                }
            });
        });
    }
    
    fn draw_3d_panel(&mut self, ui: &mut egui::Ui) {
        ui.vertical(|ui| {
            ui.heading("🎯 3D Neural Dynamics");
            ui.separator();
            
            // Interactive area - use most available space
            let rect = ui.available_rect_before_wrap();
            let response = ui.allocate_rect(rect, egui::Sense::click_and_drag());
            
            // Handle interactions
            if response.dragged() {
                let delta = response.drag_delta();
                // More sensitive controls for mobile
                let sensitivity = if ui.available_width() < 800.0 { 0.02 } else { 0.01 };
                self.rotation_y += delta.x * sensitivity;
                self.rotation_x += delta.y * sensitivity;
            }
            
            // Handle scroll wheel zoom
            let scroll_delta = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll_delta != 0.0 {
                self.zoom *= 1.0 + scroll_delta * 0.001;
                self.zoom = self.zoom.clamp(0.1, 5.0); // Limit zoom range
            }
            
            // Handle multi-touch gestures (pinch zoom) - basic implementation
            if let Some(multi_touch) = ui.input(|i| i.multi_touch()) {
                    if multi_touch.zoom_delta != 1.0 {
                        self.zoom *= multi_touch.zoom_delta;
                        self.zoom = self.zoom.clamp(0.1, 5.0);
                    }
                    
                    if multi_touch.rotation_delta != 0.0 {
                        self.rotation_y += multi_touch.rotation_delta * 0.5;
                    }
            }
            
            // Draw 3D visualization
            self.draw_3d_axes(ui, rect);
            
            // Draw trajectory
            let painter = ui.painter();
            let center = rect.center();
            let base_scale = rect.width().min(rect.height()) * 0.3;
            
            // Draw trajectory as connected lines with proper viewing-distance-aware rendering
            if self.trajectory_3d.len() > 1 {
                let trajectory_vec: Vec<[f32; 3]> = self.trajectory_3d.iter().cloned().collect();
                for i in 0..trajectory_vec.len() - 1 {
                    let (p1, depth1) = self.project_3d_to_2d(trajectory_vec[i], center, base_scale);
                    let (p2, depth2) = self.project_3d_to_2d(trajectory_vec[i + 1], center, base_scale);
                    
                    // Color gradient based on time with viewing distance fade
                    let t = i as f32 / trajectory_vec.len() as f32;
                    let avg_depth = (depth1 + depth2) * 0.5;
                    
                    // Alpha based on viewing distance (closer = more opaque)
                    let depth_alpha = (avg_depth * 180.0 + 75.0).clamp(75.0, 255.0) as u8;
                    
                    let color = Color32::from_rgba_premultiplied(
                        (100.0 + 155.0 * t) as u8,
                        100,
                        (255.0 - 155.0 * t) as u8,
                        depth_alpha,
                    );
                    
                    // Line width varies with viewing distance (closer = thicker)
                    let line_width = (0.8 + avg_depth * 1.5).clamp(0.5, 3.0);
                    painter.line_segment([p1, p2], Stroke::new(line_width, color));
                }
                
                // Draw current position with proper viewing-distance-based size scaling
                if let Some(&last_point) = self.trajectory_3d.back() {
                    let (pos, depth_factor) = self.project_3d_to_2d(last_point, center, base_scale);
                    
                    // Size varies based on distance from viewer, not just Z-coordinate
                    let base_radius = 6.0;
                    let radius = (base_radius * depth_factor).clamp(2.0, 15.0);
                    
                    // Brightness and color intensity based on viewing distance
                    let brightness = (depth_factor * 200.0 + 55.0).clamp(55.0, 255.0) as u8;
                    let ball_color = Color32::from_rgb(brightness, brightness, 0); // Yellow with depth brightness
                    
                    painter.circle_filled(pos, radius, ball_color);
                    
                    // Add a subtle outline that also scales with depth
                    let outline_width = (1.0 + depth_factor * 0.5).clamp(0.5, 2.0);
                    painter.circle_stroke(pos, radius, Stroke::new(outline_width, Color32::WHITE));
                }
            }
            
            // Instructions (adaptive to screen size) - positioned at bottom
            ui.allocate_new_ui(
                egui::UiBuilder::new().max_rect(
                    egui::Rect::from_min_size(
                        rect.left_bottom() + egui::Vec2::new(10.0, -40.0),
                        egui::Vec2::new(rect.width() - 20.0, 30.0)
                    )
                ),
                |ui| {
                    ui.horizontal(|ui| {
                        if ui.available_width() < 400.0 {
                            ui.label("Drag: rotate | Pinch: zoom");
                        } else {
                            ui.label("Drag: rotate | Scroll: zoom | Touch: pinch/rotate");
                        }
                        
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.small_button("Reset View").clicked() {
                                self.zoom = 2.5;  // Reset to better default zoom
                                self.rotation_x = 0.3;
                                self.rotation_y = 0.0;
                            }
                        });
                    });
                }
            );
        });
    }
    
    fn draw_3d_axes(&self, ui: &mut egui::Ui, rect: egui::Rect) {
        let painter = ui.painter();
        let center = rect.center();
        let base_scale = rect.width().min(rect.height()) * 0.3;
        
        // Draw axes with perspective-based thickness
        let axes = [
            ([0.0, 0.0, 0.0], [1.2, 0.0, 0.0], Color32::RED, "X"),    // X axis (slightly longer)
            ([0.0, 0.0, 0.0], [0.0, 1.2, 0.0], Color32::GREEN, "Y"),  // Y axis
            ([0.0, 0.0, 0.0], [0.0, 0.0, 1.2], Color32::BLUE, "Z"),   // Z axis
        ];
        
        for (start, end, color, label) in axes {
            let (_p1, _depth1) = self.project_3d_to_2d(start, center, base_scale);
            let (_p2, _depth2) = self.project_3d_to_2d(end, center, base_scale);
            
            // Calculate line thickness based on viewing distance
            // Sample multiple points along the axis for smooth thickness variation
            let segments = 10;
            for i in 0..segments {
                let t1 = i as f32 / segments as f32;
                let t2 = (i + 1) as f32 / segments as f32;
                
                // Interpolate between start and end
                let point1 = [
                    start[0] + t1 * (end[0] - start[0]),
                    start[1] + t1 * (end[1] - start[1]),
                    start[2] + t1 * (end[2] - start[2]),
                ];
                let point2 = [
                    start[0] + t2 * (end[0] - start[0]),
                    start[1] + t2 * (end[1] - start[1]),
                    start[2] + t2 * (end[2] - start[2]),
                ];
                
                let (seg_p1, seg_depth1) = self.project_3d_to_2d(point1, center, base_scale);
                let (seg_p2, seg_depth2) = self.project_3d_to_2d(point2, center, base_scale);
                
                // Average depth for this segment
                let avg_depth = (seg_depth1 + seg_depth2) * 0.5;
                
                // Thickness varies from 1.0 (far) to 4.0 (near) based on viewing distance
                let thickness = (1.0 + avg_depth * 3.0).clamp(1.0, 4.0);
                
                // Alpha varies with depth for additional depth perception
                let alpha = (avg_depth * 150.0 + 105.0).clamp(105.0, 255.0) as u8;
                let depth_color = Color32::from_rgba_premultiplied(color.r(), color.g(), color.b(), alpha);
                
                painter.line_segment([seg_p1, seg_p2], Stroke::new(thickness, depth_color));
            }
            
            // Draw axis labels with depth-based sizing (only if there's enough space)
            if rect.width() > 200.0 {
                let label_offset = 1.3;
                let label_point = [
                    end[0] + (label_offset - 1.2) * (end[0] - start[0]).signum(),
                    end[1] + (label_offset - 1.2) * (end[1] - start[1]).signum(),
                    end[2] + (label_offset - 1.2) * (end[2] - start[2]).signum(),
                ];
                let (label_pos, label_depth) = self.project_3d_to_2d(label_point, center, base_scale);
                
                // Font size varies with depth
                let font_size = (10.0 + label_depth * 6.0).clamp(8.0, 16.0);
                let label_alpha = (label_depth * 150.0 + 105.0).clamp(105.0, 255.0) as u8;
                let label_color = Color32::from_rgba_premultiplied(color.r(), color.g(), color.b(), label_alpha);
                
                painter.text(
                    label_pos,
                    egui::Align2::CENTER_CENTER,
                    label,
                    egui::FontId::proportional(font_size),
                    label_color,
                );
            }
        }
        
        // Draw origin marker with depth-based size
        let (origin_pos, origin_depth) = self.project_3d_to_2d([0.0, 0.0, 0.0], center, base_scale);
        let origin_radius = (2.0 + origin_depth * 3.0).clamp(1.0, 5.0);
        let origin_alpha = (origin_depth * 100.0 + 155.0).clamp(155.0, 255.0) as u8;
        let origin_color = Color32::from_rgba_premultiplied(255, 255, 255, origin_alpha);
        painter.circle_filled(origin_pos, origin_radius, origin_color);
        painter.circle_stroke(origin_pos, origin_radius, Stroke::new(1.0, Color32::BLACK));
    }
}

impl eframe::App for DLinossApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Request continuous repainting for smooth animation
        ctx.request_repaint();
        
        // Update simulation
        self.update_simulation();
        
        // Top panel with controls (more compact)
        egui::TopBottomPanel::top("controls").exact_height(50.0).show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("🧠 D-LinOSS Neural Dynamics");
                
                ui.separator();
                
                if ui.button(if self.running { "⏸ Pause" } else { "▶ Start" }).clicked() {
                    self.running = !self.running;
                }
                
                if ui.button("🔄 Reset").clicked() {
                    self.time = 0.0;
                    self.phase_data.clear();
                    self.signal_data.clear();
                    self.trajectory_3d.clear();
                }
                
                ui.separator();
                
                // More compact controls
                ui.label("Freq:");
                ui.add(egui::Slider::new(&mut self.frequency, 0.1..=10.0));
                
                ui.label("Damp:");
                ui.add(egui::Slider::new(&mut self.damping, 0.0..=2.0));
                
                ui.label("Coup:");
                ui.add(egui::Slider::new(&mut self.coupling, 0.0..=2.0));
            });
        });
        
        // Status bar (compact)
        egui::TopBottomPanel::bottom("status").exact_height(25.0).show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(format!("Time: {:.1}s", self.time));
                ui.separator();
                ui.label(format!("Pts: {}", self.phase_data.len()));
                ui.separator();
                
                #[cfg(feature = "linoss")]
                ui.label("💚 D-LinOSS");
                
                #[cfg(all(feature = "burn", not(feature = "linoss")))]
                ui.label("🔧 Burn");
                
                #[cfg(not(feature = "burn"))]
                ui.label("📝 Mock");
                
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(format!("Zoom: {:.1}x", self.zoom));
                });
            });
        });
        
        // Main content area - 3D on top, 2D graphs side by side below
        egui::CentralPanel::default().show(ctx, |ui| {
            // Calculate heights based on screen size with proper proportions
            let available_height = ui.available_height();
            let is_mobile = ctx.screen_rect().width() < 800.0;
            
            // Reduce 3D space to give more room to bottom panels
            let main_height = if is_mobile { 
                available_height * 0.45  // 45% for 3D on mobile
            } else { 
                available_height * 0.48  // 48% for 3D on desktop
            };
            let helper_height = available_height - main_height - 15.0; // Account for spacing
            
            // 3D panel at the top
            ui.allocate_ui_with_layout(
                egui::vec2(ui.available_width(), main_height),
                egui::Layout::top_down(egui::Align::Center),
                |ui| {
                    ui.group(|ui| {
                        self.draw_3d_panel(ui);
                    });
                }
            );
            
            ui.add_space(3.0); // Minimal spacing
            ui.separator();
            ui.add_space(3.0);
            
            // Bottom section: Phase Space and Signal side by side with guaranteed space
            ui.horizontal(|ui| {
                let available_width = ui.available_width();
                let panel_width = (available_width - 15.0) / 2.0; // Account for spacing
                
                // Phase space on the left
                ui.allocate_ui_with_layout(
                    egui::vec2(panel_width, helper_height),
                    egui::Layout::top_down(egui::Align::LEFT),
                    |ui| {
                        ui.group(|ui| {
                            ui.set_min_height(helper_height - 10.0); // Ensure adequate height
                            self.draw_phase_space_panel(ui, is_mobile);
                        });
                    }
                );
                
                ui.add_space(5.0); // Space between panels
                
                // Signal on the right
                ui.allocate_ui_with_layout(
                    egui::vec2(panel_width, helper_height),
                    egui::Layout::top_down(egui::Align::LEFT),
                    |ui| {
                        ui.group(|ui| {
                            ui.set_min_height(helper_height - 10.0); // Ensure adequate height
                            self.draw_signal_panel(ui, is_mobile);
                        });
                    }
                );
            });
        });
    }
}
