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
            // Create input signal
            let input_x = self.frequency * self.time.cos();
            let input_y = self.frequency * self.time.sin() * 0.7;
            
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
                
                // Store the phase space coordinates
                self.phase_data.push_back([result[0] as f64, result[1] as f64]);
                
                if self.phase_data.len() > self.max_points {
                    self.phase_data.pop_front();
                }
            }
        }
        
        #[cfg(all(feature = "burn", not(feature = "linoss")))]
        {
            // Simple oscillator simulation using Burn tensors
            let input_data = vec![
                self.frequency * self.time.cos() * (-self.damping * self.time).exp(),
                self.frequency * self.time.sin() * (-self.damping * self.time).exp(),
            ];
            
            let input_tensor = Tensor::<Backend, 1>::from_data(
                TensorData::new(input_data, [2]),
                &self.device,
            );
            
            // Simple transformation
            let scaled = input_tensor * Tensor::from_data([self.coupling, self.coupling * 0.8], &self.device);
            let tensor_data = scaled.to_data();
            let result = tensor_data.as_slice::<f32>().unwrap();
            
            self.phase_data.push_back([result[0] as f64, result[1] as f64]);
            self.signal_data.push_back([self.time as f64, result[0] as f64]);
        }
        
        #[cfg(not(feature = "burn"))]
        {
            // Mock oscillator
            let x = self.frequency * self.time.cos() * (-self.damping * self.time).exp() * self.coupling;
            let y = self.frequency * self.time.sin() * (-self.damping * self.time).exp() * self.coupling * 0.8;
            
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
            // Create a 3D trajectory using phase space + time dimension
            let z = (self.time * 0.5).sin() * 0.5; // Add some Z variation
            self.trajectory_3d.push_back([x as f32, y as f32, z]);
            
            if self.trajectory_3d.len() > self.max_points {
                self.trajectory_3d.pop_front();
            }
        }
    }
    
    fn project_3d_to_2d(&self, point: [f32; 3], center: Pos2, scale: f32) -> Pos2 {
        // Simple 3D to 2D projection with rotation
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
        
        // Simple perspective projection
        let perspective = 1.0 / (1.0 + z2 * 0.5);
        
        Pos2::new(
            center.x + x2 * scale * perspective,
            center.y - y1 * scale * perspective, // Flip Y for screen coordinates
        )
    }
    
    fn draw_3d_axes(&self, ui: &mut egui::Ui, rect: egui::Rect) {
        let painter = ui.painter();
        let center = rect.center();
        let scale = rect.width().min(rect.height()) * 0.3;
        
        // Draw axes
        let axes = [
            ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], Color32::RED),    // X axis
            ([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], Color32::GREEN),  // Y axis
            ([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], Color32::BLUE),   // Z axis
        ];
        
        for (start, end, color) in axes {
            let p1 = self.project_3d_to_2d(start, center, scale);
            let p2 = self.project_3d_to_2d(end, center, scale);
            painter.line_segment([p1, p2], Stroke::new(2.0, color));
        }
        
        // Draw axis labels
        let label_offset = 1.1;
        painter.text(
            self.project_3d_to_2d([label_offset, 0.0, 0.0], center, scale),
            egui::Align2::CENTER_CENTER,
            "X",
            egui::FontId::default(),
            Color32::RED,
        );
        painter.text(
            self.project_3d_to_2d([0.0, label_offset, 0.0], center, scale),
            egui::Align2::CENTER_CENTER,
            "Y",
            egui::FontId::default(),
            Color32::GREEN,
        );
        painter.text(
            self.project_3d_to_2d([0.0, 0.0, label_offset], center, scale),
            egui::Align2::CENTER_CENTER,
            "Z",
            egui::FontId::default(),
            Color32::BLUE,
        );
    }
}

impl eframe::App for DLinossApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Request continuous repainting for smooth animation
        ctx.request_repaint();
        
        // Update simulation
        self.update_simulation();
        
        // Top panel with controls
        egui::TopBottomPanel::top("controls").show(ctx, |ui| {
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
                }
                
                ui.separator();
                
                ui.label("Frequency:");
                ui.add(egui::Slider::new(&mut self.frequency, 0.1..=10.0).suffix(" Hz"));
                
                ui.label("Damping:");
                ui.add(egui::Slider::new(&mut self.damping, 0.0..=2.0));
                
                ui.label("Coupling:");
                ui.add(egui::Slider::new(&mut self.coupling, 0.0..=2.0));
            });
        });
        
        // Main content area with THREE panels
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                let panel_width = ui.available_width() / 3.0;
                
                // Phase space plot (left)
                ui.allocate_ui_with_layout(
                    [panel_width, ui.available_height()].into(),
                    egui::Layout::top_down(egui::Align::Center),
                    |ui| {
                        ui.heading("Phase Space");
                        
                        use egui_plot::{Plot, Points, PlotPoints};
                        
                        Plot::new("phase_plot")
                            .view_aspect(1.0)
                            .show(ui, |plot_ui| {
                                if !self.phase_data.is_empty() {
                                    let points: PlotPoints = self.phase_data.iter().cloned().collect();
                                    plot_ui.points(Points::new(points).radius(2.0).color(egui::Color32::LIGHT_BLUE));
                                    
                                    // Current position as larger point
                                    if let Some(&last_point) = self.phase_data.back() {
                                        let current: PlotPoints = vec![last_point].into();
                                        plot_ui.points(Points::new(current).radius(8.0).color(egui::Color32::RED));
                                    }
                                }
                            });
                    },
                );
                
                ui.separator();
                
                // Signal over time (middle)
                ui.allocate_ui_with_layout(
                    [panel_width, ui.available_height()].into(),
                    egui::Layout::top_down(egui::Align::Center),
                    |ui| {
                        ui.heading("Signal Over Time");
                        
                        use egui_plot::{Plot, Line, PlotPoints};
                        
                        Plot::new("signal_plot")
                            .show(ui, |plot_ui| {
                                if !self.signal_data.is_empty() {
                                    let points: PlotPoints = self.signal_data.iter().cloned().collect();
                                    plot_ui.line(Line::new(points).color(egui::Color32::GREEN));
                                }
                            });
                    },
                );
                
                ui.separator();
                
                // 3D Trajectory (right)
                ui.allocate_ui_with_layout(
                    [panel_width, ui.available_height()].into(),
                    egui::Layout::top_down(egui::Align::Center),
                    |ui| {
                        ui.heading("3D Phase Trajectory");
                        
                        // Interactive rotation
                        let rect = ui.available_rect_before_wrap();
                        let response = ui.allocate_rect(rect, egui::Sense::drag());
                        
                        if response.dragged() {
                            let delta = response.drag_delta();
                            self.rotation_y += delta.x * 0.01;
                            self.rotation_x += delta.y * 0.01;
                        }
                        
                        // Draw 3D visualization
                        self.draw_3d_axes(ui, rect);
                        
                        // Draw trajectory
                        let painter = ui.painter();
                        let center = rect.center();
                        let scale = rect.width().min(rect.height()) * 0.3;
                        
                        // Draw trajectory as connected lines
                        if self.trajectory_3d.len() > 1 {
                            let trajectory_vec: Vec<[f32; 3]> = self.trajectory_3d.iter().cloned().collect();
                            for i in 0..trajectory_vec.len() - 1 {
                                let p1 = self.project_3d_to_2d(trajectory_vec[i], center, scale);
                                let p2 = self.project_3d_to_2d(trajectory_vec[i + 1], center, scale);
                                
                                // Color gradient based on time
                                let t = i as f32 / trajectory_vec.len() as f32;
                                let color = Color32::from_rgb(
                                    (100.0 + 155.0 * t) as u8,
                                    100,
                                    (255.0 - 155.0 * t) as u8,
                                );
                                
                                painter.line_segment([p1, p2], Stroke::new(2.0, color));
                            }
                            
                            // Draw current position
                            if let Some(&last_point) = self.trajectory_3d.back() {
                                let pos = self.project_3d_to_2d(last_point, center, scale);
                                painter.circle_filled(pos, 5.0, Color32::YELLOW);
                            }
                        }
                        
                        // Instructions
                        ui.label("Drag to rotate");
                    },
                );
            });
        });
        
        // Status bar
        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(format!("Time: {:.2}s", self.time));
                ui.separator();
                ui.label(format!("Points: {}", self.phase_data.len()));
                ui.separator();
                
                #[cfg(feature = "linoss")]
                ui.label("💚 Real D-LinOSS Layer");
                
                #[cfg(all(feature = "burn", not(feature = "linoss")))]
                ui.label("🔧 Burn Tensor Demo");
                
                #[cfg(not(feature = "burn"))]
                ui.label("📝 Mock Demo");
            });
        });
    }
}
