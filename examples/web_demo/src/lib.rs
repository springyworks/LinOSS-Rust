use eframe::egui;
use nalgebra::Vector2;
use std::f64::consts::PI;
use wasm_bindgen::prelude::*;

// Conditional compilation for Burn support
#[cfg(feature = "burn")]
mod lib_burn;
#[cfg(feature = "burn")]
pub use lib_burn::main_burn;

// Conditional compilation for minimal Burn test
#[cfg(feature = "minimal-burn")]
mod lib_minimal_burn;
#[cfg(feature = "minimal-burn")]
pub use lib_minimal_burn::main_minimal_burn;

// Export basic version by default
#[wasm_bindgen]
pub fn main() {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    
    let web_options = eframe::WebOptions::default();
    
    wasm_bindgen_futures::spawn_local(async {
        let document = web_sys::window().unwrap().document().unwrap();
        let canvas = document
            .get_element_by_id("linoss_canvas")
            .unwrap()
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .unwrap();
            
        // Set fixed canvas size to prevent zooming issues
        canvas.set_width(800);
        canvas.set_height(600);
        canvas.style().set_property("width", "800px").unwrap();
        canvas.style().set_property("height", "600px").unwrap();
            
        let start_result = eframe::WebRunner::new()
            .start(
                canvas,
                web_options,
                Box::new(|cc| {
                    Ok(Box::new(LinossWebApp::new(cc)))
                }),
            )
            .await;

        // Remove this if you want the loading screen to show
        if let Err(e) = start_result {
            eframe::web_sys::console::error_1(&format!("Failed to start eframe: {e:?}").into());
        }
    });
}

/// Neural dynamics oscillator demo
#[derive(Default)]
struct LinossWebApp {
    time: f64,
    frequency: f64,
    amplitude: f64,
    damping: f64,
    phase: f64,
    oscillator_state: Vector2<f64>,
    velocity: Vector2<f64>,
    history: Vec<Vector2<f64>>,
    max_history: usize,
    paused: bool,
}

impl LinossWebApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // Configure egui style for web with proper API
        let mut style = (*cc.egui_ctx.style()).clone();
        style.visuals.panel_fill = egui::Color32::from_rgb(40, 40, 40);
        style.spacing.button_padding = egui::vec2(8.0, 4.0);
        style.spacing.item_spacing = egui::vec2(8.0, 6.0);
        cc.egui_ctx.set_style(style);
        
        Self {
            time: 0.0,
            frequency: 1.0,
            amplitude: 1.0,
            damping: 0.1,
            phase: 0.0,
            oscillator_state: Vector2::new(1.0, 0.0),
            velocity: Vector2::new(0.0, 0.0),
            history: Vec::new(),
            max_history: 1000,
            paused: false,
        }
    }
    
    fn update_oscillator(&mut self, dt: f64) {
        if self.paused {
            return;
        }
        
        // Simple damped harmonic oscillator in 2D
        let omega = 2.0 * PI * self.frequency;
        let force_x = -omega * omega * self.oscillator_state.x - 2.0 * self.damping * omega * self.velocity.x;
        let force_y = -omega * omega * self.oscillator_state.y - 2.0 * self.damping * omega * self.velocity.y + 
                      0.1 * self.amplitude * (self.time * 2.0 + self.phase).sin(); // External driving force
        
        self.velocity.x += force_x * dt;
        self.velocity.y += force_y * dt;
        
        self.oscillator_state.x += self.velocity.x * dt;
        self.oscillator_state.y += self.velocity.y * dt;
        
        // Store history for visualization
        self.history.push(self.oscillator_state.clone());
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }
        
        self.time += dt;
    }
}

impl eframe::App for LinossWebApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let dt = ctx.input(|i| i.predicted_dt as f64).min(0.016); // Cap at 60fps
        self.update_oscillator(dt);
        
        // Request continuous repaints for smooth animation
        ctx.request_repaint();
        
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("🧠 D-LinOSS Neural Dynamics Demo");
            
            ui.horizontal(|ui| {
                ui.label("Welcome to the D-LinOSS WASM demo! All controls are here in the interface.");
                if ui.small_button("ℹ About").clicked() {
                    // Could add about dialog
                }
            });
            
            ui.separator();
            
            // Control panel with better styling
            ui.horizontal(|ui| {
                let start_button_text = if self.paused { "▶ Start Simulation" } else { "⏸ Pause Simulation" };
                if ui.button(start_button_text).clicked() {
                    self.paused = !self.paused;
                }
                
                if ui.button("🔄 Reset").clicked() {
                    self.time = 0.0;
                    self.oscillator_state = Vector2::new(1.0, 0.0);
                    self.velocity = Vector2::new(0.0, 0.0);
                    self.history.clear();
                }
                
                ui.separator();
                ui.label(if self.paused { "🔴 Paused" } else { "🟢 Running" });
            });
            
            ui.separator();
            
            // Parameters in a nice grid
            ui.columns(2, |columns| {
                columns[0].vertical(|ui| {
                    ui.label("🎛️ Oscillator Parameters:");
                    ui.horizontal(|ui| {
                        ui.label("Frequency:");
                        ui.add(egui::Slider::new(&mut self.frequency, 0.1..=5.0).suffix(" Hz"));
                    });
                    
                    ui.horizontal(|ui| {
                        ui.label("Amplitude:");
                        ui.add(egui::Slider::new(&mut self.amplitude, 0.0..=3.0).suffix(" units"));
                    });
                });
                
                columns[1].vertical(|ui| {
                    ui.label("⚙️ Physics Parameters:");
                    ui.horizontal(|ui| {
                        ui.label("Damping:");
                        ui.add(egui::Slider::new(&mut self.damping, 0.0..=1.0).suffix(" coeff"));
                    });
                    
                    ui.horizontal(|ui| {
                        ui.label("Phase:");
                        ui.add(egui::Slider::new(&mut self.phase, 0.0..=(2.0 * PI)).suffix(" rad"));
                    });
                });
            });
            
            ui.separator();
            
            // Real-time status in a nice format
            ui.horizontal(|ui| {
                ui.label(format!("⏱️ Time: {:.2}s", self.time));
                ui.separator();
                ui.label(format!("📍 Position: ({:.3}, {:.3})", self.oscillator_state.x, self.oscillator_state.y));
                ui.separator();
                ui.label(format!("💨 Velocity: ({:.3}, {:.3})", self.velocity.x, self.velocity.y));
            });
            
            // Phase space plot
            let plot_size = egui::Vec2::new(400.0, 300.0);
            let (rect, _response) = ui.allocate_exact_size(plot_size, egui::Sense::hover());
            
            if ui.is_rect_visible(rect) {
                let painter = ui.painter_at(rect);
                let center = rect.center();
                let scale = 80.0;
                
                // Draw axes
                painter.line_segment(
                    [egui::Pos2::new(rect.left(), center.y), egui::Pos2::new(rect.right(), center.y)],
                    egui::Stroke::new(1.0, egui::Color32::GRAY),
                );
                painter.line_segment(
                    [egui::Pos2::new(center.x, rect.top()), egui::Pos2::new(center.x, rect.bottom())],
                    egui::Stroke::new(1.0, egui::Color32::GRAY),
                );
                
                // Draw trajectory history
                if self.history.len() > 1 {
                    for i in 1..self.history.len() {
                        let alpha = (i as f32) / (self.history.len() as f32);
                        let color = egui::Color32::from_rgba_premultiplied(
                            (255.0 * alpha) as u8,
                            (100.0 * alpha) as u8,
                            (255.0 * alpha) as u8,
                            (255.0 * alpha * 0.8) as u8,
                        );
                        
                        let p1 = egui::Pos2::new(
                            center.x + (self.history[i-1].x * scale) as f32,
                            center.y - (self.history[i-1].y * scale) as f32,
                        );
                        let p2 = egui::Pos2::new(
                            center.x + (self.history[i].x * scale) as f32,
                            center.y - (self.history[i].y * scale) as f32,
                        );
                        
                        painter.line_segment([p1, p2], egui::Stroke::new(2.0, color));
                    }
                }
                
                // Draw current position
                let current_pos = egui::Pos2::new(
                    center.x + (self.oscillator_state.x * scale) as f32,
                    center.y - (self.oscillator_state.y * scale) as f32,
                );
                painter.circle_filled(current_pos, 6.0, egui::Color32::RED);
                
                // Draw velocity vector
                let vel_end = egui::Pos2::new(
                    current_pos.x + (self.velocity.x * scale * 0.1) as f32,
                    current_pos.y - (self.velocity.y * scale * 0.1) as f32,
                );
                painter.arrow(current_pos, vel_end - current_pos, egui::Stroke::new(2.0, egui::Color32::GREEN));
                
                // Labels
                painter.text(
                    egui::Pos2::new(rect.right() - 30.0, center.y - 10.0),
                    egui::Align2::CENTER_CENTER,
                    "x",
                    egui::FontId::default(),
                    egui::Color32::GRAY,
                );
                painter.text(
                    egui::Pos2::new(center.x + 10.0, rect.top() + 15.0),
                    egui::Align2::CENTER_CENTER,
                    "y",
                    egui::FontId::default(),
                    egui::Color32::GRAY,
                );
            }
            
            ui.separator();
            
            ui.horizontal(|ui| {
                ui.label("🎯 Instructions:");
                ui.label("Click 'Start Simulation' to begin, adjust parameters in real-time, watch the phase space plot!");
            });
            
            ui.horizontal(|ui| {
                ui.label("🔬 This demonstrates basic neural oscillator dynamics that D-LinOSS uses for modeling brain networks.");
                ui.label("The red dot shows current position, green arrow shows velocity, purple trail shows history.");
            });
        });
        
        // Request repaint for smooth animation
        ctx.request_repaint();
    }
}

// Required for eframe
#[wasm_bindgen]
extern "C" {
    fn alert(s: &str);
}

#[wasm_bindgen]
pub fn greet(name: &str) {
    alert(&format!("Hello, {}!", name));
}
