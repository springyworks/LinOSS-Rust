// LinOSS Web Demo Library
// WASM-compatible neural oscillator visualization

pub mod simple_dlinoss;

// Re-export main components
pub use simple_dlinoss::*;

// WASM bindings
#[cfg(target_arch = "wasm32")]
mod wasm {
    use wasm_bindgen::prelude::*;
    
    // Import the visualizer from the binary
    // We'll create a simpler WASM-compatible version
    
    #[wasm_bindgen]
    extern "C" {
        #[wasm_bindgen(js_namespace = console)]
        fn log(s: &str);
    }
    
    macro_rules! console_log {
        ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
    }
    
    #[wasm_bindgen]
    pub fn start_linoss_web_demo() {
        console_error_panic_hook::set_once();
        console_log!("🧠 Starting LinOSS Web Demo...");
        
        let web_options = eframe::WebOptions::default();
        
        wasm_bindgen_futures::spawn_local(async {
            use wasm_bindgen::JsCast;
            use web_sys::HtmlCanvasElement;
            
            let window = web_sys::window().expect("no global `window` exists");
            let document = window.document().expect("should have a document on window");
            let canvas = document
                .get_element_by_id("linoss_canvas")
                .expect("should have a canvas element with id 'linoss_canvas'")
                .dyn_into::<HtmlCanvasElement>()
                .expect("canvas should be an HtmlCanvasElement");
                
            console_log!("🎯 Canvas found, starting eframe...");
                
            eframe::WebRunner::new()
                .start(
                    canvas,
                    web_options,
                    Box::new(|_cc| Ok(Box::new(crate::wasm::LinossWebApp::new()))),
                )
                .await
                .expect("failed to start eframe");
                
            console_log!("✅ LinOSS Web Demo started successfully!");
        });
    }
    
    // Simple WASM-compatible app
    use eframe::{App, Frame};
    use eframe::egui::{self, Context, RichText, Color32};
    use egui_plot::{Line, Plot, PlotPoints};
    use std::collections::VecDeque;
    use crate::simple_dlinoss::{SimpleLinOSS, LinOSSParams};
    use nalgebra::Vector3;
    
    pub struct LinossWebApp {
        oscillator: SimpleLinOSS,
        history: VecDeque<(f32, Vector3<f32>)>,
        time_series_x: VecDeque<[f64; 2]>,
        time_series_y: VecDeque<[f64; 2]>,
        time_series_z: VecDeque<[f64; 2]>,
        time: f64,
        frame_count: u64,
    }
    
    impl LinossWebApp {
        pub fn new() -> Self {
            console_log!("🔧 Creating LinossWebApp...");
            Self {
                oscillator: SimpleLinOSS::new(LinOSSParams::default()),
                history: VecDeque::with_capacity(1000),
                time_series_x: VecDeque::with_capacity(1000),
                time_series_y: VecDeque::with_capacity(1000),
                time_series_z: VecDeque::with_capacity(1000),
                time: 0.0,
                frame_count: 0,
            }
        }
    }
    
    impl App for LinossWebApp {
        fn update(&mut self, ctx: &Context, _frame: &mut Frame) {
            // Update simulation
            for _ in 0..5 {
                self.oscillator.step();
            }
            
            // Update time series data
            self.time += 0.01;
            let state = self.oscillator.state;
            
            self.time_series_x.push_back([self.time, state.x as f64]);
            self.time_series_y.push_back([self.time, state.y as f64]);
            self.time_series_z.push_back([self.time, state.z as f64]);
            
            if self.time_series_x.len() > 1000 {
                self.time_series_x.pop_front();
                self.time_series_y.pop_front();
                self.time_series_z.pop_front();
            }
            
            self.history.push_back((self.time as f32, state));
            if self.history.len() > 1000 {
                self.history.pop_front();
            }
            
            self.frame_count += 1;
            
            // Request continuous repainting
            ctx.request_repaint();
            
            // UI
            egui::TopBottomPanel::top("header").show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new("🧠 LinOSS Neural Oscillator").size(18.0));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(format!("Frame: {}", self.frame_count));
                    });
                });
            });
            
            egui::SidePanel::left("controls").show(ctx, |ui| {
                ui.set_width(250.0);
                
                ui.label("🎛️ Parameters:");
                ui.add(egui::Slider::new(&mut self.oscillator.params.frequency, 1.0..=50.0).text("Frequency"));
                ui.add(egui::Slider::new(&mut self.oscillator.params.damping, 0.0..=1.0).text("Damping"));
                ui.add(egui::Slider::new(&mut self.oscillator.params.coupling, -2.0..=2.0).text("Coupling"));
                ui.add(egui::Slider::new(&mut self.oscillator.params.nonlinearity, 0.0..=2.0).text("Nonlinearity"));
                ui.add(egui::Slider::new(&mut self.oscillator.params.noise_level, 0.0..=0.1).text("Noise"));
                
                if ui.button("🔄 Reset").clicked() {
                    self.oscillator.reset();
                    self.history.clear();
                    self.time_series_x.clear();
                    self.time_series_y.clear();
                    self.time_series_z.clear();
                    self.time = 0.0;
                }
                
                ui.separator();
                ui.label(format!("📍 Position: [{:.3}, {:.3}, {:.3}]", 
                    self.oscillator.state.x, self.oscillator.state.y, self.oscillator.state.z));
            });
            
            egui::CentralPanel::default().show(ctx, |ui| {
                ui.vertical(|ui| {
                    // Time series plot
                    ui.collapsing("📈 Time Series", |ui| {
                        Plot::new("time_series")
                            .height(200.0)
                            .show(ui, |plot_ui| {
                                let x_points = PlotPoints::from_iter(self.time_series_x.iter().copied());
                                let y_points = PlotPoints::from_iter(self.time_series_y.iter().copied());
                                let z_points = PlotPoints::from_iter(self.time_series_z.iter().copied());
                                
                                plot_ui.line(Line::new(x_points).color(Color32::RED).name("X"));
                                plot_ui.line(Line::new(y_points).color(Color32::GREEN).name("Y"));
                                plot_ui.line(Line::new(z_points).color(Color32::BLUE).name("Z"));
                            });
                    });
                    
                    // Phase space plots
                    ui.collapsing("🌌 Phase Space", |ui| {
                        ui.horizontal(|ui| {
                            // XY plot
                            Plot::new("xy_phase")
                                .width(200.0)
                                .height(200.0)
                                .show(ui, |plot_ui| {
                                    let xy_points: PlotPoints = self.history.iter()
                                        .map(|(_, v)| [v.x as f64, v.y as f64])
                                        .collect();
                                    plot_ui.line(Line::new(xy_points).color(Color32::YELLOW).name("XY"));
                                });
                            
                            // XZ plot  
                            Plot::new("xz_phase")
                                .width(200.0)
                                .height(200.0)
                                .show(ui, |plot_ui| {
                                    let xz_points: PlotPoints = self.history.iter()
                                        .map(|(_, v)| [v.x as f64, v.z as f64])
                                        .collect();
                                    plot_ui.line(Line::new(xz_points).color(Color32::from_rgb(0, 255, 255)).name("XZ"));
                                });
                        });
                    });
                });
            });
        }
    }
}

#[cfg(target_arch = "wasm32")]
pub use wasm::*;