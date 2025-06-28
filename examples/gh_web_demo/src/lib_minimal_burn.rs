// Neural Network demo with Burn tensors in WASM
use eframe::egui;
use burn_tensor::{Tensor, Distribution};
use burn_ndarray::{NdArray, NdArrayDevice};
use wasm_bindgen::prelude::*;

// Type alias for our WASM-compatible backend  
type WasmBackend = NdArray<f32>;

#[wasm_bindgen]
pub fn main_minimal_burn() {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    
    let web_options = eframe::WebOptions::default();
    
    wasm_bindgen_futures::spawn_local(async {
        let document = web_sys::window().unwrap().document().unwrap();
        let canvas = document
            .get_element_by_id("linoss_canvas")
            .unwrap()
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .unwrap();
            
        let start_result = eframe::WebRunner::new()
            .start(
                canvas,
                web_options,
                Box::new(|_cc| Ok(Box::new(NeuralNetworkApp::new()))),
            )
            .await;
            
        if let Err(e) = start_result {
            web_sys::console::error_1(&format!("Failed to start eframe: {:?}", e).into());
        }
    });
}

/// Simple Neural Network simulation (forward pass only for now)
pub struct SimpleNeuralNetwork {
    weights1: Tensor<WasmBackend, 2>,
    bias1: Tensor<WasmBackend, 1>,
    weights2: Tensor<WasmBackend, 2>,
    bias2: Tensor<WasmBackend, 1>,
}

impl SimpleNeuralNetwork {
    fn new(device: &NdArrayDevice) -> Self {
        // 2 inputs -> 4 hidden -> 1 output network
        let weights1 = Tensor::random([2, 4], Distribution::Normal(0.0, 0.5), device);
        let bias1 = Tensor::zeros([4], device);
        let weights2 = Tensor::random([4, 1], Distribution::Normal(0.0, 0.5), device);
        let bias2 = Tensor::zeros([1], device);
        
        Self {
            weights1,
            bias1,
            weights2,
            bias2,
        }
    }
    
    fn forward(&self, input: Tensor<WasmBackend, 2>) -> Tensor<WasmBackend, 2> {
        // First layer: input -> hidden
        let hidden = input.matmul(self.weights1.clone()) + self.bias1.clone().unsqueeze_dim(0);
        let hidden_activated = self.relu_activation(hidden);
        
        // Second layer: hidden -> output  
        let output = hidden_activated.matmul(self.weights2.clone()) + self.bias2.clone().unsqueeze_dim(0);
        self.sigmoid_activation(output)
    }
    
    fn relu_activation(&self, tensor: Tensor<WasmBackend, 2>) -> Tensor<WasmBackend, 2> {
        // Manual ReLU: max(0, x)
        tensor.clamp_min(0.0)
    }
    
    fn sigmoid_activation(&self, tensor: Tensor<WasmBackend, 2>) -> Tensor<WasmBackend, 2> {
        // Manual sigmoid: 1 / (1 + exp(-x))
        let ones = Tensor::ones_like(&tensor);
        let neg_tensor = tensor * (-1.0);
        let exp_neg = neg_tensor.exp();
        ones.clone() / (ones + exp_neg)
    }
    
    fn get_weights_data(&self) -> (Vec<f32>, Vec<f32>) {
        let w1_data = self.weights1.to_data().convert::<f32>();
        let w2_data = self.weights2.to_data().convert::<f32>();
        
        (
            w1_data.as_slice::<f32>().unwrap().to_vec(),
            w2_data.as_slice::<f32>().unwrap().to_vec()
        )
    }
    
    fn mutate_weights(&mut self, learning_rate: f32) {
        // Simple random mutation for demonstration
        let noise1 = Tensor::random_like(&self.weights1, Distribution::Normal(0.0, learning_rate as f64));
        let noise2 = Tensor::random_like(&self.weights2, Distribution::Normal(0.0, learning_rate as f64));
        
        self.weights1 = self.weights1.clone() + noise1;
        self.weights2 = self.weights2.clone() + noise2;
    }
}

/// Neural Network demo app
pub struct NeuralNetworkApp {
    device: NdArrayDevice,
    network: SimpleNeuralNetwork,
    training_data: Vec<([f32; 2], f32)>,
    current_predictions: Vec<f32>,
    epoch: usize,
    learning_rate: f32,
    auto_train: bool,
    prediction_input: [f32; 2],
    prediction_output: f32,
    loss_history: Vec<f32>,
}

impl NeuralNetworkApp {
    pub fn new() -> Self {
        let device = NdArrayDevice::Cpu;
        let network = SimpleNeuralNetwork::new(&device);
        
        // XOR-like training data
        let training_data = vec![
            ([0.0, 0.0], 0.0),
            ([0.0, 1.0], 1.0),
            ([1.0, 0.0], 1.0),
            ([1.0, 1.0], 0.0),
        ];
        
        Self {
            device,
            network,
            training_data,
            current_predictions: vec![0.0; 4],
            epoch: 0,
            learning_rate: 0.01,
            auto_train: false,
            prediction_input: [0.5, 0.5],
            prediction_output: 0.0,
            loss_history: Vec::new(),
        }
    }
    
    fn evaluate_network(&mut self) {
        let mut total_loss = 0.0;
        self.current_predictions.clear();
        
        for (input, target) in &self.training_data {
            let input_tensor = Tensor::from_floats([input.clone()], &self.device);
            let prediction = self.network.forward(input_tensor);
            let pred_value = prediction.to_data().convert::<f32>().as_slice::<f32>().unwrap()[0];
            
            self.current_predictions.push(pred_value);
            total_loss += (pred_value - target).powi(2);
        }
        
        let avg_loss = total_loss / self.training_data.len() as f32;
        self.loss_history.push(avg_loss);
        if self.loss_history.len() > 100 {
            self.loss_history.remove(0);
        }
    }
    
    fn train_step(&mut self) {
        // Simple evolutionary approach: mutate and keep if better
        let old_network = SimpleNeuralNetwork {
            weights1: self.network.weights1.clone(),
            bias1: self.network.bias1.clone(),
            weights2: self.network.weights2.clone(),
            bias2: self.network.bias2.clone(),
        };
        
        let old_loss = self.loss_history.last().copied().unwrap_or(f32::INFINITY);
        
        // Mutate the network
        self.network.mutate_weights(self.learning_rate);
        
        // Evaluate new performance
        self.evaluate_network();
        let new_loss = self.loss_history.last().copied().unwrap_or(f32::INFINITY);
        
        // Keep changes only if they improve performance
        if new_loss > old_loss {
            self.network = old_network;
            self.loss_history.pop(); // Remove the worse score
        }
        
        self.epoch += 1;
    }
    
    fn predict(&mut self) {
        let input_tensor = Tensor::from_floats([self.prediction_input], &self.device);
        let output = self.network.forward(input_tensor);
        self.prediction_output = output.to_data().convert::<f32>().as_slice::<f32>().unwrap()[0];
    }
}

impl eframe::App for NeuralNetworkApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("🧠 Neural Network Demo with Burn Tensors");
            ui.separator();
            
            // Training section
            ui.horizontal(|ui| {
                ui.label("🎯 Training XOR function:");
                if ui.button("Train Step").clicked() {
                    self.train_step();
                    self.predict(); // Update prediction
                }
                
                ui.checkbox(&mut self.auto_train, "Auto Train");
                
                if ui.button("Reset Network").clicked() {
                    self.network = SimpleNeuralNetwork::new(&self.device);
                    self.epoch = 0;
                    self.loss_history.clear();
                    self.evaluate_network();
                }
            });
            
            // Auto training
            if self.auto_train && self.epoch < 1000 {
                self.train_step();
                self.predict();
                ctx.request_repaint();
            }
            
            ui.separator();
            
            // Training status
            ui.horizontal(|ui| {
                ui.label(format!("📊 Epoch: {}", self.epoch));
                ui.separator();
                let current_loss = self.loss_history.last().copied().unwrap_or(0.0);
                ui.label(format!("📉 Loss: {:.6}", current_loss));
                ui.separator();
                ui.add(egui::Slider::new(&mut self.learning_rate, 0.001..=0.1)
                      .text("Mutation Rate")
                      .logarithmic(true));
            });
            
            ui.separator();
            
            // Training data visualization
            ui.label("📈 Training Data (XOR function):");
            egui::Grid::new("training_data")
                .num_columns(3)
                .spacing([10.0, 5.0])
                .show(ui, |ui| {
                    ui.label("Input");
                    ui.label("Target");
                    ui.label("Prediction");
                    ui.end_row();
                    
                    for (i, (input, target)) in self.training_data.iter().enumerate() {
                        ui.label(format!("[{:.1}, {:.1}]", input[0], input[1]));
                        ui.label(format!("{:.1}", target));
                        
                        if i < self.current_predictions.len() {
                            let pred = self.current_predictions[i];
                            let error = (pred - target).abs();
                            let color = if error < 0.2 {
                                egui::Color32::GREEN
                            } else if error < 0.5 {
                                egui::Color32::YELLOW
                            } else {
                                egui::Color32::RED
                            };
                            ui.colored_label(color, format!("{:.3}", pred));
                        } else {
                            ui.label("---");
                        }
                        ui.end_row();
                    }
                });
            
            ui.separator();
            
            // Interactive prediction
            ui.label("🎮 Interactive Prediction:");
            ui.horizontal(|ui| {
                ui.label("Input:");
                if ui.add(egui::Slider::new(&mut self.prediction_input[0], 0.0..=1.0).text("X")).changed() {
                    self.predict();
                }
                if ui.add(egui::Slider::new(&mut self.prediction_input[1], 0.0..=1.0).text("Y")).changed() {
                    self.predict();
                }
            });
            
            ui.label(format!("🎯 Output: {:.3}", self.prediction_output));
            
            ui.separator();
            
            // Network weights visualization
            ui.label("⚙️ Network Weights (Layer 1):");
            let (w1_data, _w2_data) = self.network.get_weights_data();
            egui::Grid::new("weights_grid")
                .num_columns(4)
                .spacing([8.0, 4.0])
                .show(ui, |ui| {
                    for (i, &weight) in w1_data.iter().enumerate() {
                        let abs_weight = weight.abs();
                        let color = if weight > 0.0 {
                            egui::Color32::from_rgb(
                                100 + (abs_weight * 155.0).min(155.0) as u8,
                                255,
                                100
                            )
                        } else {
                            egui::Color32::from_rgb(
                                255,
                                100,
                                100 + (abs_weight * 155.0).min(155.0) as u8
                            )
                        };
                        ui.colored_label(color, format!("{:.2}", weight));
                        if (i + 1) % 4 == 0 {
                            ui.end_row();
                        }
                    }
                });
            
            ui.separator();
            
            // Loss history plot (simple)
            if !self.loss_history.is_empty() {
                ui.label("📊 Loss History:");
                let plot_height = 60.0;
                let plot_width = ui.available_width();
                let (rect, _response) = ui.allocate_exact_size(
                    egui::Vec2::new(plot_width, plot_height), 
                    egui::Sense::hover()
                );
                
                if ui.is_rect_visible(rect) {
                    let painter = ui.painter_at(rect);
                    
                    // Draw background
                    painter.rect_filled(rect, 2.0, egui::Color32::from_rgb(40, 40, 40));
                    
                    // Draw loss curve
                    if self.loss_history.len() > 1 {
                        let max_loss = self.loss_history.iter().copied().fold(0.0f32, f32::max);
                        let min_loss = self.loss_history.iter().copied().fold(f32::INFINITY, f32::min);
                        let loss_range = (max_loss - min_loss).max(0.001);
                        
                        for i in 1..self.loss_history.len() {
                            let x1 = rect.left() + (i - 1) as f32 * rect.width() / (self.loss_history.len() - 1) as f32;
                            let x2 = rect.left() + i as f32 * rect.width() / (self.loss_history.len() - 1) as f32;
                            let y1 = rect.bottom() - (self.loss_history[i - 1] - min_loss) / loss_range * rect.height();
                            let y2 = rect.bottom() - (self.loss_history[i] - min_loss) / loss_range * rect.height();
                            
                            painter.line_segment(
                                [egui::Pos2::new(x1, y1), egui::Pos2::new(x2, y2)],
                                egui::Stroke::new(2.0, egui::Color32::LIGHT_BLUE)
                            );
                        }
                    }
                }
            }
            
            ui.separator();
            ui.label("✅ Neural Network with Burn tensors working in WASM!");
            ui.label("🔥 Forward pass, matrix operations, and evolutionary training!");
            
            // Initialize predictions if needed
            if self.current_predictions.is_empty() {
                self.evaluate_network();
                self.predict();
            }
        });
        
        ctx.request_repaint();
    }
}
