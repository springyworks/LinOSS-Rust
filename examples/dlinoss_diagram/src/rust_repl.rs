//! Rust REPL Integration for D-LinOSS Diagram Tool
//! 
//! Provides an interactive Rust REPL window for experimenting with code

use egui::{Color32, ScrollArea, TextEdit, Ui};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

pub struct RustRepl {
    input_buffer: String,
    output_history: Arc<Mutex<VecDeque<ReplEntry>>>,
    max_history: usize,
    is_executing: bool,
    show_help: bool,
}

#[derive(Debug, Clone)]
pub struct ReplEntry {
    pub input: String,
    pub output: String,
    pub is_error: bool,
    pub timestamp: String,
}

impl RustRepl {
    pub fn new() -> Self {
        let mut repl = Self {
            input_buffer: String::new(),
            output_history: Arc::new(Mutex::new(VecDeque::new())),
            max_history: 100,
            is_executing: false,
            show_help: false,
        };
        
        // Add welcome message
        repl.add_entry(
            "// Welcome to D-LinOSS Rust REPL".to_string(),
            "🦀 Ready for Rust experimentation!\nType 'help' for commands.".to_string(),
            false,
        );
        
        repl
    }
    
    pub fn render(&mut self, ui: &mut Ui) {
        ui.vertical(|ui| {
            // Header
            ui.horizontal(|ui| {
                ui.heading("🦀 Rust REPL");
                ui.separator();
                
                if ui.button("📋 Clear").clicked() {
                    self.clear_history();
                }
                
                if ui.button("❓ Help").clicked() {
                    self.show_help = !self.show_help;
                }
                
                if ui.button("🔄 Reset").clicked() {
                    self.reset_repl();
                }
            });
            
            ui.separator();
            
            // Help panel
            if self.show_help {
                self.render_help(ui);
                ui.separator();
            }
            
            // Output history
            self.render_history(ui);
            
            ui.separator();
            
            // Input area
            self.render_input(ui);
        });
    }
    
    fn render_help(&self, ui: &mut Ui) {
        ui.collapsing("📖 REPL Help", |ui| {
            ui.label("Available commands:");
            ui.monospace("help           - Show this help");
            ui.monospace("clear          - Clear output history");
            ui.monospace("reset          - Reset REPL state");
            ui.monospace(":vars          - Show variables");
            ui.monospace(":deps          - Show dependencies");
            ui.add_space(5.0);
            ui.label("Example expressions:");
            ui.monospace("2 + 2");
            ui.monospace("let x = vec![1, 2, 3];");
            ui.monospace("x.iter().sum::<i32>()");
            ui.monospace("use std::collections::HashMap;");
        });
    }
    
    fn render_history(&self, ui: &mut Ui) {
        ScrollArea::vertical()
            .max_height(300.0)
            .auto_shrink([false; 2])
            .stick_to_bottom(true)
            .show(ui, |ui| {
                if let Ok(history) = self.output_history.lock() {
                    for entry in history.iter() {
                        self.render_entry(ui, entry);
                        ui.separator();
                    }
                }
            });
    }
    
    fn render_entry(&self, ui: &mut Ui, entry: &ReplEntry) {
        ui.horizontal(|ui| {
            ui.label(">");
            ui.monospace(&entry.input);
        });
        
        if !entry.output.is_empty() {
            ui.horizontal(|ui| {
                ui.add_space(10.0);
                
                let color = if entry.is_error {
                    Color32::LIGHT_RED
                } else {
                    Color32::LIGHT_GREEN
                };
                
                ui.colored_label(color, &entry.output);
            });
        }
        
        ui.small(&entry.timestamp);
    }
    
    fn render_input(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.label(">");
            
            let response = ui.add(
                TextEdit::singleline(&mut self.input_buffer)
                    .desired_width(ui.available_width() - 80.0)
                    .hint_text("Enter Rust expression...")
            );
            
            if ui.button("Execute").clicked() || 
               (response.has_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter))) {
                self.execute_input();
            }
            
            if self.is_executing {
                ui.spinner();
            }
        });
    }
    
    fn execute_input(&mut self) {
        if self.input_buffer.trim().is_empty() || self.is_executing {
            return;
        }
        
        let input = self.input_buffer.trim().to_string();
        self.input_buffer.clear();
        
        // Handle special commands
        match input.as_str() {
            "help" => {
                self.add_entry(
                    input,
                    "Use the Help button above for detailed information.".to_string(),
                    false,
                );
                return;
            }
            "clear" => {
                self.clear_history();
                return;
            }
            "reset" => {
                self.reset_repl();
                return;
            }
            ":vars" => {
                self.add_entry(
                    input,
                    "Variable inspection not yet implemented.".to_string(),
                    false,
                );
                return;
            }
            ":deps" => {
                self.add_entry(
                    input,
                    "Dependencies: std, egui, eframe, linoss_rust".to_string(),
                    false,
                );
                return;
            }
            _ => {}
        }
        
        // Execute Rust expression
        self.execute_rust_expression(input);
    }
    
    fn execute_rust_expression(&mut self, input: String) {
        // For now, implement a simple mock evaluator
        // In a real implementation, you'd use evcxr or similar
        let output = self.mock_evaluate(&input);
        let is_error = output.starts_with("Error:");
        
        self.add_entry(input, output, is_error);
    }
    
    fn mock_evaluate(&self, input: &str) -> String {
        // Simple mock evaluator for demonstration
        match input {
            expr if expr.starts_with("2 + 2") => "4".to_string(),
            expr if expr.contains("vec!") => "Vec created successfully".to_string(),
            expr if expr.contains("println!") => "Output printed".to_string(),
            expr if expr.starts_with("let ") => "Variable defined".to_string(),
            expr if expr.starts_with("use ") => "Module imported".to_string(),
            expr if expr.contains(".sum()") => "Sum calculated".to_string(),
            expr if expr.contains("std::") => "Standard library function called".to_string(),
            _ => format!("Mock evaluation of: {}", input),
        }
    }
    
    fn add_entry(&self, input: String, output: String, is_error: bool) {
        let entry = ReplEntry {
            input,
            output,
            is_error,
            timestamp: chrono::Local::now().format("%H:%M:%S").to_string(),
        };
        
        if let Ok(mut history) = self.output_history.lock() {
            history.push_back(entry);
            
            // Maintain max history size
            while history.len() > self.max_history {
                history.pop_front();
            }
        }
    }
    
    fn clear_history(&self) {
        if let Ok(mut history) = self.output_history.lock() {
            history.clear();
        }
        
        // Re-add welcome message
        self.add_entry(
            "// REPL Cleared".to_string(),
            "🧹 History cleared. Ready for new experiments!".to_string(),
            false,
        );
    }
    
    fn reset_repl(&mut self) {
        self.input_buffer.clear();
        self.is_executing = false;
        self.clear_history();
        
        self.add_entry(
            "// REPL Reset".to_string(),
            "🔄 REPL reset. All state cleared.".to_string(),
            false,
        );
    }
}

impl Default for RustRepl {
    fn default() -> Self {
        Self::new()
    }
}
