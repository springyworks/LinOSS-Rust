//! Terminal Panel for D-LinOSS Diagram Tool
//! 
//! Provides a terminal-like interface for running commands and viewing output

use egui::{Color32, ScrollArea, TextEdit, Ui};
use std::collections::VecDeque;
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::thread;

pub struct Terminal {
    input_buffer: String,
    output_history: Arc<Mutex<VecDeque<TerminalEntry>>>,
    max_history: usize,
    current_directory: String,
    is_executing: bool,
    show_help: bool,
    selected_text: String,
    show_context_menu: bool,
    context_menu_pos: egui::Pos2,
}

#[derive(Debug, Clone)]
pub struct TerminalEntry {
    pub command: String,
    pub output: String,
    pub is_error: bool,
    pub timestamp: String,
    pub working_dir: String,
}

impl Terminal {
    pub fn new() -> Self {
        let current_dir = std::env::current_dir()
            .unwrap_or_else(|_| std::path::PathBuf::from("/"))
            .to_string_lossy()
            .to_string();
        
        let terminal = Self {
            input_buffer: String::new(),
            output_history: Arc::new(Mutex::new(VecDeque::new())),
            max_history: 100,
            current_directory: current_dir.clone(),
            is_executing: false,
            show_help: false,
            selected_text: String::new(),
            show_context_menu: false,
            context_menu_pos: egui::Pos2::ZERO,
        };
        
        // Add welcome message
        terminal.add_entry(
            "".to_string(),
            format!("🖥️  Terminal ready\nWorking directory: {}", current_dir),
            false,
            current_dir,
        );
        
        terminal
    }
    
    pub fn render(&mut self, ui: &mut Ui) {
        ui.vertical(|ui| {
            // Header
            ui.horizontal(|ui| {
                ui.heading("🖥️ Terminal");
                ui.separator();
                
                ui.label(format!("📁 {}", self.current_directory));
                ui.separator();
                
                if ui.button("📋 Clear").clicked() {
                    self.clear_history();
                }
                
                if ui.button("❓ Help").clicked() {
                    self.show_help = !self.show_help;
                }
                
                if ui.button("🏠 Home").clicked() {
                    self.change_directory("~");
                }
            });
            
            ui.separator();
            
            // Help panel
            if self.show_help {
                self.render_help(ui);
                ui.separator();
            }
            
            // Terminal display with old-school look
            self.render_terminal_display(ui);
        });
    }
    
    fn render_terminal_display(&mut self, ui: &mut Ui) {
        // Create a terminal-like background
        let bg_color = Color32::from_rgb(20, 20, 25);
        let text_color = Color32::from_rgb(200, 200, 200);
        let prompt_color = Color32::from_rgb(100, 255, 100);
        let error_color = Color32::from_rgb(255, 100, 100);
        
        ui.allocate_ui_with_layout(
            egui::Vec2::new(ui.available_width(), ui.available_height() - 40.0),
            egui::Layout::top_down(egui::Align::Min),
            |ui| {
                // Set terminal background
                let rect = ui.available_rect_before_wrap();
                ui.painter().rect_filled(rect, 2.0, bg_color);
                
                ScrollArea::vertical()
                    .auto_shrink([false; 2])
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        ui.style_mut().override_text_style = Some(egui::TextStyle::Monospace);
                        
                        // Collect history entries to avoid borrowing issues
                        let history_entries = if let Ok(history) = self.output_history.lock() {
                            history.iter().cloned().collect::<Vec<_>>()
                        } else {
                            Vec::new()
                        };
                        
                        // Render history
                        for entry in history_entries.iter() {
                            // Command line with prompt
                            if !entry.command.is_empty() {
                                ui.horizontal(|ui| {
                                    ui.colored_label(prompt_color, "$");
                                    self.render_selectable_text(ui, &entry.command, text_color);
                                });
                            }
                            
                            // Output
                            if !entry.output.is_empty() {
                                let output_color = if entry.is_error { error_color } else { text_color };
                                
                                // Handle multi-line output
                                for line in entry.output.lines() {
                                    self.render_selectable_text(ui, line, output_color);
                                }
                            }
                            
                            ui.add_space(2.0);
                        }
                        
                        // Current input line
                        ui.horizontal(|ui| {
                            ui.colored_label(prompt_color, "$");
                            
                            let response = ui.add(
                                egui::TextEdit::singleline(&mut self.input_buffer)
                                    .desired_width(ui.available_width() - 50.0)
                                    .hint_text("Enter command...")
                                    .text_color(text_color)
                            );
                            
                            // Auto-focus the input
                            if !response.has_focus() {
                                response.request_focus();
                            }
                            
                            // Handle enter key
                            if response.has_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                                self.execute_command();
                            }
                            
                            if self.is_executing {
                                ui.colored_label(Color32::YELLOW, "⚡");
                            }
                        });
                    });
            }
        );
        
        // Show context menu if requested
        self.render_context_menu(ui);
    }
    
    fn render_selectable_text(&mut self, ui: &mut Ui, text: &str, color: Color32) {
        // Split text into words for individual selection
        let words: Vec<&str> = text.split_whitespace().collect();
        
        ui.horizontal_wrapped(|ui| {
            for (i, word) in words.iter().enumerate() {
                let label = ui.colored_label(color, *word);
                
                // Check for right-click on this word
                if label.clicked_by(egui::PointerButton::Secondary) {
                    self.selected_text = word.to_string();
                    self.show_context_menu = true;
                    self.context_menu_pos = label.rect.center();
                }
                
                // Add space between words except for the last one
                if i < words.len() - 1 {
                    ui.label(" ");
                }
            }
        });
    }
    
    fn render_context_menu(&mut self, ui: &mut Ui) {
        if self.show_context_menu {
            let mut close_menu = false;
            let selected_text = self.selected_text.clone(); // Clone to avoid borrowing issues
            
            egui::Area::new("context_menu".into())
                .fixed_pos(self.context_menu_pos)
                .order(egui::Order::Foreground)
                .show(ui.ctx(), |ui| {
                    egui::Frame::popup(ui.style())
                        .show(ui, |ui| {
                            ui.set_min_width(150.0);
                            
                            ui.label(format!("Selected: '{}'", selected_text));
                            ui.separator();
                            
                            // Command options
                            if ui.button(format!("📁 cd {}", selected_text)).clicked() {
                                self.execute_command_with_text("cd", &selected_text);
                                close_menu = true;
                            }
                            
                            if ui.button(format!("📋 ls {}", selected_text)).clicked() {
                                self.execute_command_with_text("ls", &selected_text);
                                close_menu = true;
                            }
                            
                            if ui.button(format!("📄 cat {}", selected_text)).clicked() {
                                self.execute_command_with_text("cat", &selected_text);
                                close_menu = true;
                            }
                            
                            if ui.button(format!("🔍 find . -name '{}'", selected_text)).clicked() {
                                self.execute_command_with_text("find . -name", &format!("'{}'", selected_text));
                                close_menu = true;
                            }
                            
                            if ui.button(format!("🗑️ rm {}", selected_text)).clicked() {
                                self.execute_command_with_text("rm", &selected_text);
                                close_menu = true;
                            }
                            
                            if ui.button(format!("📊 file {}", selected_text)).clicked() {
                                self.execute_command_with_text("file", &selected_text);
                                close_menu = true;
                            }
                            
                            if ui.button(format!("💾 cp {} .", selected_text)).clicked() {
                                self.execute_command_with_text("cp", &format!("{} .", selected_text));
                                close_menu = true;
                            }
                            
                            ui.separator();
                            
                            if ui.button("❌ Cancel").clicked() {
                                close_menu = true;
                            }
                        });
                });
            
            // Close menu if clicked elsewhere
            if ui.input(|i| i.pointer.any_click()) && !ui.rect_contains_pointer(ui.available_rect_before_wrap()) {
                close_menu = true;
            }
            
            if close_menu {
                self.show_context_menu = false;
                self.selected_text.clear();
            }
        }
    }
    
    fn execute_command_with_text(&mut self, command: &str, text: &str) {
        let full_command = if text.is_empty() {
            command.to_string()
        } else {
            format!("{} {}", command, text)
        };
        
        // Set the command in input buffer and execute
        self.input_buffer = full_command;
        self.execute_command();
    }
    
    fn render_help(&self, ui: &mut Ui) {
        ui.collapsing("📖 Terminal Help", |ui| {
            ui.label("Built-in commands:");
            ui.monospace("clear          - Clear terminal history");
            ui.monospace("pwd            - Print working directory");
            ui.monospace("cd <dir>       - Change directory");
            ui.monospace("ls / dir       - List directory contents");
            ui.monospace("cat <file>     - Display file contents");
            ui.monospace("help           - Show this help");
            ui.add_space(5.0);
            ui.label("System commands:");
            ui.monospace("cargo build    - Build the project");
            ui.monospace("cargo run      - Run the project");
            ui.monospace("cargo test     - Run tests");
            ui.monospace("git status     - Git status");
            ui.monospace("ps aux         - List processes");
        });
    }
    
    fn execute_command(&mut self) {
        if self.input_buffer.trim().is_empty() || self.is_executing {
            return;
        }
        
        let command = self.input_buffer.trim().to_string();
        self.input_buffer.clear();
        self.is_executing = true;
        
        // Handle built-in commands
        match command.as_str() {
            "clear" => {
                self.clear_history();
                self.is_executing = false;
                return;
            }
            "help" => {
                self.add_entry(
                    command,
                    "Use the Help button above for detailed information.".to_string(),
                    false,
                    self.current_directory.clone(),
                );
                self.is_executing = false;
                return;
            }
            "pwd" => {
                self.add_entry(
                    command,
                    self.current_directory.clone(),
                    false,
                    self.current_directory.clone(),
                );
                self.is_executing = false;
                return;
            }
            _ => {}
        }
        
        // Handle cd command
        if command.starts_with("cd ") {
            let dir = command[3..].trim();
            self.change_directory(dir);
            self.is_executing = false;
            return;
        }
        
        // Execute system command
        self.execute_system_command(command);
    }
    
    fn change_directory(&mut self, dir: &str) {
        let target_dir = if dir == "~" {
            std::env::var("HOME").unwrap_or_else(|_| "/home".to_string())
        } else if dir.starts_with('/') {
            dir.to_string()
        } else {
            format!("{}/{}", self.current_directory, dir)
        };
        
        match std::env::set_current_dir(&target_dir) {
            Ok(_) => {
                self.current_directory = std::env::current_dir()
                    .unwrap_or_else(|_| std::path::PathBuf::from("/"))
                    .to_string_lossy()
                    .to_string();
                self.add_entry(
                    format!("cd {}", dir),
                    format!("Changed to: {}", self.current_directory),
                    false,
                    self.current_directory.clone(),
                );
            }
            Err(e) => {
                self.add_entry(
                    format!("cd {}", dir),
                    format!("Error: {}", e),
                    true,
                    self.current_directory.clone(),
                );
            }
        }
    }
    
    fn execute_system_command(&mut self, command: String) {
        let parts: Vec<String> = command.split_whitespace().map(|s| s.to_string()).collect();
        if parts.is_empty() {
            self.is_executing = false;
            return;
        }
        
        let cmd = parts[0].clone();
        let args = parts[1..].to_vec();
        
        // Execute command in a separate thread to avoid blocking UI
        let history = Arc::clone(&self.output_history);
        let working_dir = self.current_directory.clone();
        let cmd_string = command.clone();
        
        thread::spawn(move || {
            let output = Command::new(&cmd)
                .args(&args)
                .current_dir(&working_dir)
                .output();
            
            let (stdout, _stderr, is_error) = match output {
                Ok(output) => {
                    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                    let is_error = !output.status.success();
                    
                    if is_error && !stderr.is_empty() {
                        (stderr, String::new(), true)
                    } else if !stdout.is_empty() {
                        (stdout, String::new(), false)
                    } else if !stderr.is_empty() {
                        (stderr, String::new(), false)
                    } else {
                        ("Command completed successfully".to_string(), String::new(), false)
                    }
                }
                Err(e) => {
                    (format!("Error executing command: {}", e), String::new(), true)
                }
            };
            
            let entry = TerminalEntry {
                command: cmd_string,
                output: stdout,
                is_error,
                timestamp: chrono::Local::now().format("%H:%M:%S").to_string(),
                working_dir,
            };
            
            if let Ok(mut history) = history.lock() {
                history.push_back(entry);
                
                // Maintain max history size
                while history.len() > 100 {
                    history.pop_front();
                }
            }
        });
        
        self.is_executing = false;
    }
    
    fn add_entry(&self, command: String, output: String, is_error: bool, working_dir: String) {
        let entry = TerminalEntry {
            command,
            output,
            is_error,
            timestamp: chrono::Local::now().format("%H:%M:%S").to_string(),
            working_dir,
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
            "".to_string(),
            format!("🧹 Terminal cleared\nWorking directory: {}", self.current_directory),
            false,
            self.current_directory.clone(),
        );
    }
}

impl Default for Terminal {
    fn default() -> Self {
        Self::new()
    }
}
