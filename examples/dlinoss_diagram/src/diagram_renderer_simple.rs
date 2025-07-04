//! Simplified D-LinOSS Diagram Renderer
//! 
//! Creates academic-style block diagrams for D-LinOSS architectures

use egui::{Color32, Vec2, Ui};
use crate::svg_export::SvgExporter;
use mgraphrust::{self, GenericGraph, NodeId, LayoutOptions, EdgeStyle};
use std::collections::HashMap;

// mod pathfinding; // This module is now obsolete and replaced by mGraphRust

/// Main application structure for the D-LinOSS diagram generator.
/// 
/// This struct contains all the state and configuration needed to render
/// interactive diagrams with components, connections, and various visualization options.
pub struct DLinossDiagramApp {
    current_diagram: DiagramType,
    show_math_details: bool,
    svg_exporter: SvgExporter,
    show_grid: bool,
    grid_size: f32,
    graph: GenericGraph<DraggableComponent, Connection>,
    dragging: Option<NodeId>,
    drag_offset: Vec2,
    zoom_factor: f32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiagramType {
    DLinossLayer,
    DLinossBlock,
    CompletePipeline,
    Custom,
}

#[derive(Clone, serde::Deserialize, serde::Serialize)]
pub struct DraggableComponent {
    pub id: String,
    pub position: egui::Pos2,
    pub size: Vec2,
    pub symbol: String,
    pub label: String,
    pub color: Color32,
}

#[derive(Debug, Clone)]
pub struct Connection {
    pub from_id: String, // Keep string IDs for semantic connection definitions
    pub to_id: String,
    pub color: Color32,
}

impl DLinossDiagramApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let mut app = Self {
            current_diagram: DiagramType::DLinossLayer,
            show_math_details: true,
            svg_exporter: SvgExporter::new(),
            show_grid: true,
            grid_size: 20.0,
            graph: GenericGraph::new(),
            dragging: None,
            drag_offset: Vec2::ZERO,
            zoom_factor: 1.0,
        };

        // Initialize default LinOSS layer components
        app.setup_default_linoss_layer();
        app
    }

    fn setup_default_linoss_layer(&mut self) {
        self.graph = GenericGraph::new();
        let mut id_to_nodeid = HashMap::new();

        // Define component positions (will be relative to center)
        let center_x = 600.0; // Default center position
        let center_y = 400.0;
        let spacing = 120.0;

        // Colors for different components
        let input_color = Color32::from_rgb(100, 200, 255); // Light blue
        let matrix_color = Color32::from_rgb(255, 200, 100); // Orange
        let core_color = Color32::from_rgb(200, 255, 100);   // Light green
        let output_color = Color32::from_rgb(255, 150, 150); // Light red
        let feedback_color = Color32::from_rgb(200, 200, 200); // Gray

        // Create components and add them to the graph
        let components = vec![
            DraggableComponent {
                id: "input".to_string(),
                position: egui::pos2(2.0f32.mul_add(-spacing, center_x), center_y),
                size: Vec2::new(80.0, 50.0),
                symbol: "u(t)".to_string(),
                label: "Input".to_string(),
                color: input_color,
            },
            DraggableComponent {
                id: "b_matrix".to_string(),
                position: egui::pos2(center_x - spacing, center_y),
                size: Vec2::new(80.0, 50.0),
                symbol: "B".to_string(),
                label: "Input Matrix".to_string(),
                color: matrix_color,
            },
            DraggableComponent {
                id: "oscillator".to_string(),
                position: egui::pos2(center_x, center_y),
                size: Vec2::new(100.0, 80.0),
                symbol: "∫".to_string(),
                label: "Oscillator\nCore".to_string(),
                color: core_color,
            },
            DraggableComponent {
                id: "c_matrix".to_string(),
                position: egui::pos2(center_x + spacing, center_y),
                size: Vec2::new(80.0, 50.0),
                symbol: "C".to_string(),
                label: "Output Matrix".to_string(),
                color: matrix_color,
            },
            DraggableComponent {
                id: "output".to_string(),
                position: egui::pos2(2.0f32.mul_add(spacing, center_x), center_y),
                size: Vec2::new(80.0, 50.0),
                symbol: "y(t)".to_string(),
                label: "Output".to_string(),
                color: output_color,
            },
            DraggableComponent {
                id: "feedback".to_string(),
                position: egui::pos2(center_x, center_y + 100.0),
                size: Vec2::new(80.0, 50.0),
                symbol: "A".to_string(),
                label: "Feedback".to_string(),
                color: feedback_color,
            },
        ];

        for comp in components {
            let string_id = comp.id.clone();
            let node_id = self.graph.add_node(comp);
            id_to_nodeid.insert(string_id, node_id);
        }

        // Create connections
        let connections = vec![
            Connection { from_id: "input".to_string(), to_id: "b_matrix".to_string(), color: Color32::WHITE },
            Connection { from_id: "b_matrix".to_string(), to_id: "oscillator".to_string(), color: Color32::WHITE },
            Connection { from_id: "oscillator".to_string(), to_id: "c_matrix".to_string(), color: Color32::WHITE },
            Connection { from_id: "c_matrix".to_string(), to_id: "output".to_string(), color: Color32::WHITE },
            Connection { from_id: "oscillator".to_string(), to_id: "feedback".to_string(), color: feedback_color },
            Connection { from_id: "feedback".to_string(), to_id: "oscillator".to_string(), color: feedback_color },
        ];

        for conn in connections {
            if let (Some(&from_node_id), Some(&to_node_id)) = (
                id_to_nodeid.get(&conn.from_id),
                id_to_nodeid.get(&conn.to_id),
            ) {
                self.graph.add_edge(from_node_id, to_node_id, conn);
            }
        }
    }
}

impl eframe::App for DLinossDiagramApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Main diagram panel
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("📊 D-LinOSS Academic Diagram Generator");
            ui.separator();
            
            // Control panel
            self.render_control_panel(ui);
            ui.separator();
            
            // Main diagram area with grid
            self.render_diagram_with_grid(ui);
        });
    }
}

impl DLinossDiagramApp {
    fn render_control_panel(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.label("Diagram Type:");
            egui::ComboBox::from_label("")
                .selected_text(format!("{:?}", self.current_diagram))
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.current_diagram, DiagramType::DLinossLayer, "D-LinOSS Layer");
                    ui.selectable_value(&mut self.current_diagram, DiagramType::DLinossBlock, "D-LinOSS Block");
                    ui.selectable_value(&mut self.current_diagram, DiagramType::CompletePipeline, "Complete Pipeline");
                    ui.selectable_value(&mut self.current_diagram, DiagramType::Custom, "Custom");
                });
            
            ui.separator();
            ui.checkbox(&mut self.show_math_details, "Show Math Details");
            
            ui.separator();
            ui.checkbox(&mut self.show_grid, "Show Grid");
            
            ui.separator();
            ui.label("Grid Size:");
            ui.add(egui::Slider::new(&mut self.grid_size, 10.0..=50.0).text("px"));
            
            ui.separator();
            if ui.button("📄 Export SVG").clicked() {
                if let Err(e) = self.svg_exporter.export_diagram_simple("dlinoss_diagram.svg") {
                    eprintln!("Failed to export SVG: {e}");
                }
            }
        });
    }
    
    fn render_diagram_with_grid(&mut self, ui: &mut Ui) {
        // Create a frame for the diagram area
        egui::Frame::canvas(ui.style()).show(ui, |ui| {
            let (response, painter) = ui.allocate_painter(
                Vec2::new(ui.available_width(), ui.available_height().max(600.0)),
                egui::Sense::click_and_drag(),
            );
            
            let rect = response.rect;
            
            // Handle scroll wheel zoom
            if response.hovered() {
                ui.input(|i| {
                    let scroll_delta = i.smooth_scroll_delta.y;
                    if scroll_delta != 0.0 {
                        let zoom_delta = scroll_delta.mul_add(0.001, 1.0);
                        self.zoom_factor *= zoom_delta;
                        self.zoom_factor = self.zoom_factor.clamp(0.2, 3.0);
                    }
                });
            }
            
            // Simplified zoom - just use it for drawing scale
            let zoom = self.zoom_factor;
            
            // Draw grid background if enabled
            if self.show_grid {
                self.draw_grid(&painter, rect);
            }
            
            // Handle dragging logic
            if response.drag_started() {
                if let Some(hover_pos) = response.hover_pos() {
                    // Check if we're clicking on a component
                    for node_id in self.graph.nodes() {
                        if let Some(component) = self.graph.get_node(node_id) {
                            let comp_rect = egui::Rect::from_center_size(component.position, component.size * zoom);
                            if comp_rect.contains(hover_pos) {
                                self.dragging = Some(node_id);
                                self.drag_offset = component.position - hover_pos / zoom;
                                break;
                            }
                        }
                    }
                }
            }

            if response.dragged() {
                if let (Some(dragging_id), Some(hover_pos)) = (self.dragging, response.hover_pos()) {
                    if let Some(component) = self.graph.get_node_mut(dragging_id) {
                        let mut new_pos = hover_pos / zoom + self.drag_offset;

                        // Snap to grid if enabled
                        if self.show_grid {
                            let grid_size = self.grid_size;
                            new_pos.x = (new_pos.x / grid_size).round() * grid_size;
                            new_pos.y = (new_pos.y / grid_size).round() * grid_size;
                        }
                        
                        // Keep component within reasonable bounds
                        new_pos.x = new_pos.x.clamp(100.0, 1400.0);
                        new_pos.y = new_pos.y.clamp(100.0, 1000.0);
                        
                        component.position = new_pos;
                    }
                }
            }

            if response.drag_stopped() {
                self.dragging = None;
            }
            
            // Render all connections with improved routing
            self.render_connections_improved(&painter, zoom);
            
            // Render all components with zoom
            self.render_components_zoomed(&painter, rect, zoom);
            
            // Add title (not zoomed)
            painter.text(
                egui::pos2(rect.center().x, rect.top() + 30.0),
                egui::Align2::CENTER_CENTER,
                "Interactive LinOSS Layer Architecture",
                egui::FontId::proportional(24.0),
                Color32::WHITE,
            );
            
            // Add zoom info
            painter.text(
                egui::pos2(rect.right() - 100.0, rect.top() + 10.0),
                egui::Align2::LEFT_TOP,
                format!("Zoom: {zoom:.1}x"),
                egui::FontId::proportional(12.0),
                Color32::LIGHT_GRAY,
            );
            
            // Add mathematical equations if enabled (not zoomed)
            if self.show_math_details {
                let eq_start_y = rect.bottom() - 100.0;
                painter.text(
                    egui::pos2(rect.left() + 50.0, eq_start_y),
                    egui::Align2::LEFT_CENTER,
                    "x[t+1] = A·x[t] + B·u[t]",
                    egui::FontId::monospace(14.0),
                    Color32::LIGHT_GRAY,
                );
                painter.text(
                    egui::pos2(rect.left() + 50.0, eq_start_y + 20.0),
                    egui::Align2::LEFT_CENTER,
                    "y[t] = C·x[t] + D·u[t]",
                    egui::FontId::monospace(14.0),
                    Color32::LIGHT_GRAY,
                );
            }
        });
    }
    
    fn render_components_zoomed(&self, painter: &egui::Painter, _rect: egui::Rect, zoom: f32) {
        for node_id in self.graph.nodes() {
            if let Some(component) = self.graph.get_node(node_id) {
                let color = if self.dragging == Some(node_id) {
                    Color32::from_rgb(0, 200, 0)
                } else {
                    component.color
                };
                
                Self::draw_draggable_component_zoomed(
                    painter,
                    component.position * zoom,
                    component.size * zoom,
                    &component.symbol,
                    &component.label,
                    color,
                );
            }
        }
    }

    fn draw_draggable_component_zoomed(painter: &egui::Painter, center: egui::Pos2, size: Vec2, symbol: &str, label: &str, color: Color32) {
        let rect = egui::Rect::from_center_size(center, size);
        
        // Draw background rectangle with rounded corners
        painter.rect_filled(rect, 5.0 * (size.x / 150.0), color.linear_multiply(0.3));
        // Draw rectangle outline using individual lines
        let stroke_width = 2.0 * (size.x / 150.0);
        let stroke = egui::Stroke::new(stroke_width, color);
        painter.line_segment([rect.left_top(), rect.right_top()], stroke);
        painter.line_segment([rect.right_top(), rect.right_bottom()], stroke);
        painter.line_segment([rect.right_bottom(), rect.left_bottom()], stroke);
        painter.line_segment([rect.left_bottom(), rect.left_top()], stroke);
        
        // Draw symbol with scaled font
        let symbol_font_size = 18.0 * (size.x / 150.0);
        painter.text(
            center - Vec2::new(0.0, 5.0 * (size.y / 100.0)),
            egui::Align2::CENTER_CENTER,
            symbol,
            egui::FontId::proportional(symbol_font_size),
            Color32::WHITE,
        );
        
        // Draw label with scaled font
        let label_font_size = 10.0 * (size.x / 150.0);
        painter.text(
            center + Vec2::new(0.0, 12.0 * (size.y / 100.0)),
            egui::Align2::CENTER_CENTER,
            label,
            egui::FontId::proportional(label_font_size),
            Color32::LIGHT_GRAY,
        );
    }

    fn draw_grid(&self, painter: &egui::Painter, rect: egui::Rect) {
        let grid_color = Color32::from_rgba_unmultiplied(80, 80, 80, 100);
        let grid_size = self.grid_size;
        
        // Draw vertical lines
        #[allow(clippy::cast_possible_truncation)]
        let x_start = (rect.left() / grid_size).floor() as i32;
        #[allow(clippy::cast_possible_truncation)]
        let x_end = (rect.right() / grid_size).ceil() as i32;
        for i in x_start..=x_end {
            #[allow(clippy::cast_precision_loss)]
            let x = i as f32 * grid_size;
            if x >= rect.left() && x <= rect.right() {
                painter.line_segment(
                    [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
                    egui::Stroke::new(0.5, grid_color),
                );
            }
        }
        
        // Draw horizontal lines  
        #[allow(clippy::cast_possible_truncation)]
        let y_start = (rect.top() / grid_size).floor() as i32;
        #[allow(clippy::cast_possible_truncation)]
        let y_end = (rect.bottom() / grid_size).ceil() as i32;
        for i in y_start..=y_end {
            #[allow(clippy::cast_precision_loss)]
            let y = i as f32 * grid_size;
            if y >= rect.top() && y <= rect.bottom() {
                painter.line_segment(
                    [egui::pos2(rect.left(), y), egui::pos2(rect.right(), y)],
                    egui::Stroke::new(0.5, grid_color),
                );
            }
        }
    }
    
    // New improved methods for zoom and better routing
    // Simplified connection rendering with guaranteed right angles
    fn render_connections_improved(&self, painter: &egui::Painter, _zoom: f32) {
        if self.graph.node_count() < 2 {
            return;
        }

        let layout_options = LayoutOptions {
            edge_style: EdgeStyle::RightAngle,
            clearance: 15.0,
            grid_size: 20.0,
            max_iterations: 1000,
        };

    let paths = mgraphrust::layout::calculate_edge_paths(&self.graph, |node_id| {
            let component = self.graph.get_node(node_id).unwrap();
            egui::Rect::from_center_size(component.position, component.size)
        }, layout_options);

        for (edge_index, path) in paths.iter().enumerate() {
            if path.len() > 1 {
                let mut a_path = Vec::new();
                for point in path {
                    a_path.push(*point);
                }
                
                // Get connection color if available, otherwise use default
                let color = if let Some((_from, _to, conn_data)) = self.graph.get_edge(edge_index) {
                    conn_data.color
                } else {
                    Color32::GRAY
                };
                
                let shape = egui::Shape::line(a_path, egui::Stroke::new(2.0, color));
                painter.add(shape);
            }
        }
    }
}
