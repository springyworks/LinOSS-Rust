//! Publication-Quality D-LinOSS Diagram Renderer
//! 
//! Creates academic-style block diagrams similar to deep learning publications

use egui::{Color32, Pos2, Rect, Stroke, Vec2, Ui, Painter, StrokeKind};
use crate::publication_style::{AcademicStyle, DiagramTheme};
use crate::svg_export::SvgExporter;

pub struct DLinossDiagramApp {
    style: AcademicStyle,
    current_diagram: DiagramType,
    show_math_details: bool,
    export_svg: bool,
    svg_exporter: SvgExporter,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DiagramType {
    DLinossLayer,
    DLinossBlock,
    CompletePipeline,
    Custom,
}

impl DLinossDiagramApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        Self {
            style: AcademicStyle::new(DiagramTheme::Academic),
            current_diagram: DiagramType::DLinossLayer,
            show_math_details: true,
            export_svg: false,
            svg_exporter: SvgExporter::new(),
        }
    }
}

impl eframe::App for DLinossDiagramApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("📊 D-LinOSS Academic Diagram Generator");
            ui.separator();
            
            // Control panel
            self.render_control_panel(ui);
            ui.separator();
            
            // Main diagram area
            self.render_diagram(ui);
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
            if ui.button("📄 Export SVG").clicked() {
                if let Err(e) = self.svg_exporter.export_diagram_simple("dlinoss_diagram.svg") {
                    eprintln!("Failed to export SVG: {}", e);
                }
            }
        });
    }
    
    fn render_diagram(&mut self, ui: &mut Ui) {
        let available_rect = ui.available_rect_before_wrap();
        let canvas_rect = Rect::from_min_size(available_rect.min, Vec2::new(800.0, 400.0));
        
        let response = ui.allocate_rect(canvas_rect, egui::Sense::hover());
        let painter = ui.painter_at(canvas_rect);
        
        // Draw white background
        painter.rect_filled(canvas_rect, 0.0, Color32::WHITE);
        painter.rect_stroke(canvas_rect, 0.0, Stroke::new(1.0, Color32::BLACK), egui::epaint::StrokeKind::Solid);
        
        match self.current_diagram {
            DiagramType::DLinossLayer => self.render_dlinoss_layer(&painter, canvas_rect),
            DiagramType::DLinossBlock => self.render_dlinoss_block(&painter, canvas_rect),
            DiagramType::CompletePipeline => self.render_complete_pipeline(&painter, canvas_rect),
            DiagramType::Custom => self.render_custom_diagram(ui),
        }
    }
    
    fn render_dlinoss_layer(&self, painter: &Painter, canvas_rect: Rect) {
        let y_center = canvas_rect.center().y;
        let x_start = canvas_rect.min.x + 50.0;
        let spacing = 120.0;
        
        // Input u(t)
        let input_rect = Rect::from_center_size(
            Pos2::new(x_start, y_center), 
            Vec2::new(80.0, 40.0)
        );
        painter.rect_filled(input_rect, 5.0, self.style.input_color);
        painter.rect_stroke(input_rect, 5.0, Stroke::new(1.0, Color32::BLACK));
        painter.text(
            input_rect.center(),
            egui::Align2::CENTER_CENTER,
            "u(t)",
            egui::FontId::default(),
            Color32::WHITE,
        );
        
        // B Matrix
        let b_rect = Rect::from_center_size(
            Pos2::new(x_start + spacing, y_center), 
            Vec2::new(60.0, 40.0)
        );
        painter.rect_filled(b_rect, 5.0, self.style.matrix_color);
        painter.rect_stroke(b_rect, 5.0, Stroke::new(1.0, Color32::BLACK));
        painter.text(
            b_rect.center(),
            egui::Align2::CENTER_CENTER,
            "B",
            egui::FontId::proportional(16.0),
            Color32::WHITE,
        );
        
        // Oscillatory Core (∫)
        let core_rect = Rect::from_center_size(
            Pos2::new(x_start + spacing * 2.0, y_center), 
            Vec2::new(100.0, 80.0)
        );
        painter.rect_filled(core_rect, 8.0, self.style.dynamics_color);
        painter.rect_stroke(core_rect, 8.0, Stroke::new(2.0, Color32::BLACK));
        painter.text(
            core_rect.center(),
            egui::Align2::CENTER_CENTER,
            "∫",
            egui::FontId::proportional(24.0),
            Color32::WHITE,
        );
        painter.text(
            core_rect.center() + Vec2::new(0.0, 20.0),
            egui::Align2::CENTER_CENTER,
            "x(t)",
            egui::FontId::proportional(12.0),
            Color32::WHITE,
        );
        
        // A Matrix (feedback)
        let a_rect = Rect::from_center_size(
            Pos2::new(x_start + spacing * 2.0, y_center + 80.0), 
            Vec2::new(60.0, 40.0)
        );
        painter.rect_filled(a_rect, 5.0, self.style.matrix_color);
        painter.rect_stroke(a_rect, 5.0, Stroke::new(1.0, Color32::BLACK));
        painter.text(
            a_rect.center(),
            egui::Align2::CENTER_CENTER,
            "A",
            egui::FontId::proportional(16.0),
            Color32::WHITE,
        );
        
        // C Matrix
        let c_rect = Rect::from_center_size(
            Pos2::new(x_start + spacing * 3.0, y_center), 
            Vec2::new(60.0, 40.0)
        );
        painter.rect_filled(c_rect, 5.0, self.style.matrix_color);
        painter.rect_stroke(c_rect, 5.0, Stroke::new(1.0, Color32::BLACK));
        painter.text(
            c_rect.center(),
            egui::Align2::CENTER_CENTER,
            "C",
            egui::FontId::proportional(16.0),
            Color32::WHITE,
        );
        
        // Output y(t)
        let output_rect = Rect::from_center_size(
            Pos2::new(x_start + spacing * 4.0, y_center), 
            Vec2::new(80.0, 40.0)
        );
        painter.rect_filled(output_rect, 5.0, self.style.output_color);
        painter.rect_stroke(output_rect, 5.0, Stroke::new(1.0, Color32::BLACK));
        painter.text(
            output_rect.center(),
            egui::Align2::CENTER_CENTER,
            "y(t)",
            egui::FontId::default(),
            Color32::WHITE,
        );
        
        // D Matrix (feedthrough path)
        let d_rect = Rect::from_center_size(
            Pos2::new(x_start + spacing * 2.0, y_center - 80.0), 
            Vec2::new(50.0, 30.0)
        );
        painter.rect_filled(d_rect, 5.0, self.style.feedthrough_color);
        painter.rect_stroke(d_rect, 5.0, Stroke::new(1.0, Color32::BLACK));
        painter.text(
            d_rect.center(),
            egui::Align2::CENTER_CENTER,
            "D",
            egui::FontId::proportional(14.0),
            Color32::BLACK,
        );
        
        // Draw arrows
        self.draw_arrow(painter, input_rect.right_center(), b_rect.left_center());
        self.draw_arrow(painter, b_rect.right_center(), core_rect.left_center());
        self.draw_arrow(painter, core_rect.right_center(), c_rect.left_center());
        self.draw_arrow(painter, c_rect.right_center(), output_rect.left_center());
        
        // Feedback arrow
        self.draw_arrow(painter, core_rect.bottom_center(), a_rect.top_center());
        self.draw_arrow(painter, a_rect.left_center(), core_rect.left_center() + Vec2::new(0.0, 30.0));
        
        // Feedthrough path
        let feedthrough_start = Pos2::new(input_rect.center().x, d_rect.center().y);
        let feedthrough_end = Pos2::new(output_rect.center().x, d_rect.center().y);
        painter.line_segment([feedthrough_start, d_rect.left_center()], Stroke::new(2.0, Color32::BLACK));
        painter.line_segment([d_rect.right_center(), feedthrough_end], Stroke::new(2.0, Color32::BLACK));
        painter.line_segment([feedthrough_start, feedthrough_start + Vec2::new(0.0, -30.0)], Stroke::new(2.0, Color32::BLACK));
        painter.line_segment([feedthrough_end, feedthrough_end + Vec2::new(0.0, 30.0)], Stroke::new(2.0, Color32::BLACK));
        
        // Title
        painter.text(
            canvas_rect.center_top() + Vec2::new(0.0, 20.0),
            egui::Align2::CENTER_CENTER,
            "D-LinOSS Layer Architecture",
            egui::FontId::proportional(18.0),
            Color32::BLACK,
        );
        
        // Mathematical equations if enabled
        if self.show_math_details {
            painter.text(
                canvas_rect.left_bottom() + Vec2::new(20.0, -60.0),
                egui::Align2::LEFT_BOTTOM,
                "State equation: x[t+1] = A·x[t] + B·u[t]",
                egui::FontId::monospace(12.0),
                Color32::BLACK,
            );
            painter.text(
                canvas_rect.left_bottom() + Vec2::new(20.0, -40.0),
                egui::Align2::LEFT_BOTTOM,
                "Output equation: y[t] = C·x[t] + D·u[t]",
                egui::FontId::monospace(12.0),
                Color32::BLACK,
            );
        }
    }
    
    fn render_dlinoss_block(&self, painter: &Painter, canvas_rect: Rect) {
        painter.text(
            canvas_rect.center(),
            egui::Align2::CENTER_CENTER,
            "D-LinOSS Block (Coming Soon)",
            egui::FontId::proportional(16.0),
            Color32::BLACK,
        );
    }
    
    fn render_complete_pipeline(&self, painter: &Painter, canvas_rect: Rect) {
        painter.text(
            canvas_rect.center(),
            egui::Align2::CENTER_CENTER,
            "Complete Pipeline (Coming Soon)",
            egui::FontId::proportional(16.0),
            Color32::BLACK,
        );
    }
    
    fn render_custom_diagram(&mut self, ui: &mut Ui) {
        ui.label("🚧 Custom diagram builder coming soon...");
        ui.label("This will allow drag-and-drop creation of custom D-LinOSS architectures.");
    }
    
    fn draw_arrow(&self, painter: &Painter, start: Pos2, end: Pos2) {
        painter.line_segment([start, end], Stroke::new(2.0, Color32::BLACK));
        
        // Simple arrowhead
        let direction = (end - start).normalized();
        let arrow_size = 8.0;
        let arrow_angle = std::f32::consts::PI / 6.0;
        
        let left_arrow = end - direction.rot90() * arrow_size * arrow_angle.sin() - direction * arrow_size * arrow_angle.cos();
        let right_arrow = end + direction.rot90() * arrow_size * arrow_angle.sin() - direction * arrow_size * arrow_angle.cos();
        
        painter.line_segment([end, left_arrow], Stroke::new(2.0, Color32::BLACK));
        painter.line_segment([end, right_arrow], Stroke::new(2.0, Color32::BLACK));
    }
}
