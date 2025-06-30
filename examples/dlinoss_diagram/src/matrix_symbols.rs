//! Mathematical symbols and notation for D-LinOSS diagrams
//! 
//! Provides LaTeX-like mathematical notation rendering for academic diagrams

use egui::{Color32, Pos2, Vec2, Ui};

pub struct MatrixSymbol {
    pub symbol: String,
    pub subscript: Option<String>,
    pub superscript: Option<String>,
    pub style: SymbolStyle,
}

#[derive(Debug, Clone)]
pub enum SymbolStyle {
    Matrix,      // Bold capitals (A, B, C, D)
    Vector,      // Bold lowercase (x, u, y)
    Scalar,      // Italics (t, k, n)
    Function,    // Regular (sin, cos, exp)
}

impl MatrixSymbol {
    pub fn new(symbol: impl Into<String>) -> Self {
        Self {
            symbol: symbol.into(),
            subscript: None,
            superscript: None,
            style: SymbolStyle::Matrix,
        }
    }
    
    pub fn with_subscript(mut self, sub: impl Into<String>) -> Self {
        self.subscript = Some(sub.into());
        self
    }
    
    pub fn with_superscript(mut self, sup: impl Into<String>) -> Self {
        self.superscript = Some(sup.into());
        self
    }
    
    pub fn with_style(mut self, style: SymbolStyle) -> Self {
        self.style = style;
        self
    }
    
    pub fn render(&self, ui: &mut Ui, pos: Pos2) {
        let main_size = match self.style {
            SymbolStyle::Matrix => 16.0,
            SymbolStyle::Vector => 14.0,
            SymbolStyle::Scalar => 14.0,
            SymbolStyle::Function => 12.0,
        };
        
        let main_color = match self.style {
            SymbolStyle::Matrix => Color32::from_rgb(0, 0, 0),
            SymbolStyle::Vector => Color32::from_rgb(0, 0, 139),
            SymbolStyle::Scalar => Color32::from_rgb(139, 0, 0),
            SymbolStyle::Function => Color32::from_rgb(0, 100, 0),
        };
        
        // Main symbol
        let main_text = match self.style {
            SymbolStyle::Matrix => egui::RichText::new(&self.symbol).size(main_size).strong().color(main_color),
            SymbolStyle::Vector => egui::RichText::new(&self.symbol).size(main_size).strong().color(main_color),
            SymbolStyle::Scalar => egui::RichText::new(&self.symbol).size(main_size).italics().color(main_color),
            SymbolStyle::Function => egui::RichText::new(&self.symbol).size(main_size).color(main_color),
        };
        
        ui.allocate_ui_at_rect(
            egui::Rect::from_center_size(pos, Vec2::new(30.0, 20.0)),
            |ui| {
                ui.label(main_text);
                
                // Subscript
                if let Some(ref sub) = self.subscript {
                    ui.allocate_ui_at_rect(
                        egui::Rect::from_center_size(pos + Vec2::new(15.0, 8.0), Vec2::new(20.0, 15.0)),
                        |ui| {
                            ui.label(egui::RichText::new(sub).size(10.0).color(main_color));
                        }
                    );
                }
                
                // Superscript  
                if let Some(ref sup) = self.superscript {
                    ui.allocate_ui_at_rect(
                        egui::Rect::from_center_size(pos + Vec2::new(15.0, -8.0), Vec2::new(20.0, 15.0)),
                        |ui| {
                            ui.label(egui::RichText::new(sup).size(10.0).color(main_color));
                        }
                    );
                }
            }
        );
    }
}

/// Common mathematical symbols used in D-LinOSS diagrams
pub struct MathSymbols;

impl MathSymbols {
    pub fn state_matrix() -> MatrixSymbol {
        MatrixSymbol::new("A").with_style(SymbolStyle::Matrix)
    }
    
    pub fn input_matrix() -> MatrixSymbol {
        MatrixSymbol::new("B").with_style(SymbolStyle::Matrix)
    }
    
    pub fn output_matrix() -> MatrixSymbol {
        MatrixSymbol::new("C").with_style(SymbolStyle::Matrix)
    }
    
    pub fn feedthrough_matrix() -> MatrixSymbol {
        MatrixSymbol::new("D").with_style(SymbolStyle::Matrix)
    }
    
    pub fn state_vector(time: &str) -> MatrixSymbol {
        MatrixSymbol::new("x")
            .with_subscript(time)
            .with_style(SymbolStyle::Vector)
    }
    
    pub fn input_vector(time: &str) -> MatrixSymbol {
        MatrixSymbol::new("u")
            .with_subscript(time)
            .with_style(SymbolStyle::Vector)
    }
    
    pub fn output_vector(time: &str) -> MatrixSymbol {
        MatrixSymbol::new("y")
            .with_subscript(time)
            .with_style(SymbolStyle::Vector)
    }
    
    pub fn damping_function() -> MatrixSymbol {
        MatrixSymbol::new("γ")
            .with_subscript("t")
            .with_style(SymbolStyle::Function)
    }
    
    pub fn oscillator_frequency(index: usize) -> MatrixSymbol {
        MatrixSymbol::new("ω")
            .with_subscript(&index.to_string())
            .with_style(SymbolStyle::Scalar)
    }
    
    pub fn integration_symbol() -> MatrixSymbol {
        MatrixSymbol::new("∫").with_style(SymbolStyle::Function)
    }
    
    pub fn summation_symbol() -> MatrixSymbol {
        MatrixSymbol::new("Σ").with_style(SymbolStyle::Function)
    }
}
