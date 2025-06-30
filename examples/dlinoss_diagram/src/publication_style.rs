//! Publication-quality styling for academic diagrams
//! 
//! Provides color schemes and styling options that match deep learning publications

use egui::Color32;

pub struct AcademicStyle {
    pub theme: DiagramTheme,
    pub input_color: Color32,
    pub matrix_color: Color32,
    pub dynamics_color: Color32,
    pub output_color: Color32,
    pub feedthrough_color: Color32,
    pub arrow_color: Color32,
    pub text_color: Color32,
    pub background_color: Color32,
    pub grid_color: Color32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DiagramTheme {
    Academic,      // Traditional academic publication style
    Modern,        // Modern flat design
    Technical,     // Engineering diagram style
    Colorful,      // Vibrant colors for presentations
}

impl AcademicStyle {
    pub fn new(theme: DiagramTheme) -> Self {
        match theme {
            DiagramTheme::Academic => Self::academic_theme(),
            DiagramTheme::Modern => Self::modern_theme(),
            DiagramTheme::Technical => Self::technical_theme(),
            DiagramTheme::Colorful => Self::colorful_theme(),
        }
    }
    
    fn academic_theme() -> Self {
        Self {
            theme: DiagramTheme::Academic,
            // Traditional academic colors (grayscale with blue accents)
            input_color: Color32::from_rgb(70, 130, 180),      // Steel blue
            matrix_color: Color32::from_rgb(105, 105, 105),    // Dim gray
            dynamics_color: Color32::from_rgb(72, 61, 139),    // Dark slate blue
            output_color: Color32::from_rgb(178, 34, 34),      // Fire brick
            feedthrough_color: Color32::from_rgb(255, 140, 0), // Dark orange
            arrow_color: Color32::from_rgb(0, 0, 0),           // Black
            text_color: Color32::from_rgb(0, 0, 0),            // Black
            background_color: Color32::from_rgb(255, 255, 255), // White
            grid_color: Color32::from_rgb(200, 200, 200),      // Light gray
        }
    }
    
    fn modern_theme() -> Self {
        Self {
            theme: DiagramTheme::Modern,
            // Modern flat design colors
            input_color: Color32::from_rgb(52, 152, 219),      // Flat blue
            matrix_color: Color32::from_rgb(149, 165, 166),    // Concrete
            dynamics_color: Color32::from_rgb(155, 89, 182),   // Amethyst
            output_color: Color32::from_rgb(231, 76, 60),      // Alizarin
            feedthrough_color: Color32::from_rgb(230, 126, 34), // Carrot
            arrow_color: Color32::from_rgb(44, 62, 80),        // Midnight blue
            text_color: Color32::from_rgb(44, 62, 80),         // Midnight blue
            background_color: Color32::from_rgb(236, 240, 241), // Clouds
            grid_color: Color32::from_rgb(189, 195, 199),      // Silver
        }
    }
    
    fn technical_theme() -> Self {
        Self {
            theme: DiagramTheme::Technical,
            // Engineering diagram style (high contrast)
            input_color: Color32::from_rgb(0, 100, 0),         // Dark green
            matrix_color: Color32::from_rgb(0, 0, 139),        // Dark blue
            dynamics_color: Color32::from_rgb(139, 0, 139),    // Dark magenta
            output_color: Color32::from_rgb(139, 0, 0),        // Dark red
            feedthrough_color: Color32::from_rgb(255, 165, 0), // Orange
            arrow_color: Color32::from_rgb(0, 0, 0),           // Black
            text_color: Color32::from_rgb(0, 0, 0),            // Black
            background_color: Color32::from_rgb(248, 248, 255), // Ghost white
            grid_color: Color32::from_rgb(169, 169, 169),      // Dark gray
        }
    }
    
    fn colorful_theme() -> Self {
        Self {
            theme: DiagramTheme::Colorful,
            // Vibrant presentation colors
            input_color: Color32::from_rgb(46, 204, 113),      // Emerald
            matrix_color: Color32::from_rgb(52, 152, 219),     // Peter river
            dynamics_color: Color32::from_rgb(155, 89, 182),   // Amethyst
            output_color: Color32::from_rgb(231, 76, 60),      // Alizarin
            feedthrough_color: Color32::from_rgb(241, 196, 15), // Sun flower
            arrow_color: Color32::from_rgb(52, 73, 94),        // Wet asphalt
            text_color: Color32::from_rgb(52, 73, 94),         // Wet asphalt
            background_color: Color32::from_rgb(255, 255, 255), // White
            grid_color: Color32::from_rgb(189, 195, 199),      // Silver
        }
    }
}

/// Typography settings for academic diagrams
pub struct Typography {
    pub title_size: f32,
    pub heading_size: f32,
    pub body_size: f32,
    pub caption_size: f32,
    pub math_size: f32,
}

impl Default for Typography {
    fn default() -> Self {
        Self {
            title_size: 18.0,
            heading_size: 16.0,
            body_size: 14.0,
            caption_size: 12.0,
            math_size: 14.0,
        }
    }
}

/// Layout parameters for consistent spacing
pub struct Layout {
    pub block_spacing: f32,
    pub arrow_length: f32,
    pub text_padding: f32,
    pub line_thickness: f32,
}

impl Default for Layout {
    fn default() -> Self {
        Self {
            block_spacing: 2.0,
            arrow_length: 1.0,
            text_padding: 0.2,
            line_thickness: 2.0,
        }
    }
}
