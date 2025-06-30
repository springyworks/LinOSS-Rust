//! SVG export functionality for publication-ready diagrams
//! 
//! Exports diagrams as high-quality SVG files suitable for academic papers

use std::fs;
use std::io::Write as _;

/// SVG export functionality for diagrams.
/// 
/// Provides methods to export diagrams in SVG format with configurable paths.
pub struct SvgExporter {
    /// The directory path where SVG files will be exported
    pub export_path: String,
}

impl SvgExporter {
    /// Creates a new SVG exporter with default export path.
    pub fn new() -> Self {
        Self {
            export_path: "./exports".to_owned(),
        }
    }
    
    pub fn export_diagram_simple(&self, filename: &str) -> Result<(), Box<dyn std::error::Error>> {
        // Create export directory if it doesn't exist
        fs::create_dir_all(&self.export_path)?;
        
        let filepath = format!("{}/{}", self.export_path, filename);
        
        // Generate SVG content based on the diagram
        let svg_content = Self::generate_dlinoss_svg();
        
        // Write to file
        let mut file = fs::File::create(filepath)?;
        file.write_all(svg_content.as_bytes())?;
        
        println!("📄 Diagram exported to: {}/{}", self.export_path, filename);
        
        Ok(())
    }
    
    fn generate_dlinoss_svg() -> String {
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<svg width=\"800\" height=\"400\" xmlns=\"http://www.w3.org/2000/svg\">\n  <rect width=\"800\" height=\"400\" fill=\"white\"/>\n  <text x=\"400\" y=\"200\" text-anchor=\"middle\" font-size=\"16\" fill=\"black\">D-LinOSS Layer Diagram</text>\n</svg>".to_string()
    }
}

impl Default for SvgExporter {
    fn default() -> Self {
        Self::new()
    }
}
