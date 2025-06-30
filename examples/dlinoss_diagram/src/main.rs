//! D-LinOSS Academic Diagram Generator
//! 
//! This application provides an interactive diagram creation tool for academic
//! and educational purposes, featuring various diagram types and SVG export capabilities.

mod diagram_renderer_simple;
// mod publication_style; // Commented out as it's unused
mod svg_export;

use diagram_renderer_simple::DLinossDiagramApp;

fn main() -> Result<(), eframe::Error> {
    env_logger::init();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 800.0])
            .with_title("D-LinOSS Layer & Block Diagram Generator")
            .with_position([50.0, 50.0]), // Position window near top-left of screen
        ..Default::default()
    };

    eframe::run_native(
        "D-LinOSS Academic Diagram Generator",
        options,
        Box::new(|cc| {
            Ok(Box::new(DLinossDiagramApp::new(cc)))
        }),
    )
}
