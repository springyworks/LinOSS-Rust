//! dLinOSS Mesmerizing Visualizer
//! 
//! A brain-inspired visualization using dLinOSS as universal building blocks
//! Creates beautiful oscillatory patterns that mirror neural dynamics

use linoss_rust::visualization::run_dlinoss_visualizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 dLinOSS Brain Dynamics Visualizer");
    println!("=====================================");
    println!("Using dLinOSS as universal building blocks for mesmerizing visuals");
    println!("");
    println!("Controls:");
    println!("  [Q] or [ESC] - Quit");
    println!("  [R] - Reset visualization");
    println!("");
    println!("Watch as damped oscillatory dynamics create brain-like patterns...");
    println!("Press any key to start (or just wait 3 seconds)...");
    
    // Wait for key press with timeout
    use crossterm::{
        terminal::{enable_raw_mode, disable_raw_mode},
        event::{self, Event, poll},
    };
    use std::time::Duration;
    
    enable_raw_mode()?;
    
    // Wait for key or timeout
    let mut started = false;
    for _ in 0..30 { // 30 * 100ms = 3 seconds
        if poll(Duration::from_millis(100))? {
            if let Event::Key(_) = event::read()? {
                started = true;
                break;
            }
        }
    }
    
    disable_raw_mode()?;
    
    if !started {
        println!("Auto-starting...");
    } else {
        println!("Key detected! Starting...");
    }
    
    // Run the visualizer
    run_dlinoss_visualizer()?;
    
    println!("Thank you for exploring dLinOSS brain dynamics! 🌟");
    
    Ok(())
}
