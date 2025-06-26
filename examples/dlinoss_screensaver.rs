//! dLinOSS Mesmerizing Screensaver
//! 
//! A brain-inspired visualization using dLinOSS as universal building blocks
//! Creates beautiful oscillatory patterns that mirror neural dynamics

use linoss_rust::visualization::run_dlinoss_screensaver;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 dLinOSS Brain Dynamics Screensaver");
    println!("=====================================");
    println!("Using dLinOSS as universal building blocks for mesmerizing visuals");
    println!("");
    println!("Controls:");
    println!("  [Q] or [ESC] - Quit");
    println!("  [R] - Reset visualization");
    println!("");
    println!("Watch as damped oscillatory dynamics create brain-like patterns...");
    println!("Press any key to start...");
    
    // Wait for user input
    std::io::stdin().read_line(&mut String::new())?;
    
    // Run the screensaver
    run_dlinoss_screensaver()?;
    
    println!("Thank you for exploring dLinOSS brain dynamics! 🌟");
    
    Ok(())
}
