//! Simple test for robust terminal cleanup
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use std::time::Duration;
use std::io::{stdout, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing robust terminal cleanup...");
    
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    
    // Setup robust signal handler
    ctrlc::set_handler(move || {
        println!("Signal received - cleaning up...");
        r.store(false, Ordering::SeqCst);
        // Force clean terminal state
        print!("\x1b[?25h\x1b[2J\x1b[H"); // Show cursor, clear screen, go to home
        let _ = stdout().flush();
        std::process::exit(0);
    })?;
    
    println!("Running for 5 seconds... Try Ctrl+C or timeout");
    
    let mut counter = 0;
    while running.load(Ordering::SeqCst) && counter < 50 {
        println!("Count: {}", counter);
        std::thread::sleep(Duration::from_millis(100));
        counter += 1;
    }
    
    println!("Finished normally!");
    Ok(())
}
