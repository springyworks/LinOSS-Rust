//! Comprehensive signal handling test
//! Tests multiple signals and shows clean terminal restoration

use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use std::time::{Duration, Instant};
use std::io::{stdout, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 Comprehensive Signal Handling Test");
    println!("=====================================");
    println!("This will run for 10 seconds and respond to signals:");
    println!("- SIGTERM (timeout default)");
    println!("- SIGINT (Ctrl+C)");
    println!("- SIGKILL (cannot be caught, but terminal will still be clean)");
    
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    
    // Setup comprehensive signal handler
    ctrlc::set_handler(move || {
        println!("\n🛑 Signal received! Cleaning up gracefully...");
        r.store(false, Ordering::SeqCst);
        
        // Force clean terminal state (same as our TUI apps)
        print!("\x1b[?25h\x1b[2J\x1b[H"); // Show cursor, clear screen, go to home
        let _ = stdout().flush();
        
        println!("✅ Terminal restored successfully!");
        std::process::exit(0);
    })?;
    
    let start_time = Instant::now();
    let mut counter = 0;
    
    println!("🏃 Running... (send signals to test)");
    
    while running.load(Ordering::SeqCst) && start_time.elapsed() < Duration::from_secs(10) {
        print!("\r⏱️  Count: {} | Elapsed: {:02.1}s", counter, start_time.elapsed().as_secs_f32());
        let _ = stdout().flush();
        
        std::thread::sleep(Duration::from_millis(100));
        counter += 1;
        
        // Check signal flag
        if !running.load(Ordering::SeqCst) {
            println!("\n🔄 Signal flag detected, exiting gracefully...");
            break;
        }
    }
    
    println!("\n✅ Finished normally after {} iterations!", counter);
    println!("🎉 Terminal should be clean and working!");
    
    Ok(())
}
