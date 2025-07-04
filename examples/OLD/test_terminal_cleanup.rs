//! Quick test program to verify terminal cleanup works
use crossterm::{
    terminal::{enable_raw_mode, disable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    execute,
    cursor,
};
use std::io::{stdout, Write};
use std::time::Duration;

fn cleanup_terminal() {
    let _ = disable_raw_mode();
    let _ = execute!(stdout(), LeaveAlternateScreen);
    let _ = execute!(stdout(), cursor::Show);
    let _ = stdout().flush();
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing terminal cleanup...");
    
    // Setup terminal
    enable_raw_mode()?;
    execute!(stdout(), EnterAlternateScreen)?;
    
    println!("Terminal in raw mode. This will auto-cleanup in 2 seconds...");
    std::thread::sleep(Duration::from_secs(2));
    
    // Always cleanup
    cleanup_terminal();
    
    println!("Terminal should be restored now!");
    
    Ok(())
}
