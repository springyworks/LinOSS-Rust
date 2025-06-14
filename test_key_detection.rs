// Simple test for key detection
use crossterm::{
    terminal::{enable_raw_mode, disable_raw_mode},
    event::{self, Event},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing key detection...");
    println!("Press any key (this should work without needing Enter):");
    
    enable_raw_mode()?;
    
    loop {
        match event::read()? {
            Event::Key(key_event) => {
                disable_raw_mode()?;
                println!("Key detected! {:?}", key_event);
                break;
            }
            _ => {}
        }
    }
    
    println!("Key detection working correctly!");
    Ok(())
}
