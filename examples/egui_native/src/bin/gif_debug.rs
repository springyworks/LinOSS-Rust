//! Simple GIF frame counter to debug animation issues

use std::io::Cursor;
use image::{AnimationDecoder, codecs::gif::GifDecoder};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Analyzing GIF file...");
    
    let gif_path = "assets/image015.gif";
    let file_data = std::fs::read(gif_path)?;
    let cursor = Cursor::new(file_data);
    let decoder = GifDecoder::new(cursor)?;
    let frames = decoder.into_frames();
    
    let mut frame_count = 0;
    let mut total_duration = std::time::Duration::ZERO;
    
    for (i, frame_result) in frames.enumerate() {
        let frame = frame_result?;
        let delay = frame.delay();
        let duration = std::time::Duration::from_millis(
            (delay.numer_denom_ms().0 as u64 * 1000) / delay.numer_denom_ms().1 as u64
        );
        
        let (width, height) = frame.buffer().dimensions();
        
        println!("Frame {}: {}x{}, duration: {:?}ms", 
            i, width, height, duration.as_millis());
        
        frame_count += 1;
        total_duration += duration;
    }
    
    println!("\n📊 Summary:");
    println!("Total frames: {}", frame_count);
    println!("Total duration: {:?}ms", total_duration.as_millis());
    println!("Average FPS: {:.1}", 1000.0 * frame_count as f32 / total_duration.as_millis() as f32);
    
    if frame_count <= 1 {
        println!("⚠️  WARNING: GIF has only {} frame(s) - not animated!", frame_count);
    } else {
        println!("✅ GIF is animated with {} frames", frame_count);
    }
    
    Ok(())
}
