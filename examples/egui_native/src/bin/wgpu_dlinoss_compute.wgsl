// WGPU compute shader for D-LinOSS neural dynamics
// This shader implements D-LinOSS oscillator dynamics directly on the GPU
// for zero-copy integration with 3D visualization rendering
//
// Key features:
// - Real-time D-LinOSS neural dynamics computation
// - Inter-oscillator coupling and phase relationships
// - GPU memory shared between compute and rendering
// - Optimized for parallel execution (64 oscillators per workgroup)

struct TimeUniforms {
    current_time: f32,
    delta_time: f32,
    frequency: f32,
    amplitude: f32,
}

struct LinossParams {
    alpha: f32,           // X oscillation strength
    beta: f32,            // Y oscillation strength  
    gamma: f32,           // Z oscillation strength / damping
    coupling: f32,        // Inter-oscillator coupling strength
    damping_enabled: u32, // Boolean flag for damping
    noise_level: f32,     // Neural noise injection
    reserved1: f32,       // Padding for alignment
    reserved2: f32,       // Padding for alignment
}

@group(0) @binding(0) var<uniform> time_uniforms: TimeUniforms;
@group(0) @binding(1) var<storage, read_write> positions: array<vec3<f32>>;
@group(0) @binding(2) var<storage, read_write> velocities: array<vec3<f32>>;
@group(0) @binding(3) var<storage, read_write> neural_outputs: array<f32>;

// Simple pseudo-random number generator for neural noise
fn random(seed: f32) -> f32 {
    return fract(sin(seed * 12.9898 + 78.233) * 43758.5453);
}

// D-LinOSS dynamics with GPU-optimized implementation
fn dlinoss_update(
    oscillator_id: u32,
    current_pos: vec3<f32>,
    current_vel: vec3<f32>,
    t: f32,
    dt: f32
) -> vec3<f32> {
    let total_oscillators = arrayLength(&positions);
    let phase = f32(oscillator_id) * 6.28318530718 / f32(total_oscillators);
    let osc_ratio = f32(oscillator_id) / f32(total_oscillators);
    
    // D-LinOSS parameters (future: could be dynamic uniforms)
    let alpha = 1.2;
    let beta = 0.8; 
    let gamma = 0.5;
    let coupling = 0.3;
    
    let freq = time_uniforms.frequency;
    let amp = time_uniforms.amplitude;
    
    // Individual frequency shifts for oscillator diversity
    let freq_x = freq * (1.0 + 0.1 * osc_ratio);
    let freq_y = freq * (1.1 + 0.05 * osc_ratio);
    let freq_z = freq * (0.8 + 0.15 * osc_ratio);
    
    // Exponential damping (D-LinOSS characteristic)
    let damping = exp(-gamma * t * 0.1);
    
    // Base neural oscillations
    let base_x = alpha * sin(t * freq_x + phase) * damping;
    let base_y = beta * cos(t * freq_y + phase + 1.5707963267948966) * damping;
    let base_z = gamma * sin(t * freq_z * 0.5 + phase) * cos(t * 0.3 + phase) * damping;
    
    // D-LinOSS coupling: current oscillator influenced by previous ones
    var coupled_pos = vec3<f32>(base_x, base_y, base_z);
    
    // Sequential coupling (characteristic of D-LinOSS)
    if (oscillator_id > 0u) {
        let prev_pos = positions[oscillator_id - 1u];
        let coupling_strength = coupling * 0.1;
        coupled_pos += prev_pos * coupling_strength;
    }
    
    // Long-range coupling (every 4th oscillator for complexity)
    if (oscillator_id >= 4u) {
        let long_range_pos = positions[oscillator_id - 4u];
        let long_coupling = coupling * 0.05;
        coupled_pos += long_range_pos * long_coupling;
    }
    
    // Neural noise injection for biological realism
    let noise_seed = f32(oscillator_id) * t * 0.1;
    let noise_x = (random(noise_seed) - 0.5) * 0.01;
    let noise_y = (random(noise_seed + 1.0) - 0.5) * 0.01;
    let noise_z = (random(noise_seed + 2.0) - 0.5) * 0.01;
    coupled_pos += vec3<f32>(noise_x, noise_y, noise_z);
    
    return coupled_pos;
}

// Main compute shader entry point
@compute @workgroup_size(64)
fn neural_dynamics(@builtin(global_invocation_id) id: vec3<u32>) {
    let oscillator_id = id.x;
    let total_oscillators = arrayLength(&positions);
    
    // Bounds checking
    if (oscillator_id >= total_oscillators) {
        return;
    }
    
    let t = time_uniforms.current_time;
    let dt = time_uniforms.delta_time;
    
    // Current state
    let current_pos = positions[oscillator_id];
    let current_vel = velocities[oscillator_id];
    
    // Compute new position using D-LinOSS dynamics
    let new_pos = dlinoss_update(oscillator_id, current_pos, current_vel, t, dt);
    
    // Update velocity with momentum (for smooth dynamics)
    let momentum = 0.1;
    let new_vel = momentum * current_vel + (1.0 - momentum) * (new_pos - current_pos);
    
    // Write back to GPU buffers (shared with rendering)
    positions[oscillator_id] = new_pos;
    velocities[oscillator_id] = new_vel;
    
    // Compute neural output signal (used for visualization coloring)
    // This combines all spatial dimensions with different weights
    let neural_signal = (new_pos.x + new_pos.y * 0.7 + new_pos.z * 0.3) * time_uniforms.amplitude;
    neural_outputs[oscillator_id] = neural_signal;
    
    // Advanced feature: Cross-oscillator influence (future enhancement)
    // This could implement more sophisticated D-LinOSS coupling patterns
    // or even real-time learning/adaptation of coupling strengths
}
