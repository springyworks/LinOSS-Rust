// ⚡ Neural Compute Shader for GPU-only processing ⚡
// Processes neural oscillator data entirely on GPU
// Input: positions and velocities from Burn WGPU backend
// Output: enhanced/filtered neural dynamics for WGPU renderer

@group(0) @binding(0)
var<storage, read_write> input_data: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output_data: array<f32>;

// Neural processing parameters
const OSCILLATOR_SIZE: u32 = 6u; // 3 position + 3 velocity components
const ENHANCEMENT_FACTOR: f32 = 1.2;
const COUPLING_STRENGTH: f32 = 0.1;
const TIME_STEP: f32 = 0.016; // ~60 FPS

@compute @workgroup_size(8, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let oscillator_id = global_id.x;
    let total_oscillators = arrayLength(&input_data) / OSCILLATOR_SIZE;
    
    // Bounds check
    if (oscillator_id >= total_oscillators) {
        return;
    }
    
    let base_idx = oscillator_id * OSCILLATOR_SIZE;
    
    // Read current oscillator state
    let pos_x = input_data[base_idx + 0u];
    let pos_y = input_data[base_idx + 1u];
    let pos_z = input_data[base_idx + 2u];
    let vel_x = input_data[base_idx + 3u];
    let vel_y = input_data[base_idx + 4u];
    let vel_z = input_data[base_idx + 5u];
    
    // 🔥 GPU-only neural processing algorithms
    
    // 1. Nonlinear enhancement for dramatic Z-axis movement
    let z_enhancement = sin(pos_z * 2.0) * ENHANCEMENT_FACTOR;
    let enhanced_pos_z = pos_z + z_enhancement * TIME_STEP;
    
    // 2. Inter-oscillator coupling (GPU-parallel computation)
    var coupling_force_x = 0.0;
    var coupling_force_y = 0.0;
    var coupling_force_z = 0.0;
    
    // Sample coupling with nearby oscillators (limited for performance)
    let coupling_range = min(4u, total_oscillators);
    for (var i = 0u; i < coupling_range; i += 1u) {
        let neighbor_id = (oscillator_id + i + 1u) % total_oscillators;
        let neighbor_base = neighbor_id * OSCILLATOR_SIZE;
        
        let neighbor_x = input_data[neighbor_base + 0u];
        let neighbor_y = input_data[neighbor_base + 1u];
        let neighbor_z = input_data[neighbor_base + 2u];
        
        // Distance-based coupling
        let dx = neighbor_x - pos_x;
        let dy = neighbor_y - pos_y;
        let dz = neighbor_z - pos_z;
        let distance = sqrt(dx * dx + dy * dy + dz * dz) + 0.001; // Avoid division by zero
        
        let coupling_factor = COUPLING_STRENGTH / (distance * distance);
        coupling_force_x += dx * coupling_factor;
        coupling_force_y += dy * coupling_factor;
        coupling_force_z += dz * coupling_factor;
    }
    
    // 3. Apply nonlinear dynamics for visual appeal
    let nonlinear_x = pos_x + sin(pos_y * 1.5) * 0.1;
    let nonlinear_y = pos_y + cos(pos_x * 1.3) * 0.1;
    let nonlinear_z = enhanced_pos_z + sin(pos_x + pos_y) * 0.15;
    
    // 4. Velocity damping with coupling influence
    let damping = 0.98;
    let enhanced_vel_x = (vel_x + coupling_force_x) * damping;
    let enhanced_vel_y = (vel_y + coupling_force_y) * damping;
    let enhanced_vel_z = (vel_z + coupling_force_z) * damping;
    
    // 5. Apply time-based modulation for dynamic effects
    let time_phase = f32(oscillator_id) * 0.1 + pos_x * 0.1;
    let time_modulation = sin(time_phase) * 0.05;
    
    // Write enhanced results back to output buffer
    output_data[base_idx + 0u] = nonlinear_x + time_modulation;
    output_data[base_idx + 1u] = nonlinear_y + time_modulation * 0.7;
    output_data[base_idx + 2u] = nonlinear_z + time_modulation * 1.2; // Enhanced Z movement
    output_data[base_idx + 3u] = enhanced_vel_x;
    output_data[base_idx + 4u] = enhanced_vel_y;
    output_data[base_idx + 5u] = enhanced_vel_z;
    
    // 🎨 Additional GPU-computed rendering hints
    // These could be used by the renderer for color/size modulation
    let activity_level = sqrt(enhanced_vel_x * enhanced_vel_x + 
                             enhanced_vel_y * enhanced_vel_y + 
                             enhanced_vel_z * enhanced_vel_z);
    
    // Store activity level in unused buffer space (if available)
    // This demonstrates bidirectional data flow: computation → rendering hints
}
