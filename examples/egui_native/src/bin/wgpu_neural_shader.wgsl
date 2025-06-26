// WGSL Shader for LinOSS Neural Oscillator Visualization
// This shader implements the same neural dynamics as the OpenGL version
// but using modern WGSL syntax for WGPU backend

// Uniforms for neural parameters and camera
struct Uniforms {
    view_proj: mat4x4<f32>,
    time: f32,
    alpha: f32,
    beta: f32,
    gamma: f32,
    coupling: f32,
    oscillator_count: u32,
    _padding: vec2<f32>,
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

// Vertex shader output
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(2) depth: f32,
    @location(3) point_size: f32,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let t = uniforms.time;
    let phase = f32(vertex_index) * 6.28318 / f32(uniforms.oscillator_count);
    let normalized_id = f32(vertex_index) / f32(uniforms.oscillator_count);
    
    // LinOSS-inspired dynamics with individual variations
    let freq_x = 1.0 + 0.1 * normalized_id;
    let freq_y = 1.1 + 0.05 * normalized_id;
    let freq_z = 0.8 + 0.15 * normalized_id;
    
    // D-LinOSS damped oscillations
    let damping = exp(-uniforms.gamma * t * 0.1);
    
    // Base neural oscillator positions
    var pos: vec3<f32>;
    pos.x = uniforms.alpha * sin(t * freq_x + phase) * damping;
    pos.y = uniforms.beta * cos(t * freq_y + phase + 1.5707) * damping;
    pos.z = uniforms.gamma * sin(t * freq_z * 0.5 + phase) * cos(t * 0.3 + phase) * damping;
    
    // Add inter-oscillator coupling (LinOSS characteristic)
    if (vertex_index > 0u) {
        let prev_phase = f32(vertex_index - 1u) * 6.28318 / f32(uniforms.oscillator_count);
        let coupling_effect = uniforms.coupling * 0.1;
        
        let prev_pos = vec3<f32>(
            uniforms.alpha * sin(t * freq_x + prev_phase) * damping,
            uniforms.beta * cos(t * freq_y + prev_phase + 1.5707) * damping,
            uniforms.gamma * sin(t * freq_z * 0.5 + prev_phase) * cos(t * 0.3 + prev_phase) * damping
        );
        
        pos += prev_pos * coupling_effect;
    }
    
    // Transform to clip space
    let world_pos_4 = vec4<f32>(pos, 1.0);
    let clip_position = uniforms.view_proj * world_pos_4;
    
    // Calculate depth for effects
    let depth = -clip_position.z;
    
    // Dynamic neural coloring based on energy and activity
    let color_factor = (sin(t + phase) + 1.0) * 0.5;
    let energy = length(pos) / 3.0; // Normalize energy
    
    // LinOSS energy-based color palette
    let low_energy = vec3<f32>(0.2, 0.4, 1.0);   // Blue (low activity)
    let mid_energy = vec3<f32>(0.8, 1.0, 0.2);   // Green (medium activity)
    let high_energy = vec3<f32>(1.0, 0.3, 0.2);  // Red (high activity)
    
    var color: vec3<f32>;
    if (energy < 0.5) {
        color = mix(low_energy, mid_energy, energy * 2.0);
    } else {
        color = mix(mid_energy, high_energy, (energy - 0.5) * 2.0);
    }
    
    // Add temporal variation for neural pulse effects
    color = color * (0.8 + 0.2 * color_factor);
    
    // Dynamic point size based on neural activity and depth
    let base_size = 6.0;
    let activity = (uniforms.alpha + uniforms.beta + uniforms.gamma) / 3.0;
    let size_variation = 3.0 * (sin(t * 2.0 + phase) + 1.0);
    let depth_scale = max(0.3, 1.0 / (1.0 + depth * 0.03));
    let point_size = (base_size + activity * 2.0 + size_variation) * depth_scale;
    
    var out: VertexOutput;
    out.clip_position = clip_position;
    out.color = color;
    out.world_pos = pos;
    out.depth = depth;
    out.point_size = point_size;
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Generate point coordinates for circular rendering
    // Since WGSL doesn't have gl_PointCoord, we simulate it
    let coord_x = (f32(int(in.clip_position.x)) % in.point_size) / in.point_size - 0.5;
    let coord_y = (f32(int(in.clip_position.y)) % in.point_size) / in.point_size - 0.5;
    let coord = vec2<f32>(coord_x, coord_y) * 2.0; // Convert to [-1, 1]
    let dist = length(coord);
    
    if (dist > 1.0) {
        discard;
    }
    
    // Neural oscillator rendering with energy core
    let core = 1.0 - smoothstep(0.0, 0.3, dist);      // Bright center
    let body = 1.0 - smoothstep(0.3, 0.8, dist);      // Main body
    let glow = 1.0 - smoothstep(0.8, 1.0, dist);      // Outer glow
    
    let intensity = core * 1.0 + body * 0.7 + glow * 0.3;
    
    // Neural activity pulse based on position and depth
    let activity_pulse = 0.9 + 0.1 * sin(in.depth * 0.3 + length(in.world_pos) * 2.0);
    let final_intensity = intensity * activity_pulse;
    
    // Depth effects for 3D perception
    let depth_factor = 1.0 / (1.0 + in.depth * 0.02);
    
    // Enhanced color with core brightening
    var enhanced_color = in.color;
    enhanced_color = mix(enhanced_color, vec3<f32>(1.0), core * 0.3);
    
    let final_color = enhanced_color * final_intensity * depth_factor;
    let alpha = glow * (0.9 + 0.1 * depth_factor);
    
    return vec4<f32>(final_color, alpha);
}
