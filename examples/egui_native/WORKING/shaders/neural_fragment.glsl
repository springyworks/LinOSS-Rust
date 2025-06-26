// Neural Oscillator Fragment Shader
// Renders neural oscillators as glowing spheres with depth effects

precision mediump float;

in vec3 v_color;
in vec3 v_world_pos;
in float v_depth;

out vec4 out_color;

void main() {
    // Create smooth circular points with neural glow effect
    vec2 coord = gl_PointCoord - vec2(0.5);
    float dist = length(coord);
    
    // Discard pixels outside the circle
    if (dist > 0.5) {
        discard;
    }
    
    // Multi-layer neural glow effect
    float center_core = 1.0 - smoothstep(0.0, 0.1, dist);   // Bright center
    float main_body = 1.0 - smoothstep(0.1, 0.35, dist);    // Main sphere
    float outer_glow = 1.0 - smoothstep(0.35, 0.5, dist);   // Outer glow
    
    // Combine layers with different intensities
    float intensity = center_core * 1.0 + main_body * 0.8 + outer_glow * 0.3;
    
    // Add neural pulse effect
    float pulse = 0.9 + 0.1 * sin(v_depth * 0.5 + v_world_pos.x * 2.0);
    intensity *= pulse;
    
    // Depth-based effects for 3D perception
    float depth_factor = 1.0 / (1.0 + v_depth * 0.03);
    float depth_alpha = 0.8 + 0.2 * depth_factor;
    
    // Color enhancement based on position
    vec3 enhanced_color = v_color;
    
    // Add subtle highlight in center for 3D sphere effect
    float highlight = center_core * 0.4;
    enhanced_color = mix(enhanced_color, vec3(1.0, 1.0, 1.0), highlight);
    
    // Final color with neural glow
    vec3 final_color = enhanced_color * intensity * depth_factor;
    
    // Anti-aliased alpha with depth falloff
    float alpha = outer_glow * depth_alpha;
    
    // Add slight additive blending effect for neural glow
    final_color += v_color * 0.1 * outer_glow;
    
    out_color = vec4(final_color, alpha);
}
