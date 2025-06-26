// Neural Oscillator Vertex Shader
// D-LinOSS dynamics with true 3D perspective projection

uniform mat4 u_view;
uniform mat4 u_proj;
uniform float u_time;
uniform float u_alpha;
uniform float u_beta;
uniform float u_gamma;
uniform vec3 u_positions[256]; // Max 256 oscillators

out vec3 v_color;
out vec3 v_world_pos;
out float v_depth;

void main() {
    int vertex_id = gl_VertexID;
    
    // Get position from uniform array or generate procedurally
    vec3 pos;
    if (vertex_id < u_positions.length()) {
        pos = u_positions[vertex_id];
    } else {
        // Fallback procedural generation
        float t = u_time;
        float phase = float(vertex_id) * 0.1;
        
        // D-LinOSS dynamics
        pos.x = u_alpha * sin(t + phase) * 2.0;
        pos.y = u_beta * cos(t + phase + 1.57) * 1.5; // pi/2 phase shift
        pos.z = u_gamma * sin(t * 0.5 + phase) * cos(t * 0.3 + phase) * 1.2;
    }
    
    v_world_pos = pos;
    
    // Transform to camera space for depth calculation
    vec4 view_pos = u_view * vec4(pos, 1.0);
    v_depth = -view_pos.z;
    
    // Final transformation
    gl_Position = u_proj * view_pos;
    
    // Dynamic coloring based on D-LinOSS parameters and position
    float color_factor = (sin(u_time + float(vertex_id) * 0.2) + 1.0) * 0.5;
    float depth_factor = (pos.z + 3.0) / 6.0; // Normalize Z to [0,1]
    
    // Color palette inspired by neural activity
    vec3 color1 = vec3(1.0, 0.3, 0.5);  // Neural red
    vec3 color2 = vec3(0.3, 0.7, 1.0);  // Synapse blue  
    vec3 color3 = vec3(0.7, 1.0, 0.3);  // Activity green
    
    float osc_factor = float(vertex_id % 3) / 2.0;
    if (osc_factor < 0.33) {
        v_color = mix(color1, color2, color_factor);
    } else if (osc_factor < 0.66) {
        v_color = mix(color2, color3, color_factor);
    } else {
        v_color = mix(color3, color1, color_factor);
    }
    
    // Add depth-based intensity variation
    v_color *= 0.7 + 0.3 * depth_factor;
    
    // Dynamic point size based on depth and D-LinOSS parameters
    float base_size = 8.0;
    float param_variation = 2.0 * (u_alpha + u_beta) / 4.0; // Scale with params
    float time_variation = 2.0 * (sin(u_time * 2.0 + float(vertex_id) * 0.3) + 1.0);
    float depth_scale = max(0.3, 1.0 / (1.0 + v_depth * 0.05)); // Perspective scaling
    
    gl_PointSize = (base_size + param_variation + time_variation) * depth_scale;
}
