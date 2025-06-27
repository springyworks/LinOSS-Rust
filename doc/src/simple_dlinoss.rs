// Simple D-LinOSS implementation for WASM compatibility
// This is a standalone implementation that doesn't depend on the Burn framework

use nalgebra::{Vector3, Matrix3};

/// Simple RNG for generating noise (WASM-compatible)
#[derive(Clone)]
pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    
    pub fn next_f32(&mut self) -> f32 {
        // Linear congruential generator
        self.state = self.state.wrapping_mul(1103515245).wrapping_add(12345);
        ((self.state >> 16) & 0x7fff) as f32 / 32768.0
    }
    
    pub fn next_f32_range(&mut self, min: f32, max: f32) -> f32 {
        min + (max - min) * self.next_f32()
    }
}

/// Simple D-LinOSS oscillator for web demo
#[derive(Clone)]
pub struct SimpleLinOSS {
    pub state: Vector3<f32>,
    pub velocity: Vector3<f32>,
    pub params: LinOSSParams,
    pub time: f32,
    pub dt: f32,
    pub rng: SimpleRng,
}

#[derive(Clone)]
pub struct LinOSSParams {
    pub frequency: f32,
    pub damping: f32,
    pub coupling: f32,
    pub nonlinearity: f32,
    pub noise_level: f32,
}

impl Default for LinOSSParams {
    fn default() -> Self {
        Self {
            frequency: 10.0,
            damping: 0.1,
            coupling: 0.5,
            nonlinearity: 0.8,
            noise_level: 0.01,
        }
    }
}

impl SimpleLinOSS {
    pub fn new(params: LinOSSParams) -> Self {
        Self {
            state: Vector3::new(0.1, 0.0, 0.0),
            velocity: Vector3::zeros(),
            params,
            time: 0.0,
            dt: 0.001,
            rng: SimpleRng::new(42),
        }
    }
    
    pub fn step(&mut self) {
        // D-LinOSS dynamics: simplified version
        let omega = 2.0 * std::f32::consts::PI * self.params.frequency;
        let gamma = self.params.damping;
        let beta = self.params.nonlinearity;
        let coupling = self.params.coupling;
        
        // Coupling matrix (simplified)
        let coupling_matrix = Matrix3::new(
            0.0, coupling, -coupling,
            -coupling, 0.0, coupling,
            coupling, -coupling, 0.0
        );
        
        // Nonlinear term
        let nonlinear = Vector3::new(
            beta * self.state.x * (self.state.x * self.state.x - 1.0),
            beta * self.state.y * (self.state.y * self.state.y - 1.0),
            beta * self.state.z * (self.state.z * self.state.z - 1.0),
        );
        
        // Noise
        let noise = Vector3::new(
            self.rng.next_f32_range(-1.0, 1.0) * self.params.noise_level,
            self.rng.next_f32_range(-1.0, 1.0) * self.params.noise_level,
            self.rng.next_f32_range(-1.0, 1.0) * self.params.noise_level,
        );
        
        // Update equations (simplified Euler integration)
        let acceleration = -omega * omega * self.state 
            - 2.0 * gamma * omega * self.velocity
            + coupling_matrix * self.state
            + nonlinear
            + noise;
        
        self.velocity += acceleration * self.dt;
        self.state += self.velocity * self.dt;
        self.time += self.dt;
        
        // Apply some bounds to prevent explosion
        for i in 0..3 {
            if self.state[i].abs() > 10.0 {
                self.state[i] = self.state[i].signum() * 10.0;
                self.velocity[i] *= 0.5;
            }
        }
    }
    
    pub fn get_trajectory(&self, history: &[(f32, Vector3<f32>)]) -> Vec<Vector3<f32>> {
        history.iter().map(|(_, state)| *state).collect()
    }
    
    pub fn reset(&mut self) {
        self.state = Vector3::new(0.1, 0.0, 0.0);
        self.velocity = Vector3::zeros();
        self.time = 0.0;
        self.rng = SimpleRng::new(42);
    }
}
