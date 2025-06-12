# Getting Started with LinOSS Rust

Welcome to LinOSS Rust! This guide will help you get up and running with Linear Oscillatory State-Space Models in Rust.

## What is LinOSS?

LinOSS (Linear Oscillatory State-Space) models are a class of neural networks designed for sequence modeling. They use oscillatory dynamics to capture temporal patterns, making them particularly effective for:

- **Time series forecasting**
- **Signal processing** 
- **Long-range dependency modeling**
- **Periodic pattern recognition**

## Installation

### Prerequisites

- **Rust 1.70+**: Install from [rustup.rs](https://rustup.rs/)
- **GPU Support (Optional)**: CUDA or ROCm for GPU acceleration

### Adding to Your Project

Add LinOSS Rust to your `Cargo.toml`:

```toml
[dependencies]
linoss_rust = "0.2.0"
burn = "0.17"
```

For GPU support:
```toml
[dependencies]
linoss_rust = "0.2.0"
burn = { version = "0.17", features = ["wgpu"] }
```

### Clone from Source

```bash
git clone https://github.com/example/linoss-rust.git
cd linoss-rust
cargo test  # Run tests to verify installation
```

## Quick Start

### 1. Basic LinOSS Layer

```rust
use linoss_rust::linoss::{LinossLayer, LinossLayerConfig};
use burn::backend::NdArray;
use burn::tensor::Tensor;

fn main() {
    // Choose backend
    type Backend = NdArray<f32>;
    let device = Default::default();
    
    // Configure layer
    let config = LinossLayerConfig {
        d_state_m: 8,      // Hidden state dimension
        d_input_p: 1,      // Input dimension
        d_output_q: 1,     // Output dimension
        delta_t: 0.1,      // Time step
        init_std: 0.02,    // Parameter initialization
        enable_d_feedthrough: false,
    };
    
    // Create layer
    let layer: LinossLayer<Backend> = config.init(&device);
    
    // Create input (single time step)
    let input = Tensor::zeros([1, 1], &device); // [batch=1, input_dim=1]
    
    // Forward pass
    let output = layer.forward_step(input, None);
    println!("Output shape: {:?}", output.output.dims());
}
```

### 2. D-LinOSS with Damping

```rust
use linoss_rust::linoss::{DLinossLayer, DLinossLayerConfig};
use burn::backend::{Autodiff, NdArray};

fn main() {
    // Use autodiff backend for training
    type Backend = Autodiff<NdArray<f32>>;
    let device = Default::default();
    
    // Create D-LinOSS with damping
    let config = DLinossLayerConfig::new_dlinoss(1, 16, 1);
    let layer = DLinossLayer::<Backend>::new(&config, &device);
    
    // Create sequence input
    let batch_size = 4;
    let seq_len = 50;
    let input = Tensor::zeros([batch_size, seq_len, 1], &device);
    
    // Forward pass through entire sequence
    let output = layer.forward(input);
    println!("Output shape: {:?}", output.dims()); // [4, 50, 1]
    
    // Check if damping is active
    if layer.has_damping() {
        println!("Damping is enabled!");
    }
}
```

## Core Concepts

### 1. Oscillatory Dynamics

LinOSS models use pairs of oscillators to capture temporal patterns:

```rust
// d_model must be even for oscillatory pairs
let config = DLinossLayerConfig {
    d_model: 16,  // 8 oscillator pairs
    // ... other config
};
```

Each pair represents `[position, velocity]` of a damped harmonic oscillator.

### 2. Time Discretization

The `delta_t` parameter controls the time step size:

```rust
let config = LinossLayerConfig {
    delta_t: 0.1,  // Smaller = more stable, larger = faster
    // ...
};
```

**Guidelines:**
- **0.01-0.05**: High-frequency signals
- **0.1**: General purpose (default)
- **0.5+**: Low-frequency signals

### 3. Damping (D-LinOSS)

D-LinOSS adds learnable damping for energy dissipation:

```rust
let config = DLinossLayerConfig {
    enable_damping: true,       // Enable damping
    init_damping: 0.1,         // Initial damping coefficient
    num_damping_scales: 4,     // Multiple timescales
    // ...
};
```

## Common Use Cases

### 1. Time Series Prediction

```rust
use burn::optim::{AdamConfig, GradientsParams, Optimizer};

fn train_time_series() {
    type Backend = Autodiff<NdArray<f32>>;
    let device = Default::default();
    
    // Create model
    let config = DLinossLayerConfig::new_dlinoss(1, 32, 1);
    let mut model = DLinossLayer::<Backend>::new(&config, &device);
    
    // Create optimizer
    let mut optimizer = AdamConfig::new().init();
    
    // Training loop
    for epoch in 0..100 {
        // Forward pass
        let predictions = model.forward(train_input.clone());
        
        // Compute loss
        let loss = (predictions - train_targets.clone()).powi_scalar(2).mean();
        
        // Backward pass
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        
        // Update parameters
        model = optimizer.step(0.001, model, grads);
        
        if epoch % 10 == 0 {
            let loss_val: f32 = loss.into_scalar();
            println!("Epoch {}: Loss = {:.6}", epoch, loss_val);
        }
    }
}
```

### 2. Signal Processing

```rust
fn process_signal() {
    type Backend = NdArray<f32>;
    let device = Default::default();
    
    // Configure for signal processing
    let config = DLinossLayerConfig {
        d_input: 1,
        d_model: 64,      // More oscillators for complex signals
        d_output: 1,
        delta_t: 0.01,    // Fine time resolution
        enable_damping: true,
        // ...
    };
    
    let layer = DLinossLayer::<Backend>::new(&config, &device);
    
    // Process signal
    let filtered_signal = layer.forward(noisy_signal);
}
```

### 3. Pattern Recognition

```rust
fn recognize_patterns() {
    // Use multiple output dimensions for classification
    let config = DLinossLayerConfig::new_dlinoss(
        1,   // Single input channel
        32,  // Hidden oscillators
        10   // 10 pattern classes
    );
    
    let classifier = DLinossLayer::new(&config, &device);
    
    // Get pattern probabilities
    let logits = classifier.forward(input_sequence);
    let probabilities = logits.softmax(2); // Softmax over output dimension
}
```

## Backend Selection

### CPU vs GPU

| Backend | Use Case | Performance | Memory |
|---------|----------|-------------|---------|
| `NdArray<f32>` | Development, small models | Moderate | Low |
| `Autodiff<NdArray<f32>>` | CPU training | Moderate | Moderate |
| `Wgpu<f32>` | GPU inference | High | High |
| `Autodiff<Wgpu<f32>>` | GPU training | Very High | Very High |

### Backend Examples

```rust
// CPU development
type DevBackend = NdArray<f32>;

// CPU training  
type CpuTrainBackend = Autodiff<NdArray<f32>>;

// GPU training
type GpuTrainBackend = Autodiff<Wgpu<f32>>;

// Choose based on your needs
let layer: DLinossLayer<GpuTrainBackend> = config.init(&device);
```

## Troubleshooting

### Common Issues

#### 1. Dimension Mismatch
```
Error: DimensionMismatch { expected: 16, got: 15 }
```
**Solution**: Ensure `d_model` is even for oscillatory pairs.

#### 2. Numerical Instability
```
Error: NumericalInstability
```
**Solutions:**
- Reduce `delta_t` (try 0.01-0.05)
- Lower `init_std` (try 0.01)
- Enable layer normalization

#### 3. GPU Memory Issues
```
Error: OutOfMemory
```
**Solutions:**
- Reduce batch size
- Use smaller `d_model`
- Process sequences in chunks

### Performance Tips

1. **Choose appropriate dimensions**:
   ```rust
   // Good: Powers of 2, reasonable sizes
   d_model: 32,   // 16 oscillator pairs
   
   // Avoid: Too large, odd numbers
   d_model: 1000, // May cause memory issues
   d_model: 15,   // Not even (invalid)
   ```

2. **Monitor damping coefficients**:
   ```rust
   if let Some(damping) = layer.get_damping_coefficients() {
       let avg_damping: f32 = damping.mean().into_scalar();
       println!("Average damping: {:.4}", avg_damping);
       
       // Healthy range: 0.01 - 0.5
       if avg_damping > 1.0 {
           println!("Warning: High damping may suppress dynamics");
       }
   }
   ```

3. **Use appropriate time steps**:
   ```rust
   // For different signal types
   let config = match signal_type {
       SignalType::Audio => DLinossLayerConfig { delta_t: 0.01, .. },
       SignalType::Daily => DLinossLayerConfig { delta_t: 1.0, .. },
       SignalType::General => DLinossLayerConfig { delta_t: 0.1, .. },
   };
   ```

## Next Steps

### Examples to Try

1. **Basic Usage**: `cargo run --example basic_usage`
2. **D-LinOSS Comparison**: `cargo run --example dlinoss_comparison`
3. **Performance Test**: `cargo run --example compare_scan_methods`

### Advanced Topics

- [System Architecture](../architecture/system-overview.md)
- [API Reference](../documentation/api-reference.md)
- [Research Papers](../research-notes/papers.md)
- [Performance Analysis](../performance-analysis/benchmarks.md)

### Getting Help

- **Examples**: Check the `examples/` directory
- **Tests**: Look at unit tests for usage patterns
- **Issues**: Report bugs and ask questions on GitHub
- **Documentation**: Explore the `development-management/` folder

## Community and Contributing

### Contributing

We welcome contributions! Please:

1. Read the [development standards](../development-process/coding-standards.md)
2. Check existing issues and examples
3. Add tests for new features
4. Update documentation

### Research

LinOSS Rust implements cutting-edge research:

- **LinOSS**: arXiv:2410.03943 (2024)
- **D-LinOSS**: arXiv:2505.12171 (2025)

See [research notes](../research-notes/papers.md) for details.

Happy modeling with LinOSS! 🌊

Last updated: June 12, 2025
