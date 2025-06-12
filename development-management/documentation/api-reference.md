# LinOSS Rust API Reference

## Core Modules

### `linoss_rust::linoss`

Main module containing all LinOSS implementations and utilities.

#### Re-exports
```rust
pub use layer::{LinossLayer, LinossLayerConfig};
pub use model::{FullLinossModel, LinossOutput};
pub use dlinoss_layer::{DLinossLayer, DLinossLayerConfig};
```

---

## LinOSS Layer

### `LinossLayer<B: Backend>`

Core Linear Oscillatory State-Space layer implementation using LinOSS-IM (Implicit Method).

#### Configuration

```rust
pub struct LinossLayerConfig {
    pub d_state_m: usize,           // State dimension
    pub d_input_p: usize,           // Input dimension  
    pub d_output_q: usize,          // Output dimension
    pub delta_t: f32,               // Time step
    pub init_std: f64,              // Initialization std dev
    pub enable_d_feedthrough: bool, // Enable direct feedthrough
}
```

#### Methods

##### `forward_step`
```rust
pub fn forward_step(
    &self,
    input: Tensor<B, 2>,                    // Shape: [batch, d_input_p]
    hidden_state: Option<Tensor<B, 2>>      // Shape: [batch, d_state_m]
) -> LinossOutput<B, 2>
```

Performs a single forward step through the LinOSS layer.

**Parameters:**
- `input`: Input tensor for current time step
- `hidden_state`: Optional previous hidden state (zeros if None)

**Returns:**
- `LinossOutput` containing output tensor and next hidden state

##### Accessor Methods
```rust
pub fn d_state_m(&self) -> usize;   // Get state dimension
pub fn d_input_p(&self) -> usize;   // Get input dimension
pub fn d_output_q(&self) -> usize;  // Get output dimension
pub fn delta_t(&self) -> f32;       // Get time step
```

#### Example Usage

```rust
use linoss_rust::linoss::{LinossLayer, LinossLayerConfig};
use burn::backend::NdArray;

type Backend = NdArray<f32>;

// Create configuration
let config = LinossLayerConfig {
    d_state_m: 8,
    d_input_p: 1,
    d_output_q: 1,
    delta_t: 0.1,
    init_std: 0.02,
    enable_d_feedthrough: false,
};

// Initialize layer
let layer: LinossLayer<Backend> = config.init(&device);

// Single step forward pass
let input = Tensor::zeros([batch_size, 1], &device);
let output = layer.forward_step(input, None);
```

---

## D-LinOSS Layer

### `DLinossLayer<B: Backend>`

Damped Linear Oscillatory State-Space layer with learnable energy dissipation.

#### Configuration

```rust
pub struct DLinossLayerConfig {
    pub d_input: usize,             // Input dimension
    pub d_model: usize,             // Model dimension (must be even)
    pub d_output: usize,            // Output dimension
    pub delta_t: f64,               // Time step
    pub init_std: f64,              // Initialization std dev
    pub enable_layer_norm: bool,    // Enable layer normalization
    pub enable_damping: bool,       // Enable learnable damping
    pub init_damping: f64,          // Initial damping coefficient
    pub num_damping_scales: usize,  // Number of damping timescales
}
```

#### Configuration Constructors

##### `new_dlinoss`
```rust
pub fn new_dlinoss(d_input: usize, d_model: usize, d_output: usize) -> Self
```

Creates D-LinOSS configuration with damping enabled.

##### `vanilla_linoss`
```rust
pub fn vanilla_linoss(d_input: usize, d_model: usize, d_output: usize) -> Self
```

Creates vanilla LinOSS configuration (damping disabled).

#### Methods

##### `new`
```rust
pub fn new(config: &DLinossLayerConfig, device: &B::Device) -> Self
```

Creates a new D-LinOSS layer from configuration.

##### `forward`
```rust
pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3>
```

Forward pass through entire sequence.

**Parameters:**
- `input`: Input tensor of shape `[batch, sequence_length, d_input]`

**Returns:**
- Output tensor of shape `[batch, sequence_length, d_output]`

##### Analysis Methods
```rust
pub fn get_damping_coefficients(&self) -> Option<Tensor<B, 2>>;
pub fn get_damping_scales(&self) -> Option<Tensor<B, 1>>;
pub fn has_damping(&self) -> bool;
```

#### Example Usage

```rust
use linoss_rust::linoss::{DLinossLayer, DLinossLayerConfig};
use burn::backend::{Autodiff, NdArray};

type Backend = Autodiff<NdArray<f32>>;

// Create D-LinOSS with damping
let config = DLinossLayerConfig::new_dlinoss(1, 16, 1);
let layer = DLinossLayer::<Backend>::new(&config, &device);

// Sequence forward pass
let input = Tensor::zeros([batch_size, seq_len, 1], &device);
let output = layer.forward(input);

// Check damping status
if layer.has_damping() {
    println!("Damping coefficients: {:?}", layer.get_damping_coefficients());
}
```

---

## Full LinOSS Model

### `FullLinossModel<B: Backend>`

High-level model container for complete LinOSS-based sequence models.

#### Configuration

```rust
pub struct FullLinossModelConfig {
    pub d_input: usize,     // Input dimension
    pub d_model: usize,     // Hidden dimension
    pub d_output: usize,    // Output dimension
    pub num_layers: usize,  // Number of layers
    pub delta_t: f32,       // Time step
    pub init_std: f64,      // Initialization std dev
}
```

#### Methods

##### `forward`
```rust
pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3>
```

Forward pass through the complete model.

**Parameters:**
- `input`: Input tensor of shape `[batch, sequence_length, d_input]`

**Returns:**
- Output tensor of shape `[batch, sequence_length, d_output]`

#### Example Usage

```rust
use linoss_rust::linoss::{FullLinossModel, FullLinossModelConfig};

let config = FullLinossModelConfig {
    d_input: 1,
    d_model: 64,
    d_output: 1,
    num_layers: 4,
    delta_t: 0.1,
    init_std: 0.02,
};

let model: FullLinossModel<Backend> = config.init(&device);
let output = model.forward(input);
```

---

## Utility Types

### `LinossOutput<B: Backend, const D: usize>`

Output container for LinOSS layer forward passes.

```rust
pub struct LinossOutput<B: Backend, const D: usize> {
    pub output: Tensor<B, D>,           // Layer output
    pub hidden_state: Option<Tensor<B, D>>, // Next hidden state
}
```

---

## Backend Requirements

### Supported Backends

| Backend | Features | Use Case |
|---------|----------|----------|
| `NdArray<f32>` | CPU computation | Development, testing |
| `Autodiff<NdArray<f32>>` | CPU + gradients | Training |
| `Wgpu<f32>` | GPU computation | Inference |
| `Autodiff<Wgpu<f32>>` | GPU + gradients | GPU training |

### Backend Selection Example

```rust
// CPU-only inference
type CpuBackend = NdArray<f32>;

// CPU training
type CpuTrainBackend = Autodiff<NdArray<f32>>;

// GPU inference  
type GpuBackend = Wgpu<f32>;

// GPU training
type GpuTrainBackend = Autodiff<Wgpu<f32>>;

// Usage
let layer: DLinossLayer<CpuTrainBackend> = config.init(&device);
```

---

## Training Utilities

### Optimizer Integration

```rust
use burn::optim::{AdamConfig, GradientsParams, Optimizer};

// Create optimizer
let mut optimizer = AdamConfig::new().init();

// Training step
let output = model.forward(input);
let loss = compute_loss(output, target);
let grads = loss.backward();
let grads = GradientsParams::from_grads(grads, &model);
model = optimizer.step(learning_rate, model, grads);
```

### Loss Functions

Common loss functions for sequence modeling:

```rust
// Mean Squared Error
let mse_loss = (predictions - targets).powi_scalar(2).mean();

// Mean Absolute Error  
let mae_loss = (predictions - targets).abs().mean();

// Huber Loss (smooth L1)
let huber_loss = smooth_l1_loss(predictions, targets, beta);
```

---

## Error Handling

### Common Error Types

Most methods return `Result<T, E>` where `E` can be:

- **Configuration Errors**: Invalid dimensions, parameters
- **Tensor Errors**: Shape mismatches, device mismatches
- **Numerical Errors**: NaN/Inf values, instability

### Error Handling Example

```rust
match layer.forward_step(input, hidden_state) {
    Ok(output) => {
        // Process output
        println!("Forward pass successful");
    },
    Err(e) => {
        eprintln!("Forward pass failed: {}", e);
        // Handle error appropriately
    }
}
```

---

## Performance Considerations

### Memory Usage

- **LinOSS Layer**: O(M×(P+Q)) parameters
- **D-LinOSS Layer**: O(M×(P+Q) + K×M/2) parameters  
- **Sequence Processing**: O(B×T×M) temporary memory

### Computational Complexity

- **Sequential Processing**: O(T) time steps
- **Parallel Scan**: O(log T) with sufficient parallelism
- **Parameter Count**: Linear in dimensions

### Optimization Tips

1. **Use appropriate backend** for your hardware
2. **Enable layer normalization** for training stability
3. **Choose even d_model** for oscillatory pairs
4. **Monitor damping coefficients** in D-LinOSS
5. **Use parallel scan** for long sequences

---

## Examples and Tutorials

See the `examples/` directory for complete working examples:

- `basic_usage.rs` - Simple LinOSS layer usage
- `dlinoss_comparison.rs` - D-LinOSS vs vanilla LinOSS
- `compare_scan_methods.rs` - Performance comparison

Last updated: June 12, 2025
