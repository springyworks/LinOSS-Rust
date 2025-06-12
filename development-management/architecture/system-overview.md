# LinOSS Rust System Architecture

## Overview

The LinOSS Rust project implements Linear Oscillatory State-Space Models with a modular, extensible architecture supporting multiple backends, research variants, and deployment targets.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LinOSS Rust System                      │
├─────────────────────────────────────────────────────────────┤
│                   Application Layer                        │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐   │
│  │    CLI      │ │   Web Demo  │ │    Examples &       │   │
│  │   Tools     │ │    (WASM)   │ │   Benchmarks        │   │
│  └─────────────┘ └─────────────┘ └─────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│                      API Layer                             │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              High-Level Models                         │ │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────┐   │ │
│  │  │FullLinoss   │ │  D-LinOSS   │ │   Custom        │   │ │
│  │  │   Model     │ │   Model     │ │   Models        │   │ │
│  │  └─────────────┘ └─────────────┘ └─────────────────┘   │ │
│  └─────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                     Core Layer                             │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                Core Components                         │ │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────┐   │ │
│  │  │  LinOSS     │ │  D-LinOSS   │ │   Parallel      │   │ │
│  │  │   Layer     │ │   Layer     │ │    Scan         │   │ │
│  │  └─────────────┘ └─────────────┘ └─────────────────┘   │ │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────┐   │ │
│  │  │   Block     │ │ Activation  │ │ Visualization   │   │ │
│  │  │ Components  │ │ Functions   │ │   Utilities     │   │ │
│  │  └─────────────┘ └─────────────┘ └─────────────────┘   │ │
│  └─────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                   Backend Layer                            │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                 Burn Framework                         │ │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────┐   │ │
│  │  │   NdArray   │ │    WGPU     │ │    Candle       │   │ │
│  │  │    (CPU)    │ │   (GPU)     │ │   (Optional)    │   │ │
│  │  └─────────────┘ └─────────────┘ └─────────────────┘   │ │
│  └─────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                  Platform Layer                            │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────┐   │ │
│  │  │   Native    │ │    WASM     │ │   GPU Compute   │   │ │
│  │  │  (x86_64)   │ │   (Web)     │ │   (CUDA/ROCm)   │   │ │
│  │  └─────────────┘ └─────────────┘ └─────────────────┘   │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Module Structure

### Core Modules (`src/linoss/`)

```
src/linoss/
├── mod.rs                  # Module exports and public API
├── layer.rs                # LinOSS layer implementation
├── dlinoss_layer.rs        # D-LinOSS with learnable damping
├── model.rs                # High-level model containers
├── block.rs                # Reusable building blocks
├── parallel_scan.rs        # Parallel processing algorithms
├── activation.rs           # Activation functions
├── layers.rs               # Layer utilities
└── vis_utils.rs           # Visualization helpers
```

### Key Components

#### 1. LinOSS Layer (`layer.rs`)
- **Purpose**: Core linear oscillatory state-space implementation
- **Features**: LinOSS-IM implicit time integration
- **Dependencies**: Burn framework for tensor operations
- **API**: Single-step forward pass with hidden state management

#### 2. D-LinOSS Layer (`dlinoss_layer.rs`)
- **Purpose**: Damped LinOSS with learnable energy dissipation
- **Features**: Multi-timescale damping, configurable damping
- **Dependencies**: Extends LinOSS with additional parameters
- **API**: Sequence-level forward pass with automatic damping

#### 3. Model Container (`model.rs`)
- **Purpose**: High-level model orchestration
- **Features**: Multi-layer composition, training utilities
- **Dependencies**: Core layers and Burn modules
- **API**: Complete model training and inference

#### 4. Parallel Scan (`parallel_scan.rs`)
- **Purpose**: Efficient parallel processing of sequences
- **Features**: GPU-optimized prefix operations
- **Dependencies**: Backend-specific tensor operations
- **API**: Parallel and sequential scan implementations

## Data Flow Architecture

### Training Pipeline

```
Input Data
    ↓
┌─────────────────┐
│  Data Loading   │
│   & Batching    │
└─────────────────┘
    ↓
┌─────────────────┐
│  Preprocessing  │
│  & Validation   │
└─────────────────┘
    ↓
┌─────────────────┐
│   Model         │
│  Forward Pass   │
└─────────────────┘
    ↓
┌─────────────────┐
│  Loss           │
│  Computation    │
└─────────────────┘
    ↓
┌─────────────────┐
│  Gradient       │
│  Computation    │
└─────────────────┘
    ↓
┌─────────────────┐
│  Parameter      │
│    Update       │
└─────────────────┘
    ↓
Output/Metrics
```

### Inference Pipeline

```
Input Sequence
    ↓
┌─────────────────┐
│  Input          │
│  Validation     │
└─────────────────┘
    ↓
┌─────────────────┐
│  State          │
│ Initialization  │
└─────────────────┘
    ↓
┌─────────────────┐
│  Sequential/    │
│ Parallel Scan   │
└─────────────────┘
    ↓
┌─────────────────┐
│  Output         │
│ Projection      │
└─────────────────┘
    ↓
Predictions
```

## Backend Architecture

### Burn Framework Integration

```
┌─────────────────────────────────────┐
│            Application              │
├─────────────────────────────────────┤
│         LinOSS Modules              │
│  ┌─────────────────────────────────┐ │
│  │  Generic over Backend <B>       │ │
│  │  ┌─────────────────────────────┐ │ │
│  │  │    Tensor<B, D>             │ │ │
│  │  │    Module<B>                │ │ │
│  │  │    Param<Tensor<B, D>>      │ │ │
│  │  └─────────────────────────────┘ │ │
│  └─────────────────────────────────┘ │
├─────────────────────────────────────┤
│           Burn Core                 │
│  ┌─────────────────────────────────┐ │
│  │  Tensor Operations              │ │
│  │  Automatic Differentiation     │ │
│  │  Optimization                  │ │
│  └─────────────────────────────────┘ │
├─────────────────────────────────────┤
│        Backend Implementations     │
│  ┌─────────┐ ┌─────────┐ ┌────────┐ │
│  │ NdArray │ │  WGPU   │ │ Candle │ │
│  │  (CPU)  │ │ (GPU)   │ │ (Alt)  │ │
│  └─────────┘ └─────────┘ └────────┘ │
└─────────────────────────────────────┘
```

### Multi-Backend Support

| Backend | Platform | Use Case | Status |
|---------|----------|----------|--------|
| NdArray | CPU | Development, Testing | ✅ Working |
| WGPU | GPU | Training, Inference | ✅ Working |
| Candle | CPU/GPU | Alternative Backend | 📋 Planned |
| WASM | Web | Demo, Visualization | ✅ Custom Implementation |

## Memory Architecture

### Tensor Memory Layout

```
LinOSS Layer State:
┌─────────────────────────────────────────────────────────────┐
│                    Batch Dimension                         │
├─────────────────────────────────────────────────────────────┤
│  Sequence    │  Hidden State   │  Output                   │
│  Length      │  (Oscillators)  │  Projection               │
│              │                 │                           │
│  [T]         │  [2M] (x,ẋ)     │  [Q]                      │
│              │   pairs         │                           │
└─────────────────────────────────────────────────────────────┘

Memory Access Patterns:
- Sequential: O(T) time steps
- Parallel: O(log T) with parallel scan
- GPU: Coalesced memory access for tensor operations
```

### Parameter Memory

```
LinOSS Parameters:
├── A_diag: [M] - Diagonal oscillator frequencies
├── B_matrix: [M, P] - Input projection
├── C_matrix: [Q, M] - Output projection
├── D_matrix: [Q, P] - Direct feedthrough (optional)
└── bias_b: [M] - Input bias

D-LinOSS Additional Parameters:
├── damping_coeffs: [M/2, K] - Damping coefficients
└── damping_scales: [K] - Timescale parameters

Total Memory: O(M*(P+Q) + K*M/2)
```

## Configuration Architecture

### Hierarchical Configuration

```
Global Config
├── Model Config
│   ├── Layer Config
│   │   ├── Core Parameters
│   │   ├── Initialization
│   │   └── Behavior Flags
│   └── Training Config
│       ├── Optimizer Settings
│       ├── Learning Rate
│       └── Regularization
└── Backend Config
    ├── Device Selection
    ├── Memory Settings
    └── Precision Options
```

### Configuration Example

```rust
// D-LinOSS Configuration
let config = DLinossLayerConfig {
    d_input: 1,           // Input dimension
    d_model: 16,          // Hidden dimension (must be even)
    d_output: 1,          // Output dimension
    delta_t: 0.1,         // Time step
    init_std: 0.02,       // Parameter initialization
    enable_layer_norm: true,     // Layer normalization
    enable_damping: true,        // Enable D-LinOSS damping
    init_damping: 0.1,          // Initial damping coefficient
    num_damping_scales: 4,      // Multiple timescales
};
```

## Error Handling Architecture

### Error Propagation

```
Application Level
    ↓ Result<T, AppError>
Model Level
    ↓ Result<T, ModelError>  
Layer Level
    ↓ Result<T, LayerError>
Backend Level
    ↓ Result<T, BackendError>
System Level
```

### Error Types

```rust
pub enum LinossError {
    ConfigurationError(String),
    DimensionMismatch { expected: usize, got: usize },
    BackendError(String),
    NumericalInstability,
    InvalidInput(String),
}
```

## Testing Architecture

### Test Hierarchy

```
Integration Tests
├── End-to-End Model Testing
├── Multi-Backend Compatibility
└── Performance Benchmarks

Unit Tests
├── Layer-Level Testing
├── Component Testing
└── Utility Function Testing

Property Tests
├── Numerical Stability
├── Gradient Correctness
└── Invariant Checking
```

### Test Coverage

| Component | Unit Tests | Integration Tests | Benchmarks |
|-----------|------------|-------------------|------------|
| LinOSS Layer | ✅ | ✅ | ✅ |
| D-LinOSS Layer | ✅ | ✅ | ✅ |
| Parallel Scan | ✅ | ✅ | ✅ |
| Full Model | ✅ | ✅ | ✅ |
| Web Demo | Manual | ✅ | Manual |

## Performance Architecture

### Optimization Levels

1. **Algorithm Level**: Parallel scan, efficient discretization
2. **Implementation Level**: Memory layout, tensor operations
3. **Backend Level**: GPU kernels, SIMD instructions
4. **System Level**: Memory management, I/O optimization

### Performance Characteristics

| Operation | CPU (NdArray) | GPU (WGPU) | Web (WASM) |
|-----------|---------------|------------|------------|
| Training | ~15s (50 epochs) | ~5s (50 epochs) | N/A |
| Inference | ~10ms (seq=200) | ~2ms (seq=200) | ~50ms |
| Memory | ~100MB | ~500MB VRAM | ~10MB |

## Deployment Architecture

### Build Targets

```
cargo build                    # Native development
cargo build --release         # Optimized native
cargo build --target wasm32   # Web deployment
cargo build --features gpu    # GPU-optimized build
```

### Distribution

- **Native**: Cargo crate publication
- **Web**: WASM package + JavaScript bindings
- **Documentation**: Generated docs + examples
- **Research**: Reproducible benchmarks

## Security Considerations

### Input Validation
- Tensor dimension checking
- Numerical range validation
- Configuration parameter bounds

### Memory Safety
- Rust memory safety guarantees
- Safe tensor operations via Burn
- Bounds checking on all array accesses

### Computational Security
- Numerical stability checks
- Gradient clipping for training
- Resource usage monitoring

## Extensibility Points

### Adding New Layers
1. Implement the Layer trait
2. Add configuration struct
3. Include in module exports
4. Add tests and examples

### Adding New Backends
1. Implement Backend trait
2. Add backend-specific optimizations
3. Update build configuration
4. Validate compatibility

### Adding New Applications
1. Use existing high-level API
2. Add application-specific utilities
3. Create examples and documentation
4. Consider performance requirements

Last updated: June 12, 2025
