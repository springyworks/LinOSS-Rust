# LinOSS Rust Development Standards

## Code Style and Formatting

### Rust Conventions

Follow standard Rust conventions and idioms:

```rust
// Good: Snake case for functions and variables
fn compute_oscillator_state() -> f32 { }
let damping_coefficient = 0.1;

// Good: Pascal case for types
struct DLinossLayer<B: Backend> { }
enum ActivationFunction { }

// Good: SCREAMING_SNAKE_CASE for constants
const DEFAULT_DELTA_T: f32 = 0.1;
const MAX_OSCILLATORS: usize = 1024;
```

### Formatting

Use `rustfmt` for consistent formatting:

```bash
# Format all code
cargo fmt

# Check formatting without applying
cargo fmt -- --check
```

Configuration in `.rustfmt.toml`:
```toml
max_width = 100
hard_tabs = false
tab_spaces = 4
newline_style = "Unix"
use_small_heuristics = "Default"
```

### Linting

Use `clippy` for code quality:

```bash
# Run clippy
cargo clippy

# Clippy with all features
cargo clippy --all-features

# Deny warnings in CI
cargo clippy -- -D warnings
```

## Documentation Standards

### Public API Documentation

All public items must have documentation:

```rust
/// Configuration for D-LinOSS layer with learnable damping.
/// 
/// # Examples
/// 
/// ```rust
/// use linoss_rust::linoss::DLinossLayerConfig;
/// 
/// let config = DLinossLayerConfig::new_dlinoss(1, 16, 1);
/// assert_eq!(config.d_input, 1);
/// assert_eq!(config.enable_damping, true);
/// ```
#[derive(Config, Debug)]
pub struct DLinossLayerConfig {
    /// Input dimension for the layer
    pub d_input: usize,
    
    /// Model dimension - must be even for oscillatory pairs
    /// 
    /// Each pair represents [position, velocity] of an oscillator.
    /// Total number of oscillators = d_model / 2
    pub d_model: usize,
    
    // ... more fields
}
```

### Implementation Documentation

Document complex algorithms and design decisions:

```rust
impl<B: Backend> DLinossLayer<B> {
    /// Apply learnable damping to the state (D-LinOSS key operation).
    /// 
    /// This implements the energy dissipation mechanism described in
    /// "Learning to Dissipate Energy in Oscillatory State-Space Models"
    /// (Boyer, Rusch, Rus, arXiv:2505.12171, 2025).
    /// 
    /// # Algorithm
    /// 
    /// For each oscillator pair [x, ẋ]:
    /// 1. Extract velocity component ẋ
    /// 2. Apply exponential damping: ẋ_new = ẋ * exp(-γ * Δt)
    /// 3. Use learned damping coefficient γ
    fn apply_learnable_damping(&self, state: Tensor<B, 3>) -> Tensor<B, 3> {
        // Implementation...
    }
}
```

### Module Documentation

Each module should have a clear description:

```rust
//! # D-LinOSS (Damped Linear Oscillatory State-Space) Implementation
//! 
//! This module implements the D-LinOSS layer as described in:
//! "Learning to Dissipate Energy in Oscillatory State-Space Models"
//! Boyer, Rusch, Rus (arXiv:2505.12171, 2025)
//! 
//! ## Key Features
//! 
//! - Learnable damping coefficients for energy dissipation
//! - Multiple timescale damping mechanisms  
//! - Backward compatibility with vanilla LinOSS
//! 
//! ## Usage
//! 
//! ```rust
//! use linoss_rust::linoss::{DLinossLayer, DLinossLayerConfig};
//! 
//! let config = DLinossLayerConfig::new_dlinoss(1, 16, 1);
//! let layer = DLinossLayer::new(&config, &device);
//! ```

use burn::prelude::*;
// ... rest of module
```

## Testing Standards

### Test Organization

```
tests/
├── unit/                   # Unit tests for individual components
│   ├── layer_tests.rs     # Test individual layers
│   ├── model_tests.rs     # Test model compositions
│   └── utils_tests.rs     # Test utility functions
├── integration/           # Integration tests
│   ├── end_to_end.rs     # Full pipeline tests
│   └── backend_compat.rs  # Multi-backend tests
└── benchmarks/            # Performance benchmarks
    ├── layer_bench.rs     # Layer performance
    └── memory_bench.rs    # Memory usage tests
```

### Unit Test Standards

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    
    type TestBackend = NdArray<f32>;
    
    #[test]
    fn test_dlinoss_layer_creation() {
        let device = Default::default();
        let config = DLinossLayerConfig::new_dlinoss(2, 8, 1);
        
        let layer = DLinossLayer::<TestBackend>::new(&config, &device);
        
        assert_eq!(layer.has_damping(), true);
        assert!(layer.get_damping_coefficients().is_some());
    }
    
    #[test]
    fn test_forward_pass_shapes() {
        let device = Default::default();
        let config = DLinossLayerConfig::new_dlinoss(1, 4, 2);
        let layer = DLinossLayer::<TestBackend>::new(&config, &device);
        
        let batch_size = 3;
        let seq_len = 10;
        let input = Tensor::zeros([batch_size, seq_len, 1], &device);
        
        let output = layer.forward(input);
        
        assert_eq!(output.dims(), [batch_size, seq_len, 2]);
    }
    
    #[test]
    #[should_panic(expected = "d_model must be even")]
    fn test_odd_d_model_panics() {
        let device = Default::default();
        let config = DLinossLayerConfig {
            d_model: 5, // Odd number should panic
            // ... other fields
        };
        
        DLinossLayer::<TestBackend>::new(&config, &device);
    }
}
```

### Property-Based Testing

For numerical code, use property-based tests:

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_damping_stability(
        damping_coeff in 0.0f32..1.0,
        delta_t in 0.01f32..0.5,
    ) {
        let device = Default::default();
        let mut config = DLinossLayerConfig::new_dlinoss(1, 4, 1);
        config.init_damping = damping_coeff as f64;
        config.delta_t = delta_t as f64;
        
        let layer = DLinossLayer::<TestBackend>::new(&config, &device);
        let input = Tensor::ones([1, 10, 1], &device);
        
        let output = layer.forward(input);
        
        // Output should be finite
        prop_assert!(output.to_data().iter().all(|&x| x.is_finite()));
    }
}
```

### Benchmark Standards

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_dlinoss_forward(c: &mut Criterion) {
    let device = Default::default();
    let config = DLinossLayerConfig::new_dlinoss(1, 64, 1);
    let layer = DLinossLayer::<NdArray<f32>>::new(&config, &device);
    let input = Tensor::zeros([8, 100, 1], &device);
    
    c.bench_function("dlinoss_forward_8x100x1", |b| {
        b.iter(|| {
            let output = layer.forward(black_box(input.clone()));
            black_box(output)
        })
    });
}

criterion_group!(benches, bench_dlinoss_forward);
criterion_main!(benches);
```

## Error Handling

### Error Types

Define clear error types:

```rust
#[derive(Debug, thiserror::Error)]
pub enum LinossError {
    #[error("Configuration error: {0}")]
    Configuration(String),
    
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    
    #[error("Invalid parameter: {parameter} = {value} (must be {constraint})")]
    InvalidParameter {
        parameter: String,
        value: String,
        constraint: String,
    },
    
    #[error("Numerical instability detected")]
    NumericalInstability,
    
    #[error("Backend error: {0}")]
    Backend(String),
}
```

### Error Handling Patterns

```rust
impl<B: Backend> DLinossLayer<B> {
    pub fn new(config: &DLinossLayerConfig, device: &B::Device) -> Result<Self, LinossError> {
        // Validate configuration
        if config.d_model % 2 != 0 {
            return Err(LinossError::InvalidParameter {
                parameter: "d_model".to_string(),
                value: config.d_model.to_string(),
                constraint: "even number (for oscillatory pairs)".to_string(),
            });
        }
        
        if config.delta_t <= 0.0 {
            return Err(LinossError::InvalidParameter {
                parameter: "delta_t".to_string(),
                value: config.delta_t.to_string(),
                constraint: "positive".to_string(),
            });
        }
        
        // Create layer...
        Ok(layer)
    }
}
```

## Performance Standards

### Memory Management

```rust
impl<B: Backend> DLinossLayer<B> {
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, seq_len, _] = input.dims();
        
        // Pre-allocate output tensor to avoid reallocations
        let mut output = Tensor::zeros([batch_size, seq_len, self.d_output], &input.device());
        
        // Reuse tensors when possible
        let mut hidden_state = Tensor::zeros([batch_size, self.d_model], &input.device());
        
        for t in 0..seq_len {
            // Process step without creating unnecessary intermediates
            hidden_state = self.step_function(input.clone(), hidden_state);
        }
        
        output
    }
}
```

### GPU Optimization

```rust
impl<B: Backend> DLinossLayer<B> {
    /// GPU-optimized parallel processing when available
    fn forward_parallel(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        // Use parallel scan for GPU backends
        if B::supports_parallel_scan() {
            self.parallel_scan_forward(input)
        } else {
            self.sequential_forward(input)
        }
    }
}
```

## Git Workflow

### Branch Naming

```
feature/dlinoss-implementation
bugfix/tensor-dimension-error
refactor/layer-organization
docs/api-reference-update
```

### Commit Messages

Follow conventional commits:

```
feat: implement D-LinOSS layer with learnable damping

- Add DLinossLayer with configurable damping coefficients
- Implement multiple timescale damping mechanisms  
- Add comprehensive tests and examples
- Update documentation with usage examples

Closes #42
```

```
fix: resolve tensor dimension mismatch in forward pass

The forward pass was incorrectly handling batch dimensions when
sequence length was 1. This fix ensures proper tensor reshaping
for all input shapes.

Fixes #38
```

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature  
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Benchmarks show no regression
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings introduced
```

## Code Review Standards

### Review Checklist

**Functionality:**
- [ ] Code solves the intended problem
- [ ] Edge cases are handled
- [ ] Error conditions are addressed

**Quality:**
- [ ] Code is readable and well-structured
- [ ] No unnecessary complexity
- [ ] Performance considerations addressed

**Testing:**
- [ ] Adequate test coverage
- [ ] Tests are meaningful and reliable
- [ ] Benchmarks updated if needed

**Documentation:**
- [ ] Public APIs documented
- [ ] Complex algorithms explained
- [ ] Examples provided where helpful

### Review Comments

```rust
// Good: Constructive feedback
// Consider using `iter().fold()` here for better readability:
// let sum = values.iter().fold(0.0, |acc, &x| acc + x);

// Good: Questions for clarification  
// Why do we need to clone here? Can we use a reference instead?

// Good: Performance suggestions
// This could be expensive for large tensors. Consider using in-place operations.
```

## Release Process

### Version Numbering

Follow semantic versioning (SemVer):

- **Major** (1.0.0): Breaking API changes
- **Minor** (0.1.0): New features, backward compatible
- **Patch** (0.0.1): Bug fixes, backward compatible

### Release Checklist

1. **Pre-release:**
   - [ ] All tests pass
   - [ ] Documentation updated
   - [ ] Changelog updated
   - [ ] Version bumped in `Cargo.toml`

2. **Release:**
   - [ ] Tag created: `git tag v0.2.0`
   - [ ] Crate published: `cargo publish`
   - [ ] GitHub release created
   - [ ] Documentation deployed

3. **Post-release:**
   - [ ] Announce on relevant channels
   - [ ] Update examples and tutorials
   - [ ] Monitor for issues

## Continuous Integration

### GitHub Actions

```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
          
      - name: Run tests
        run: cargo test --all-features
        
      - name: Check formatting
        run: cargo fmt -- --check
        
      - name: Run clippy
        run: cargo clippy -- -D warnings
        
      - name: Build examples
        run: cargo build --examples
```

Last updated: June 12, 2025
