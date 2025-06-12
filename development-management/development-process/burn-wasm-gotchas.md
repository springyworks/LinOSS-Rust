# Burn Framework & WASM Development Gotchas

*Extracted from LinOSS development experience - practical lessons learned*

## 🚨 TL;DR - Critical Issues

1. **Burn 0.17.1 + WASM = Incompatible** - dependency version conflicts make it impossible
2. **Always pin exact versions** - `burn = "0.17.1"` not `"0.17"`  
3. **Use `TensorData::new()` not `Data::new()`** - API changed
4. **WASM needs `getrandom = { version = "0.2", features = ["js"] }`**
5. **Generic Burn code needs extensive trait bounds**
6. **Check latest Burn examples, not docs** - API evolves fast

---

This document captures the key struggles, workarounds, and gotchas encountered while developing the LinOSS neural network implementation using the Burn framework and creating a WASM web demo.

## 🔥 Burn Framework Challenges

### 1. **Version Dependencies & API Evolution**
**Problem**: Burn is rapidly evolving, causing frequent breaking changes.
- Different versions have incompatible APIs (tensor creation, optimizer interfaces, etc.)
- Examples and documentation often lag behind latest API changes
- GitHub Copilot suggestions may use outdated API patterns

**Solutions**:
- Pin exact Burn versions in `Cargo.toml` (`burn = "0.17.1"` not `"0.17"`)  
- Always check latest Burn examples for current API patterns
- Use workspace inheritance to ensure consistent versions across sub-crates

### 2. **TensorData vs Data Confusion**
**Problem**: API changes from `Data::new()` to `TensorData::new()`
```rust
// OLD (doesn't work in newer versions)
let tensor = Tensor::from_data(Data::new(vec, shape), &device);

// NEW (Burn 0.17+)
let tensor = Tensor::from_data(TensorData::new(vec, shape), &device);
```

**Solution**: Always use `TensorData` for tensor creation and `.into_data()` for extraction.

### 3. **Complex Trait Bounds**
**Problem**: Generic backend functions require extensive trait bounds that aren't obvious.
```rust
// This WON'T compile
fn process<B: Backend>(tensor: Tensor<B, 2>) -> Tensor<B, 2> {
    tensor.add_scalar(1.0) // ERROR: trait bounds not satisfied
}

// This WILL compile  
fn process<B: Backend>(tensor: Tensor<B, 2>) -> Tensor<B, 2> 
where
    B::FloatElem: From<f32> + num_traits::Float,
{
    tensor.add_scalar(1.0)
}
```

**Solution**: Add comprehensive trait bounds or use concrete backends during development.

### 4. **Backend Selection Confusion**
**Problem**: Different backends (NdArray, WGPU) have different capabilities and requirements.
- WGPU backend requires GPU context and may not work in all environments
- Some operations are backend-specific
- Feature flags can conflict

**Solution**: Use type aliases and feature-gated compilation:
```rust
#[cfg(feature = "ndarray_backend")]
type MyBackend = burn::backend::ndarray::NdArray<f32>;

#[cfg(feature = "wgpu_backend")]  
type MyBackend = burn::backend::wgpu::Wgpu<f32, i32>;
```

### 5. **Optimizer API Changes**
**Problem**: Optimizer initialization and usage patterns change between versions.
```rust
// Burn 0.17.1 pattern
let optimizer_config = AdamConfig::new();
let mut optimizer = optimizer_config.init();
let grads = GradientsParams::from_grads(loss.backward(), &model);
model = optimizer.step(learning_rate, model, grads);
```

**Solution**: Follow the exact patterns from latest Burn examples, not older tutorials.

## 🌐 WASM Development Challenges

### 1. **Dependency Version Hell**
**Major Problem**: WASM has strict dependency requirements that conflict with Burn's dependencies.

**Specific Issues**:
- `getrandom` version conflicts: Burn 0.17.1 → rand 0.9.1 → getrandom 0.3.3 (no WASM support)
- `ndarray` version mismatches between Burn and direct usage
- Feature flag conflicts (`std` vs `no_std`)

**The Big Realization**: **Burn 0.17.1 is fundamentally incompatible with WASM** due to its dependency tree.

**Solution**: Create WASM-compatible alternative implementations:
```rust
// Instead of using Burn directly in WASM:
use burn::tensor::Tensor;  // Won't work!

// Use ndarray directly:
use ndarray::{Array1, Array2, Array3};
```

### 2. **getrandom Configuration Nightmare**
**Problem**: Random number generation in WASM requires specific configuration.
```toml
# This doesn't work in WASM:
rand = "0.9"

# This works:
rand = { version = "0.8", default-features = false, features = ["small_rng"] }
getrandom = { version = "0.2", features = ["js"] }
```

**Key Learning**: WASM random number generation must use JS-backed entropy.

### 3. **Size and Performance Constraints**
**Problem**: WASM bundles must be small and performant.
- Full Burn dependency tree creates huge WASM files (>5MB)
- Complex tensor operations are slow in WASM
- Memory allocation patterns matter more

**Solution**: Implement simplified versions for WASM:
```rust
// Instead of full Burn model
struct SimpleLinOSSModel {
    weights_in: Array2<f32>,    // Direct ndarray
    weights_out: Array2<f32>,   // Much smaller footprint
    // ... simplified parameters
}
```

### 4. **Import/Export Mismatches**
**Problem**: JavaScript expects specific class names and method signatures.
```rust
// Rust exports this:
#[wasm_bindgen]
pub struct LinOSSTrainer { ... }

// But JavaScript tries to import this:
const { LinOSSModel } = await import('./pkg/demo.js');  // ERROR!
```

**Solution**: Ensure exact name matching between Rust exports and JS imports.

### 5. **CORS and MIME Type Issues**
**Problem**: Browsers require correct MIME types for WASM files.
```python
# Simple Python server won't serve WASM correctly
# Need custom handler:
def guess_type(self, path):
    if path.endswith('.wasm'):
        return 'application/wasm', None
    return super().guess_type(path)
```

## 💡 Key Architectural Decisions

### 1. **Hybrid Approach for WASM**
**Decision**: Use main crate for configuration types, but implement WASM-specific math.
```rust
// Reuse configuration from main crate
use linoss_rust::linoss::block::LinossBlockConfig;

// But implement simplified math for WASM compatibility
struct SimpleLinOSSModel { ... }
```

### 2. **Workspace Inheritance**
**Decision**: Use Cargo workspace features to manage complex dependency trees.
```toml
[workspace]
members = ["examples/web_demo"]

[workspace.dependencies]
burn = { version = "0.17.1", default-features = false }
# ... shared dependencies
```

### 3. **Feature-Gated Backends**
**Decision**: Make backend selection explicit and feature-gated.
```rust
#[cfg(feature = "ndarray_backend")]
type MyBackend = NdArray<f32>;

#[cfg(feature = "wgpu_backend")]
type MyBackend = Wgpu<f32, i32>;
```

## 🚨 Critical Gotchas Summary

1. **WASM + Burn = Pain**: Burn 0.17.1 is not WASM-compatible due to dependency conflicts
2. **Pin Exact Versions**: Use exact version numbers, not ranges
3. **Check Recent Examples**: Burn API changes frequently; examples > documentation
4. **Trait Bounds Hell**: Generic Burn code needs extensive trait bounds
5. **getrandom.js**: WASM random numbers need `features = ["js"]`
6. **MIME Types Matter**: Custom server needed for WASM files
7. **Import/Export Names**: Exact name matching required between Rust and JS
8. **Size Matters**: Full Burn creates huge WASM bundles; simplify for web

## 🎯 Recommendations

### For Burn Development:
- Use concrete backends during prototyping, generics later
- Follow official examples religiously
- Test with multiple Burn versions before publishing
- Use workspace inheritance for complex projects

### For WASM + ML:
- Consider lighter alternatives to full ML frameworks
- Implement simplified versions specifically for WASM
- Use ndarray directly instead of heavy tensor libraries
- Budget for significant architecture differences vs native code

### For AI Assistant Usage:
- Explicitly mention Burn version in prompts
- Ask for trait bounds when using generics
- Verify API patterns against latest examples
- Be skeptical of complex dependency suggestions

---

**Bottom Line**: Combining cutting-edge ML frameworks with WASM is still bleeding edge. Expect to build bridges between ecosystems rather than using frameworks directly.

*Lessons learned from LinOSS development, June 2025*
