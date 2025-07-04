# 📁 LinOSS Rust Examples

This directory contains examples demonstrating D-LinOSS and LinOSS implementations.

## 🚀 Working Examples

### ✅ Tested & Functional
- **`dlinoss_response_visualizer.rs`** - D-LinOSS response visualization
  - Status: ✅ COMPILES & RUNS
  - Run: `cargo run --example dlinoss_response_visualizer`
  
- **`burn_demo.rs`** - Basic Burn framework demonstration  
  - Status: ✅ COMPILES
  - Run: `cargo run --example burn_demo`

### 🎯 Interactive Applications
- **Main Application**: `../src/main.rs` - Interactive D-LinOSS Lissajous Visualizer
  - Status: ✅ WORKING (Enhanced with trail fading & damping controls)
  - Run: `cargo run --bin linoss_rust`
  - Features: Real-time visualization, 8 pulse types, damping controls

### 📁 Sub-Projects
- **`egui_native/`** - Native egui examples and GPU tests
  - Status: ✅ WORKING
  - Contains: Bidirectional GPU tests, shader examples

- **`dlinoss_diagram/`** - Architecture diagram generation
  - Status: ✅ WORKING  
  - Purpose: Generate D-LinOSS system diagrams

## 🗃️ Archived Examples

The `OLD/` directory contains examples that are currently broken or outdated:

### ❌ API Compatibility Issues
- `dlinoss_comprehensive_showcase.rs` - Uses outdated ProductionDLinossConfig API
- `dlinoss_oscillator_demo.rs` - Missing burn_wgpu dependency  
- `dlinoss_time_series.rs` - Uses deprecated burn::train module
- `dlinoss_vs_vanilla_comparison.rs` - API mismatch issues
- `dlinoss_simple_mnist.rs` - Missing WGPU backend

### 🐍 Python Scripts (Out of Scope)
- `analyze_brain_dynamics.py` - Python analysis script
- `velocity_monitor.py` - Python monitoring tool

### 📦 Legacy Code
- `brain_dynamics_explorer.rs` - Outdated brain dynamics implementation
- `burn_iris_loader.rs` - Legacy Iris dataset loader
- `dlinoss_visualizer.rs` - Superseded by main.rs Lissajous visualizer

## 🔧 Quick Test All Examples

```bash
# Test working examples
cargo check --example dlinoss_response_visualizer
cargo check --example burn_demo

# Run automated status checker
./scripts/check_project_status.sh

# Run main interactive application
cargo run --bin linoss_rust
```

## 📊 Current Status Summary

- ✅ **Working**: 2 examples + main app + 2 sub-projects = **5 functional items**
- 🗃️ **Archived**: 11 examples moved to OLD/ directory
- 🎯 **Focus**: Main Lissajous visualizer with D-LinOSS integration

## 🎯 Development Focus

**Primary Application**: `src/main.rs` - Enhanced D-LinOSS Lissajous Visualizer
- Interactive real-time visualization
- Trail fading controls to reduce visual clutter  
- D-LinOSS damping parameter adjustment
- 8 different pulse generation types
- Phase-shifted inputs for complex Lissajous patterns

This is the recommended starting point for exploring D-LinOSS capabilities.

## 📈 Future Plans

1. **Update archived examples** to current D-LinOSS API
2. **Create tutorial examples** for different D-LinOSS use cases  
3. **Add benchmark examples** for performance testing
4. **Implement example testing** in CI/CD pipeline

---

**Last Updated**: 2025-07-04  
**Status Tracking**: See `PROJECT_STATUS.md` in project root
