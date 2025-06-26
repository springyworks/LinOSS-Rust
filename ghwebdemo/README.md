# 🧠 LinOSS Web Demo

Interactive neural oscillator dynamics visualization running in the browser via WebAssembly.

## 🚀 Live Demo

Visit the live demo at: **[https://yourusername.github.io/LinossRust/](https://yourusername.github.io/LinossRust/)**

## ✨ Features

### 🎛️ Interactive Controls
- **Real-time parameter adjustment**: Frequency, damping, coupling, noise level
- **Simulation control**: Play, pause, reset functionality
- **Neural layer toggle**: Enable/disable GPU-accelerated D-LinOSS processing

### 📊 Advanced Visualization
- **3D Isometric View**: Real-time position visualization of coupled oscillators
- **Multi-channel Time Series**: Color-coded neural signals with live plotting
- **Performance Monitoring**: FPS counter and simulation metrics

### 🧠 Neural Architecture
- **D-LinOSS Implementation**: Diagonal Linear Oscillating State Space models
- **Coupled Dynamics**: Multi-oscillator interaction simulation
- **Biologically-inspired**: Neural signal modeling with realistic dynamics

## 🛠️ Technology Stack

- **🦀 Rust**: High-performance systems programming
- **🎨 egui**: Immediate mode GUI framework
- **⚡ WGPU**: Modern graphics API for GPU acceleration
- **🔥 Burn**: Machine learning framework for neural layers
- **🧠 D-LinOSS**: Custom neural oscillator architecture
- **🌐 WebAssembly**: Browser-compatible execution
- **📊 nalgebra**: Linear algebra and mathematical operations
- **📈 egui_plot**: Interactive plotting and visualization

## 🔧 Building Locally

### Prerequisites
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install wasm-pack
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# Add WASM target
rustup target add wasm32-unknown-unknown
```

### Build WASM
```bash
cd ghwebdemo
wasm-pack build --target web --out-dir wasm --out-name linoss_web_demo
```

### Serve Locally
```bash
# Simple HTTP server (Python)
python -m http.server 8000

# Or with Node.js
npx http-server -p 8000

# Visit http://localhost:8000
```

### Native Development
```bash
cd ghwebdemo
cargo run --bin linoss_web_visualizer
```

## 🎯 Usage

1. **Load the demo** - The WASM module loads automatically
2. **Adjust parameters** - Use the control panel to modify simulation
3. **Observe dynamics** - Watch real-time neural oscillator behavior
4. **Toggle views** - Switch between 3D and time series visualization
5. **Experiment** - Try different parameter combinations

### Key Parameters

- **Frequency (0.1-50 Hz)**: Base oscillation frequency
- **Damping (0-1)**: Energy dissipation rate
- **Coupling (0-0.5)**: Inter-oscillator interaction strength
- **Noise Level (0-0.1)**: Random perturbation amplitude

## 🧬 Scientific Background

D-LinOSS (Diagonal Linear Oscillating State Space) models represent a novel approach to neural signal modeling with applications in neural signal processing, brain dynamics simulation, and coupled oscillator networks.

## 📄 License

This project is part of the LinOSS neural dynamics framework.
