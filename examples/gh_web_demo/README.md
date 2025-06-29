# D-LinOSS Web Demo

This directory contains the WebAssembly demo for D-LinOSS neural dynamics that's deployed to GitHub Pages.

## 🚀 Quick Start

### Build and serve locally:
```bash
./build.sh debug    # Fast build for development
./serve.sh          # Start local server on port 8000
```

### Deploy to GitHub Pages:
```bash
./deploy.sh debug   # Build and prepare for deployment
# Files are automatically copied to the repository root
```

## 📁 Project Structure

```
gh_web_demo/
├── src/
│   └── lib.rs              # Main WASM library with egui 3-pane interface
├── build.sh                # Build script (debug/release/minimal/burn modes)
├── deploy.sh               # GitHub Pages deployment script
├── serve.sh                # Local development server
├── Cargo.toml              # Rust project configuration
└── README.md               # This file
```

## 🛠️ Build Modes

- **debug**: Fast build with full D-LinOSS (~2 seconds, ~5MB)
- **release**: Optimized build with full D-LinOSS (~30 seconds, ~2MB)
- **minimal**: Basic Burn tensors without D-LinOSS
- **burn**: Burn tensors + autodiff without D-LinOSS

## 🌐 Live Demo

The web demo is live at: https://springyworks.github.io/LinOSS-Rust/

## ✨ Features

- **3-Pane Interface**: 2D phase space, 2D time series, 3D trajectory
- **Real-time Visualization**: Interactive neural dynamics simulation
- **WebAssembly Performance**: Near-native speed in the browser
- **Burn Framework**: Tensor operations with NdArray backend

## 🧬 Architecture

- **Input**: 10-dimensional vectors
- **Hidden**: 32 neurons
- **Output**: 10-dimensional predictions
- **Backend**: Burn NdArray (CPU-optimized for WASM)
