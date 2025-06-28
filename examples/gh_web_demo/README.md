# D-LinOSS Web Demo

A web-based demonstration of Damped Linear Oscillatory State Space (D-LinOSS) neural dynamics using WASM, egui, and the Burn ML framework.

## Features

- **Real-time 3D visualization** with interactive rotation
- **Phase space plots** showing dynamic neural state evolution  
- **Signal timeline** displaying oscillatory patterns over time
- **Interactive controls** for frequency, damping, and coupling parameters
- **Multiple build configurations** supporting different levels of complexity

## Build Options

### Full D-LinOSS Neural Network
```bash
wasm-pack build --target web --debug --features linoss
```
Uses the genuine D-LinOSS layer with learnable damping, oscillatory dynamics, and neural network transformations.

### Burn Tensor Demo
```bash
wasm-pack build --target web --debug --features burn
```
Simplified version using Burn tensor operations for mathematical oscillator simulation.

### Basic Mock Demo
```bash
wasm-pack build --target web --debug
```
Lightweight version with mathematical oscillator (no ML dependencies).

## Running the Demo

1. Build with your preferred configuration (see above)
2. Start a web server:
   ```bash
   python3 -m http.server 8000
   ```
3. Open http://localhost:8000 in your browser

## Architecture

- **Frontend**: egui with WebGL rendering via 'glow' backend
- **Neural Engine**: D-LinOSS layer with Burn ML framework
- **Visualization**: Custom 3D projection with wireframe rendering
- **WASM Target**: Optimized for browser execution with minimal dependencies

## Interactive Elements

- **Frequency slider**: Controls oscillation speed (0.1-10 Hz)
- **Damping slider**: Adjusts energy dissipation (0-2.0)
- **Coupling slider**: Modifies phase space coupling (0-2.0)
- **Start/Pause**: Control simulation execution
- **Reset**: Clear trajectories and restart
- **3D rotation**: Drag to rotate the 3D phase trajectory view

## Neural Dynamics

When built with `--features linoss`, the demo showcases:

- **Learnable damping** adapted from neural network parameters
- **Oscillatory state dynamics** with complex phase relationships  
- **Real-time neural processing** of 2D input coordinates
- **Emergent trajectory patterns** from learned neural dynamics

The D-LinOSS layer processes input coordinates through:
1. 3D tensor reshaping [batch, sequence, features]
2. Neural forward pass with oscillatory transformations
3. Output projection back to 2D phase space coordinates
4. Real-time visualization of the resulting dynamics

## Status Indicators

The bottom status bar shows:
- ✅ **💚 Real D-LinOSS Layer** - Using genuine neural network
- ✅ **🔧 Burn Tensor Demo** - Using tensor mathematics  
- ✅ **📝 Mock Demo** - Using basic simulation

## Technical Notes

- Built following WASM compatibility guidelines from `.github/copilot-instructions.md`
- Uses conditional compilation for feature-based builds
- Optimized tensor operations with explicit dimension annotations
- Cross-platform compatibility (Linux/Windows/macOS browsers)
