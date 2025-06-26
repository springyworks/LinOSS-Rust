# LinOSS 3D Neural Visualization Examples

This subdirectory contains high-quality 3D visualization applications for LinOSS neural oscillator dynamics using egui with OpenGL integration.

## Architecture

This is structured as a **sub-crate** within the main LinossRust project:
- It has its own `Cargo.toml` with egui dependencies  
- References the main `linoss_rust` library for core LinOSS functionality
- Contains two production-ready 3D visualization applications

## 🎯 Main Applications

### 🧠 LinOSS 3D Visualizer (`linoss_3d_visualizer`)

**Full-featured 3D neural oscillator visualization** with true OpenGL rendering:

- **True 3D OpenGL rendering** with perspective projection and depth perception
- **Real LinOSS/D-LinOSS integration** using the actual neural dynamics library
- **Interactive parameter controls** with live visual feedback
- **Mouse-based 3D camera** (drag to rotate, scroll to zoom)
- **Combined visualization** with 3D view + 2D signal plotting
- **Burn tensor backend** (NdArray) for efficient neural computation
- **Dynamic neural coloring** based on oscillator energy and activity
- **Scalable oscillator count** (8-128 neural oscillators)

```bash
cargo run --bin linoss_3d_visualizer
```

**Features:**
- LinOSS library integration with D-LinOSS damped dynamics
- Interactive parameter adjustment (alpha, beta, gamma, frequency, coupling)
- Real-time 3D neural network visualization
- Energy-based color coding (blue → green → red)
- Neural activity pulse effects
- Perspective depth effects

### 🌟 Simple LinOSS 3D (`simple_linoss_3d`)

**Lightweight demonstration** of 3D LinOSS visualization:

- **Isometric 3D projection** using egui_plot (no OpenGL complexity)
- **Real-time neural oscillator dynamics** 
- **Live parameter adjustment** with immediate visual updates
- **Dual-view plotting** (3D positions + signal time series)
- **Clean, maintainable code** perfect for learning and extension

```bash
cargo run --bin simple_linoss_3d
```

**Features:**
- Simple 3D-to-2D isometric projection
- 8 neural oscillators with phase relationships
- Real-time parameter controls (frequency, amplitude, phase shift)
- Live signal plotting with history
- Minimal dependencies
## 🚀 Getting Started

Both applications are ready to run:

```bash
# Full-featured 3D visualizer with OpenGL
cargo run --bin linoss_3d_visualizer

# Lightweight demo version
cargo run --bin simple_linoss_3d
```

## 🎮 Controls

**3D Camera Controls:**
- **Mouse drag**: Rotate 3D view (spherical coordinates)
- **Mouse scroll**: Zoom in/out
- **Reset button**: Return to default camera position

**Parameter Controls:**
- **Alpha/Beta/Gamma**: LinOSS oscillation strength parameters
- **Frequency**: Base oscillation frequency for all neural oscillators
- **Amplitude**: Output signal amplitude scaling
- **Coupling**: Inter-oscillator coupling strength
- **Oscillator Count**: Number of neural oscillators (8-128)

## 🔧 Technical Implementation

### LinOSS Integration
- Uses actual `DLinossLayer` from the main LinossRust crate
- Real D-LinOSS (Damped Linear Oscillatory State-Space) dynamics
- Burn backend (NdArray) for efficient tensor operations
- nalgebra for 3D mathematics and transformations

### OpenGL Rendering
- Custom vertex/fragment shaders for neural visualization
- True 3D perspective projection with depth testing
- Energy-based dynamic coloring (low → medium → high activity)
- Depth-aware point sizing and transparency effects
- Real-time neural activity pulse effects

### Performance
- 60 FPS target with real-time parameter updates
- GPU-accelerated rendering (OpenGL doesn't overload PCIe)
- Efficient memory management for large oscillator counts
- Optimized shader computations for smooth visualization

## 📊 Visualization Features

- **Neural Network Topology**: 3D spatial arrangement of oscillators
- **Real-time Dynamics**: Live neural oscillation visualization
- **Energy Color Coding**: Blue (low) → Green (medium) → Red (high activity)
- **Coupling Visualization**: Connected oscillator networks
- **Signal Plotting**: 2D time series alongside 3D view
- **Parameter Feedback**: Immediate visual response to control changes

## 🛠️ Dependencies

```toml
egui = "0.29"           # Immediate mode GUI
eframe = "0.29"         # Native application framework  
egui_plot = "0.29"      # Real-time plotting
nalgebra = "0.33"       # 3D mathematics
burn = "0.17"           # Neural computation backend
linoss_rust = { path = "../.." }  # Main LinOSS library
```

## 🎯 Use Cases

- **Research Visualization**: Explore LinOSS neural dynamics in 3D
- **Educational Demos**: Interactive learning of neural oscillators
- **Parameter Studies**: Real-time exploration of neural behavior
- **Algorithm Development**: Visual debugging of LinOSS implementations
- **Scientific Presentation**: High-quality neural visualization

## 🚀 Project Status

✅ **Fully Functional**: Both applications compile and run successfully  
✅ **Production Ready**: Clean, maintainable, and well-documented code  
✅ **LinOSS Integrated**: Uses actual neural dynamics from main library  
✅ **Interactive**: Real-time parameter control with visual feedback  
✅ **3D Capable**: True OpenGL rendering with perspective projection

This implementation provides a solid foundation for advanced LinOSS neural visualization and research applications! 🧠✨

The `egui_native` examples are designed to be:
- **Maintainable**: Clean separation from main library
- **Extensible**: Easy to add new examples
- **Educational**: Clear demonstration of D-LinOSS capabilities
- **Production-ready**: Performance optimized for real-time use

This complements the web-based demonstrations while providing full native capabilities for advanced neural dynamics exploration.
