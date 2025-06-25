# LinossRust egui Native Examples

This subdirectory contains native GUI applications built with [egui](https://github.com/emilk/egui) for interactive exploration of D-LinOSS (Damped Linear Oscillatory State-Space) neural dynamics.

## Architecture

This is structured as a **sub-crate** within the main LinossRust project:
- It has its own `Cargo.toml` with egui dependencies
- References the main `linoss_rust` library for core functionality
- Contains multiple binary examples for different neural exploration scenarios

## Examples

### 🧠 D-LinOSS Explorer (`dlinoss_explorer`)

A comprehensive interactive application for exploring D-LinOSS neural dynamics with:

- **Real-time simulation** with adjustable parameters
- **Multiple input signal types**: Sine, Cosine, Square, Sawtooth, Noise, Impulse
- **Interactive parameter controls**: Network architecture, damping, frequency, time step
- **Advanced plotting**: Time series visualization with egui_plot
- **Live neural dynamics**: Watch oscillatory patterns evolve in real-time

#### Features

- 🎛️ **Parameter Controls**:
  - Network architecture (input/model/output dimensions)
  - D-LinOSS damping coefficients
  - Input signal generation (type, amplitude, phase, frequency)
  - Simulation time step and update rate

- 📊 **Visualization**:
  - Real-time time series plots with multiple neuron traces
  - Color-coded individual neuron outputs
  - Scrolling history with configurable buffer size

- 🔄 **Interactive Simulation**:
  - Play/pause simulation controls
  - Live parameter adjustment while running
  - Automatic network reinitialization on parameter changes

#### Usage

```bash
# From the LinossRust root directory
cargo run --manifest-path examples/egui_native/Cargo.toml --bin dlinoss_explorer
```

Or from the `examples/egui_native` directory:
```bash
cargo run --bin dlinoss_explorer
```

## Technical Details

### Dependencies

- **eframe**: Native egui application framework
- **egui**: Immediate mode GUI library
- **egui_plot**: Real-time plotting capabilities
- **egui_extras**: Additional UI components (image, SVG support)
- **burn**: Deep learning framework for neural networks
- **nalgebra**: Linear algebra for mathematical operations

### Integration with LinossRust

The examples directly import and use:
- `DLinossLayer` and `DLinossLayerConfig` from the main library
- `LinossLayer` for comparison studies
- Burn backend integration (NdArray backend)

### Future Extensions

Planned additional examples:
- `burn_neural_playground`: Interactive neural network design
- `realtime_brain_monitor`: Live EEG/brain signal processing
- `comparative_analysis`: Side-by-side LinOSS vs D-LinOSS comparison

## Building and Running

The sub-crate builds independently but requires the main LinossRust library:

```bash
# Build all examples
cargo build

# Run specific example
cargo run --bin dlinoss_explorer

# Build in release mode for better performance
cargo build --release
cargo run --release --bin dlinoss_explorer
```

## Development Notes

This native egui approach was chosen over WASM after encountering build complexities with web deployment. The native approach provides:

- Full access to Burn's capabilities
- Better performance for real-time neural dynamics
- Easier integration with system resources
- More straightforward development and debugging

For web deployment, consider using the separate `web_demo` with simpler, WASM-compatible implementations.

## Integration with Main Project

The `egui_native` examples are designed to be:
- **Maintainable**: Clean separation from main library
- **Extensible**: Easy to add new examples
- **Educational**: Clear demonstration of D-LinOSS capabilities
- **Production-ready**: Performance optimized for real-time use

This complements the web-based demonstrations while providing full native capabilities for advanced neural dynamics exploration.
