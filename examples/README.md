# LinossRust Examples

This directory contains runnable Rust examples demonstrating various features, models, and utilities in the LinossRust project.

## 🌊 Featured: NeuroBreeze - Advanced Brain Dynamics Visualization

**NEW**: Experience the beauty of neural dynamics with our enhanced 3×16 spinning Lissajous intercommunication visualization!

```bash
cargo run --example pure_dlinoss_brain_dynamics
```

**What makes this special:**
- **3×16 Signal Matrix**: Each neural region (🧠PFC, 🌐DMN, ⚙️THL) displays 16 individual intercommunication signals
- **Velocity-Driven Spinning Patterns**: High-velocity signals show as colorful emojis (🔴🟠🟡🟢🔵🟣⚫⚪), medium-velocity as traditional spinning symbols (◐◓◑◒)
- **Real-time Brain Dynamics**: Watch bidirectional neural communication evolve in real-time
- **dLinOSS Architecture**: Three interconnected blocks with different A-matrix parameterizations (ReLU, GELU, Squared)
- **Live Data Streaming**: Neural states stream as JSON to `/tmp/dlinoss_brain_pipe`

This represents the cutting edge of neural simulation visualization, combining mathematical rigor with intuitive, beautiful displays of complex brain dynamics.

---

## How to Run

Use Cargo to run an example:

```bash
cargo run --example <example_name>
```

## Example Index

- [`basic_usage.rs`](#basic_usagers): Minimal working example for LinOSS/Burn usage.
- [`burn_iris_loader.rs`](#burn_iris_loaderrs): Loads and batches the Iris dataset for Burn.
- [`chaotic_2d_linoss.rs`](#chaotic_2d_linossrs): Interactive TUI visualization of a chaotic 2D LinOSS system.
- [`compare_scan_methods.rs`](#compare_scan_methodsrs): Performance comparison of scan algorithms.
- [`damped_sine_response.rs`](#damped_sine_responsers): Training with TUI for damped sine response.
- [`dlinoss_comparison.rs`](#dlinoss_comparisonrs): Compare D-LinOSS variants.
- [`flyLinoss.rs`](#flylinossrs): Tensor visualization TUI.
- [`pure_dlinoss_brain_dynamics.rs`](#pure_dlinoss_brain_dynamicsrs): **🌊 NeuroBreeze** - Advanced 3×16 spinning Lissajous intercommunication visualization
- ... (add more as needed)

## Example Details

### basic_usage.rs
A minimal example showing how to instantiate and use a LinOSS model with Burn.

### burn_iris_loader.rs
Loads the processed Iris dataset and demonstrates batching for Burn.

### chaotic_2d_linoss.rs
Interactive TUI visualization of a chaotic 2D LinOSS system.

### compare_scan_methods.rs
Compares the performance of different scan algorithms implemented in LinOSS.

### damped_sine_response.rs
Interactive training demonstration for impulse response learning with **LinOSS/dLinOSS model toggle**.

**Features:**
- **Dual Model Architecture**: Switch between standard LinOSS and damped dLinOSS models
- **Interactive Training**: Real-time training visualization with live loss plotting
- **Impulse Response Task**: Learn to map impulse input → damped sine wave output
- **Model Comparison**: Compare LinOSS vs dLinOSS performance on the same task
- **Controls**: Press **[M]** to toggle models, **[Q]** to quit

**Technical Details:**
- **LinOSS Model**: 2-layer FullLinossModel (32D state, 16D hidden)
- **dLinOSS Model**: Single-layer with GELU A-parameterization and damping
- **Target Function**: `e^(-0.5*t) * sin(5*t)` - classic damped oscillator response
- **Training**: Adam optimizer with MSE loss, visualized in real-time TUI

### dlinoss_comparison.rs
Compares different D-LinOSS model variants on a benchmark task.

### flyLinoss.rs
Visualizes tensors and model states in a TUI.

### pure_dlinoss_brain_dynamics.rs 🌊
**NeuroBreeze v1.0** - The most advanced neural dynamics visualization in the LinossRust suite.

**Features:**
- **3 Neural Regions**: Prefrontal Cortex (🧠), Default Mode Network (🌐), Thalamus (⚙️) 
- **16 Signals per Region**: Each region outputs 16 distinct intercommunication signals
- **Spinning Lissajous Patterns**: Velocity-driven animation with colorful high-activity indicators
- **Bidirectional Connectivity**: Full 6×6 matrices connecting all dLinOSS blocks
- **Multiple A-Parameterizations**: Fast (ReLU), Medium (GELU), Slow (Squared) blocks
- **Real-time Data Export**: JSON streaming to FIFO pipe for external analysis
- **Interactive Controls**: Pause, damping toggle, coupling strength adjustment

**Technical Details:**
- Based on dLinOSS (Damped Linear Oscillatory State-Space) models
- Three-timescale architecture modeling different neural frequencies
- Ultra-low latency streaming with non-blocking I/O
- TUI-based visualization with activity bars, trajectory plots, and signal matrices

**Usage:**
```bash
cargo run --example pure_dlinoss_brain_dynamics

# Monitor live data stream:
cat /tmp/dlinoss_brain_pipe
```

**Controls:**
- `p`: Pause/unpause simulation
- `d`: Toggle damping on/off  
- `+/-`: Increase/decrease coupling strength
- `q`: Quit

This example showcases the full power of LinOSS models for complex neural system modeling with stunning real-time visualization.

---

For more details on each example, see the comments at the top of each file or run with `--help` if supported.

## Navigation

If your editor supports Markdown link navigation, you can jump to the section for a specific example by clicking the corresponding link in the index above.

If you right-click an example file and your editor supports it, you may be able to search for the section header (e.g., `### burn_iris_loader.rs`) in this README to jump directly to the relevant documentation.
