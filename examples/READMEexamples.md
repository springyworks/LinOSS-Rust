# Examples

This directory contains examples demonstrating the production D-LinOSS (Damped Linear Operator State Space) implementation.

## Production D-LinOSS Examples

### Core Demonstrations
- `burn_demo.rs` - Complete D-LinOSS vs vanilla LinOSS comparison with statistical analysis
- `dlinoss_comprehensive_showcase.rs` - Full showcase of all D-LinOSS production features
- `dlinoss_vs_vanilla_comparison.rs` - Detailed comparison between D-LinOSS and vanilla LinOSS

### Application Examples
- `dlinoss_time_series.rs` - Time series prediction with chaotic Lorenz dynamics
- `dlinoss_oscillator_demo.rs` - Oscillator damping effects demonstration

### Analysis Tools
- `dlinoss_visualizer.rs` - Brain dynamics visualization with D-LinOSS
- `dlinoss_response_visualizer.rs` - Interactive D-LinOSS layer response visualization with egui-plot

## Legacy Examples (OLD Directory)

All examples using the old LinOSS API have been moved to `OLD/` directory:
- Basic usage examples
- Legacy brain dynamics analyzers
- Chaotic system examples using old APIs
- Performance benchmarks with old implementation

## Key Features Demonstrated

### D-LinOSS Production Features
- **Learnable Damping**: Multiple timescale energy dissipation
- **IMEX Discretization**: Euler, Midpoint, and RK4 schemes
- **Activation Functions**: ReLU, GELU, and Tanh variants
- **Energy Control**: Controllable energy conservation/dissipation
- **Stability**: Long-term stable dynamics

### Comparison Capabilities
- Statistical analysis of damping effects
- Frequency response analysis
- Phase space trajectory analysis
- Energy dissipation quantification

## Running Examples

```bash
# Run the comprehensive showcase
cargo run --example dlinoss_comprehensive_showcase

# Compare D-LinOSS vs vanilla LinOSS
cargo run --example dlinoss_vs_vanilla_comparison

# Time series prediction demo
cargo run --example dlinoss_time_series

# Oscillator damping demonstration
cargo run --example dlinoss_oscillator_demo

# Interactive response visualizer with egui-plot
cargo run --example dlinoss_response_visualizer
```

## Notes

- All examples use the production `ProductionDLinossModel` and `ProductionDLinossConfig`
- Examples demonstrate real performance benefits of learnable damping
- Statistical outputs show quantified improvements in stability and control
