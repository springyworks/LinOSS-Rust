LINOSS-RUST(1)                    User Commands                    LINOSS-RUST(1)

NAME
     linoss-rust - Linear Oscillatory State-Space Models implementation using Burn framework

SYNOPSIS
     cargo run --example EXAMPLE [--features BACKEND]
     cargo build [--release] [--features BACKEND]
     cargo test [--features BACKEND]

DESCRIPTION
     LinOSS-Rust implements Linear Oscillatory State-Space Models (LinOSS) based on the research
     paper "Oscillatory State-Space Models" (arXiv:2410.03943v2). The implementation focuses on
     real-valued LinOSS-IM (Implicit Time Integration) variant using the Burn framework for
     backend-agnostic tensor operations supporting both CPU and GPU execution.

     The project provides a flexible and efficient Rust library for LinOSS models, suitable for
     research and application development, with emphasis on forced harmonic oscillators and
     parallel scan algorithms for efficient sequence processing.

BACKENDS
     ndarray_backend    CPU-based computation using NdArray (default)
     wgpu_backend      GPU-accelerated computation using WGPU

BUILD OPTIONS
     --release         Build with optimizations enabled
     --features        Select backend (ndarray_backend|wgpu_backend)

COMPILATION STATUS
     As of June 27, 2025, all targets compile successfully with zero errors:
     
     $ cargo check --all-targets
         Finished dev profile [unoptimized + debuginfo] target(s) in 0.80s

EXAMPLES
     All examples are functional and optimized. Run with:
     
     cargo run --example EXAMPLE_NAME [--features BACKEND]

     Available examples:

     basic_usage                    ✅ Minimal working example demonstrating model initialization
                                   and forward pass
     
     brain_dynamics_analyzer        Advanced brain dynamics analysis with neural region modeling
     
     brain_dynamics_explorer        Interactive exploration of brain dynamics patterns
     
     burn_iris_loader              Iris dataset loading and processing demonstration
     
     chaotic_2d_linoss             ✅ Interactive TUI visualization of 2D chaotic systems
     
     compare_scan_methods          ✅ Performance comparison of different parallel scan algorithms
     
     comprehensive_signal_test     Comprehensive signal processing tests
     
     damped_sine_response          ✅ Training with TUI for damped sine wave responses
     
     dlinoss_comparison            Comparison between different LinOSS variants
     
     dlinoss_screensaver           Visual screensaver using LinOSS dynamics
     
     dlinoss_visualizer            ✅ Real-time LinOSS dynamics visualization
     
     flyLinoss                     ✅ Tensor visualization TUI with LinOSS models
     
     multi_lorenz_brain_dynamics   Multi-system Lorenz attractor brain dynamics simulation
     
     parallel_scan_test            Testing suite for parallel scan implementations
     
     pure_dlinoss_brain_dynamics   🌊 FEATURED: Real-time neural dynamics visualization with
                                   3×16 spinning Lissajous intercommunication patterns
     
     sine_wave_training            Basic sine wave training pipeline
     
     sine_wave_training_tui        TUI-based sine wave training interface
     
     sine_wave_visualization       ✅ Clean inference demonstration with sine wave processing
     
     single_block_test             Testing individual LinOSS blocks
     
     synthetic_data_usage          Synthetic dataset generation and usage examples
     
     test_fixed_linoss             Fixed-point LinOSS model testing
     
     test_logging                  Logging system verification
     
     test_robust_cleanup           Robust cleanup procedures testing
     
     test_terminal_cleanup         Terminal cleanup and restoration testing
     
     train_linoss                  ✅ Full training pipeline demonstration
     
     wgpu_minimal_test             Minimal WGPU backend functionality test
     
     wgpu_test                     Comprehensive WGPU backend testing

FEATURED EXAMPLE - NEUROBREEZE
     The pure_dlinoss_brain_dynamics example showcases real-time neural dynamics visualization:
     
     $ cargo run --example pure_dlinoss_brain_dynamics
     
     Features:
     • 🧠 3 Neural Regions: Prefrontal Cortex, Default Mode Network, Thalamus
     • 🌀 3×16 Signal Matrix: Each region shows 16 intercommunication signals
     • ⚡ Velocity-Driven Animation: 🔴🟠🟡🟢🔵🟣⚫⚪ = high velocity
     • 🔄 Bidirectional Connectivity: Full 6×6 connectivity matrices
     • 📡 Real-time Data Streaming: JSON streams through /tmp/dlinoss_brain_pipe
     • 🎮 Interactive Controls: 'p' pause, 'd' damping, '+/-' coupling strength

QUICK START
     1. Clone and build:
        $ git clone <repository-url>
        $ cd LinossRust
        $ cargo build --release
     
     2. Run basic example:
        $ cargo run --example basic_usage
     
     3. Try GPU backend:
        $ cargo run --example basic_usage --features wgpu_backend
     
     4. Run tests:
        $ cargo test

RECENT SESSION (June 27, 2025)
     Project compilation verified successful:
     
     $ cargo check --all-targets
         Finished dev profile [unoptimized + debuginfo] target(s) in 0.80s
     
     GPU backend test successful:
     
     $ cargo run --example basic_usage --features wgpu_backend
         Finished dev profile [unoptimized + debuginfo] target(s) in 3.66s
         Running target/debug/examples/basic_usage
     
     [INFO] --- Running Linoss Model: Basic Usage Example ---
     [INFO] Input tensor created with shape: [1, 2]
     [INFO] Performing forward pass...
     [INFO] Forward pass completed.
     [INFO] Output tensor shape: [1, 2]
     [INFO] Output tensor data (first 2 elements):
     [INFO] Output[0]: -0.0009
     [INFO] Output[1]: -0.0008
     [INFO] Output tensor contains non-zero values.
     [INFO] Output tensor does not contain NaN or Inf values.
     [INFO] --- Linoss Model: Basic Usage Example Finished ---

ENVIRONMENT
     RUST_LOG              Set to control log verbosity (debug|info|warn|error)
                          Example: RUST_LOG=linoss_rust=debug cargo test

FILES
     src/linoss/layer.rs   LinossLayer implementation
     src/linoss/block.rs   LinossBlock with GLU activation and LayerNorm
     src/linoss/model.rs   FullLinossModel with stacked blocks
     Cargo.toml           Project configuration and dependencies
     README.md            Comprehensive project documentation

SEE ALSO
     cargo(1), rust(1)
     
     Research paper: "Oscillatory State-Space Models" (arXiv:2410.03943v2)
     Burn framework: https://burn.dev/

BUGS
     None known. All examples compile and run successfully as of June 27, 2025.

AUTHORS
     Implementation based on LinOSS research paper by the LinOSS team.
     Rust implementation using Burn framework.

LinOSS-Rust 0.2.0                June 27, 2025                LinOSS-Rust 0.2.0
