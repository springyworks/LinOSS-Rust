# LinOSS Rust Implementation (Burn Framework)

## 🎯 **Quick Reference** - Project Management Commands

Never forget these essential commands for project status tracking:

```bash
# 📊 Daily status check (MOST IMPORTANT)
./scripts/check_project_status.sh

# 🔍 Check specific file status  
git check-attr status src/main.rs
git check-attr status examples/some_example.rs

# 📋 Find all working files
git ls-files | xargs git check-attr status | grep 'working'

# ⚠️ Find files needing attention
git ls-files | xargs git check-attr status | grep -E 'broken|needs-test'

# 🎛️ View all management tools
./scripts/git_status_manager.sh

# 🗃️ Archive broken code
mv examples/broken.rs examples/OLD/

# ✅ Mark file as working
echo 'examples/fixed.rs status=working' >> .gitattributes
```

**💡 REMEMBER**: Always run `./scripts/check_project_status.sh` before commits!

---

<details>
<summary><b>Table of Contents</b> (click to expand) | Sections: 1. Project Overview | 2. Project Management | 3. Install & Run | 4. Abstract | 5. Core Concepts | 6. Rust & Burn | 7. Structure | 8. Usage | 9. Status</summary>

- [LinOSS Rust Implementation (Burn Framework)](#linoss-rust-implementation-burn-framework)
  - [🎯 **Quick Reference** - Project Management Commands](#-quick-reference---project-management-commands)
  - [1. Project Overview](#1-project-overview)
  - [2. 🎯 Project Management \& Status Tracking](#2--project-management--status-tracking)
    - [2.1. Quick Status Check](#21-quick-status-check)
    - [2.2. File Status Categories](#22-file-status-categories)
    - [2.3. Daily Workflow Commands](#23-daily-workflow-commands)
    - [2.4. Project Status Overview](#24-project-status-overview)
  - [3. Install \& Run](#3-install--run)
    - [3.1. Prerequisites](#31-prerequisites)
    - [3.2. Cloning](#32-cloning)
    - [3.3. Building](#33-building)
    - [3.4. Running Tests](#34-running-tests)
    - [3.5. Running Examples](#35-running-examples)
    - [3.6. Backend Selection](#36-backend-selection)
    - [3.7. Enabling Log Output (Optional)](#37-enabling-log-output-optional)
  - [4. Abstract Summary](#4-abstract-summary)
  - [5. Core Concepts for Implementation](#5-core-concepts-for-implementation)
    - [5.a. State-Space Model (SSM) Fundamentals](#5a-state-space-model-ssm-fundamentals)
    - [5.b. Forced Harmonic Oscillators](#5b-forced-harmonic-oscillators)
    - [5.c. Discretization Methods](#5c-discretization-methods)
      - [5.c.i. LinOSS-IM (Implicit Time Integration)](#5ci-linoss-im-implicit-time-integration)
      - [5.c.ii. LinOSS-IMEX (Implicit-Explicit Time Integration)](#5cii-linoss-imex-implicit-explicit-time-integration)
    - [5.d. Fast Recurrence via Associative Parallel Scans](#5d-fast-recurrence-via-associative-parallel-scans)
  - [6. Rust and Burn Crate Considerations](#6-rust-and-burn-crate-considerations)
    - [6.a. Mapping to Burn Tensors](#6a-mapping-to-burn-tensors)
    - [6.b. Burn Modules](#6b-burn-modules)
    - [6.c. Efficient Computations](#6c-efficient-computations)
    - [6.d. Parameterization and Initialization](#6d-parameterization-and-initialization)
  - [7. Project Structure](#7-project-structure)
  - [7. Usage](#7-usage)
    - [7.1. Building the Project](#71-building-the-project)
    - [7.2. Running Tests](#72-running-tests)
    - [7.3. Running Examples](#73-running-examples)
    - [7.4. Backend Selection](#74-backend-selection)
  - [8. Development Status (Current Version)](#8-development-status-current-version)
    - [8.a. Implemented Features ✅ **ALL WORKING**](#8a-implemented-features--all-working)
    - [8.b. Key Characteristics](#8b-key-characteristics)
    - [8.c. Recent Fixes (June 2025)](#8c-recent-fixes-june-2025)
    - [8.d. Future Enhancements](#8d-future-enhancements)
  - [🌊 Featured: NeuroBreeze - Brain Dynamics Visualization](#-featured-neurobreeze---brain-dynamics-visualization)
    - [Key Features:](#key-features)

</details>


## 1. Project Overview

This project is a Rust implementation of Linear Oscillatory State-Space Models (LinOSS) using the Burn framework. It is based on the research paper "Oscillatory State-Space Models" ([arXiv:2410.03943v2](https://arxiv.org/html/2410.03943v2#bib)). The current implementation focuses on the real-valued LinOSS-IM (Implicit Time Integration) variant.

The primary goal is to provide a flexible and efficient Rust library for LinOSS models, suitable for research and application, leveraging Burn's backend-agnostic tensor operations for CPU and GPU execution.

## 2. 🎯 Project Management & Status Tracking

**NEW FEATURE**: Comprehensive Git-based project management system for tracking file status and maintaining code quality.

### 2.1. Quick Status Check
```bash
# Check all examples and get status report
./scripts/check_project_status.sh

# View comprehensive Git management tools
./scripts/git_status_manager.sh

# Check specific file status
git check-attr status src/main.rs
git check-attr status examples/some_example.rs
```

### 2.2. File Status Categories
- ✅ **`status=working`** - Tested, compiles, and runs correctly
- ❌ **`status=broken`** - Compilation or runtime failures  
- 🔍 **`status=needs-test`** - Untested, needs verification
- 📦 **`status=stale`** - Outdated, consider archiving

### 2.3. Daily Workflow Commands
```bash
# Morning status check
./scripts/check_project_status.sh

# Find all working files
git ls-files | xargs git check-attr status | grep 'working'

# Find files needing attention
git ls-files | xargs git check-attr status | grep -E 'broken|needs-test'

# Archive broken examples
mv examples/broken_example.rs examples/OLD/
git add examples/OLD/broken_example.rs
git commit -m 'archive(examples): move broken example [BROKEN]'

# Mark file as working after fixing
echo 'examples/fixed_example.rs status=working' >> .gitattributes
git add .gitattributes
git commit -m 'status: mark example as working [WORKING]'
```

### 2.4. Project Status Overview
Current status (as of 2025-07-04):
- ✅ **5 Working items**: `src/main.rs` (Enhanced Lissajous visualizer), 2 examples, 2 sub-projects
- 🗃️ **Archived**: 11 broken examples moved to `examples/OLD/`
- 📁 **Focus**: Main interactive D-LinOSS Lissajous visualizer

**💡 Pro Tip**: Run `./scripts/check_project_status.sh` before making commits to ensure code quality!

---

## 3. Install & Run

### 3.1. Prerequisites
- Rust programming language (latest stable version recommended). Install from [rustup.rs](https://rustup.rs/).
- Cargo (Rust's package manager, installed with Rust).
- For WGPU backend (GPU support): Ensure your system has compatible graphics drivers and Vulkan/Metal/DX12 support as required by WGPU.

### 3.2. Cloning
```bash
git clone <repository-url> # Replace <repository-url> with the actual URL
cd LinossRust

# 🎯 FIRST THING: Check project status
./scripts/check_project_status.sh
```

### 3.3. Building
To build with the default backend (NdArray - CPU):
```bash
cargo build --release
```
To build with the WGPU backend (GPU):
```bash
cargo build --release --features wgpu_backend
```
Remove `--release` for a debug build.

### 3.4. Running Tests
With default backend (NdArray):
```bash
cargo test

# 🎯 Check which tests are working
git check-attr status tests/*
```
With WGPU backend:
```bash
cargo test --features wgpu_backend
```

### 3.5. Running Examples
Examples are located in the `examples/` directory.

**🎯 PRIMARY EXAMPLE**: Enhanced D-LinOSS Lissajous Visualizer
```bash
# Main interactive application (✅ WORKING)
cargo run --bin linoss_rust
# Features: Trail fading, damping controls, 8 pulse types
```

**✅ VERIFIED WORKING EXAMPLES**:
```bash
# Check which examples are working
git ls-files examples/ | xargs git check-attr status | grep 'working'

# Run working examples
cargo run --example dlinoss_response_visualizer  # ✅ WORKING
cargo run --example burn_demo                    # ✅ WORKING
```

To run an example (e.g., `basic_usage.rs`) with the default backend:
```bash
cargo run --example basic_usage
```
To run an example with the WGPU backend:
```bash
cargo run --example basic_usage --features wgpu_backend
```
To run an example in release mode for better performance:
```bash
cargo run --release --example basic_usage --features wgpu_backend
```

Available examples include:
- ✅ `dlinoss_response_visualizer.rs`: Interactive D-LinOSS response visualization
- ✅ `burn_demo.rs`: Basic Burn framework demonstration
- 🗃️ **Archived examples**: See `examples/OLD/` for 11 archived examples
- 📊 **Status check**: Run `./scripts/check_project_status.sh` to verify current status

### 3.6. Backend Selection
The project supports multiple backends via Cargo features:
- `ndarray_backend` (default if no other backend feature is specified and it's listed as a default feature in `Cargo.toml`): Uses NdArray for CPU-based computation.
- `wgpu_backend`: Uses WGPU for GPU-accelerated computation.

Select the backend when building, running tests, or running examples using the `--features` flag. If both features are enabled, `wgpu_backend` typically takes precedence if not explicitly managed otherwise.

### 3.7. Enabling Log Output (Optional)
This project uses the `log` crate. To see log output, set the `RUST_LOG` environment variable.
```bash
# Example: Debug logs from linoss_rust crate for a test with WGPU
RUST_LOG=linoss_rust=debug cargo test --features wgpu_backend your_test_name

# Example: Info logs from linoss_rust crate for an example with NdArray
RUST_LOG=linoss_rust=info cargo run --example basic_usage --features ndarray_backend
```
Ensure a logger like `env_logger` is initialized in `main.rs` or test setups (e.g., `env_logger::init()` or `env_logger::builder().is_test(true).try_init()`).

---

## 4. Abstract Summary

LinOSS models, inspired by cortical dynamics, are based on forced harmonic oscillators. They aim for efficient learning on long sequences, offering stable dynamics and universal approximation. The paper proposes LinOSS-IM (Implicit Time Integration) and LinOSS-IMEX (Implicit-Explicit Time Integration). This implementation currently focuses on the real-valued LinOSS-IM.

## 5. Core Concepts for Implementation

### 5.a. State-Space Model (SSM) Fundamentals
The model uses a system of forced linear second-order ODEs:
```
𝐲''(𝑡) = -𝐀𝐲(𝑡) + 𝐁𝐮(𝑡) + 𝐛bias_input
𝐱(𝑡) = 𝐂𝐲(𝑡) + 𝐃𝐮(𝑡)
```
- `𝐲(𝑡)`: hidden state (vector of `d_state_m` dimensions)
- `𝐮(𝑡)`: input signal (vector of `d_input_p` dimensions for `LinossLayer`, `d_model` for `LinossBlock` input)
- `𝐱(𝑡)`: output signal from the layer/block (vector of `d_output_q` dimensions for `LinossLayer`, `d_model` for `LinossBlock` output before GLU)
- `𝐀`: diagonal matrix (`m x m`), its learnable component `a_diag_mat` ensures `A𝑘𝑘 ≥ 0` via `relu`.
- `𝐁`: weight matrix (`m x p` for layer), learnable (`b_matrix`).
- `𝐂`: weight matrix (`q x m` for layer), learnable (`c_matrix`).
- `𝐃`: optional weight matrix (`q x p` for layer), learnable (`d_matrix`), controlled by `enable_d_feedthrough`.
- `𝐛bias_input`: input bias vector (`m`), learnable (`bias_b`).

With auxiliary state `𝐳(𝑡) = 𝐲'(𝑡)`, it becomes a first-order system:
```
𝐳'(𝑡) = -𝐀𝐲(𝑡) + 𝐁𝐮(𝑡) + 𝐛bias_input // Bias is included with the 𝐁𝐮 term
𝐲'(𝑡) = 𝐳(𝑡)
```

### 5.b. Forced Harmonic Oscillators
The model is founded on uncoupled forced harmonic oscillators.

### 5.c. Discretization Methods

#### 5.c.i. LinOSS-IM (Implicit Time Integration)
Discretized states `𝐳𝑛, 𝐲𝑛` at `𝑡𝑛 = 𝑛Δ𝑡`:
```
𝐳𝑛 = 𝐳𝑛−1 + Δ𝑡(-𝐀𝐲𝑛 + 𝐁𝐮𝑛 + 𝐛bias_input)
𝐲𝑛 = 𝐲𝑛−1 + Δ𝑡𝐳𝑛
```
This can be written as `𝐌𝐱𝑛 = 𝐱𝑛−1 + 𝐅𝑛`, where `𝐱𝑛 = [𝐳𝑛, 𝐲𝑛]⊤`.
The recurrence is `𝐱𝑛 = 𝐌_IM𝐱𝑛−1 + 𝐅_IM_𝑛`, where `𝐌_IM = 𝐌⁻¹`.
`𝐌⁻¹` is computed efficiently. This method is dissipative.
The current implementation is real-valued and focuses on this variant.

#### 5.c.ii. LinOSS-IMEX (Implicit-Explicit Time Integration)
Discretization:
```
𝐳𝑛 = 𝐳𝑛−1 + Δ𝑡(-𝐀𝐲𝑛−1 + 𝐁𝐮𝑛 + 𝐛bias_input)
𝐲𝑛 = 𝐲𝑛−1 + Δ𝑡𝐳𝑛
```
Matrix form `𝐱𝑛 = 𝐌_IMEX𝐱𝑛−1 + 𝐅_IMEX_𝑛`. This method is conservative.
(Not yet implemented, planned for future).

### 5.d. Fast Recurrence via Associative Parallel Scans
The recurrence `𝐱𝑛 = 𝐌𝐱𝑛−1 + 𝐅𝑛` can be parallelized.
The associative operation: `(𝐚₁, 𝐚₂) ⋅ (𝐛₁, 𝐛₂) = (𝐛₁∘𝐚₁, 𝐛₁∘𝐚₂ + 𝐛₂)`
The `LinossLayer` implements several scan algorithms:
- `forward_step`: Single step update (used by sequential scan).
- `forward_recurrent_sequential_scan`: Sequential loop over the sequence.
- `forward_parallel_scan`: Parallel scan (recursive doubling).
- `forward_tree_scan`: Tree-based parallel scan.
- `forward_work_efficient_scan`: Work-efficient parallel scan.
These operate on real-valued states and matrices.

## 6. Rust and Burn Crate Considerations

### 6.a. Mapping to Burn Tensors
- Matrices (`a_diag_mat`, `b_matrix`, `c_matrix`, `d_matrix`) and vectors (`bias_b`, states `y_state`, `z_state`, inputs `u_input`, outputs `x_output`) are Burn tensors.
- **Tensor Data Handling**:
    - **Creation (Input to Tensor)**: Use `burn::tensor::TensorData::new(data_vec, shape)` to wrap raw data, then `Tensor::<B, D>::from_data(tensor_data.convert::<ElementType>(), &device)`.
    - **Extraction (Output from Tensor)**: Use `tensor.into_data()` to get a `burn::tensor::Data<Element, Rank>` struct. Then, `data_struct.into_vec()` can convert it to a `Vec<Element>`. For scalars, `tensor.into_scalar()` can be used if the tensor is a single element.
- `a_diag_mat`: 1D tensor for the diagonal of `𝐀`. `𝐀` itself is `relu(a_diag_mat)`.
- All parameters and states in the current LinOSS-IM are real-valued (typically `f32`).
- **Naming**: Struct fields and variables use `snake_case` (e.g., `a_matrix`, `b_matrix`). Comments map to mathematical notation.

### 6.b. Burn Modules
- `src/linoss/layer.rs`: `LinossLayer<B: Backend>` - Implements a single LinOSS-IM layer (real-valued).
    - Manages parameters: `a_diag_mat`, `b_matrix`, `c_matrix`, `d_matrix` (optional), `bias_b`.
    - `forward_step(u_input, y_prev_state, z_prev_state)` method for single time-step update.
    - Various scan methods (e.g., `forward_parallel_scan`) for sequence processing.
- `src/linoss/block.rs`: `LinossBlock<B: Backend>` - Comprises a `LinossLayer`, followed by GLU activation and `LayerNorm`.
- `src/linoss/model.rs`: `FullLinossModel<B: Backend>` - Stacks `LinossBlock`s with input/output linear projection layers and residual connections.

### 6.c. Efficient Computations
- Diagonal `𝐀` allows element-wise operations.
- `𝐌⁻¹` for LinOSS-IM is constructed efficiently from `𝐒 = (I + Δ𝑡²𝐀)⁻¹`.
- Parallel scans are implemented for sequence processing.

### 6.d. Parameterization and Initialization
- `𝐀 = relu(a_diag_mat)` ensures non-negative diagonal elements `A𝑘𝑘 ≥ 0`.
- `a_diag_mat` initialized (e.g., from `Distribution::Normal(0.0, init_std)`).
- `delta_t` (Δ𝑡) is a configurable `f32`.
- Other weights (`b_matrix`, `c_matrix`, `d_matrix`, projection weights) use standard Burn initializers (e.g., `Initializer::KaimingUniform` or `Initializer::Normal`).

## 7. Project Structure

```
LinossRust/
├── Cargo.toml
├── README.md
├── linoss_development_directive.md # Detailed dev notes
├── src/
│   ├── lib.rs         # Main library file
│   ├── main.rs        # Main binary (example usage)
│   ├── bin/           # Other binaries (benchmarks, specific tests)
│   │   └── benchmark_scan_methods.rs
│   └── linoss/        # LinOSS specific modules
│       ├── mod.rs     # Declares submodules (layer, block, model, parallel_scan)
│       ├── activation.rs # Activation functions (GLU, SiLU, etc.)
│       ├── block.rs     # LinossBlock module
│       ├── layer.rs     # LinossLayer module
│       ├── model.rs     # FullLinossModel module
│       └── parallel_scan.rs # Parallel scan algorithms
├── examples/            # Example usage scripts
│   ├── basic_usage.rs
│   ├── sine_wave_visualization.rs
│   └── ...
├── tests/               # Integration tests
│   └── integration_test.rs
└── linoss_development_directive.md # In-depth development notes and guidelines
```

## 7. Usage

### 7.1. Building the Project
As described in section [2.3. Building](#23-building).

### 7.2. Running Tests
As described in section [2.4. Running Tests](#24-running-tests).

### 7.3. Running Examples
As described in section [2.5. Running Examples](#25-running-examples).
Key examples to explore:
- `basic_usage.rs`: Minimal example to see the model run.
- `sine_wave_visualization.rs`: Non-TUI console output of model processing a sine wave.
- `chaotic_2d_linoss.rs`: Interactive TUI visualization.
- `compare_scan_methods.rs`: Demonstrates and compares different scan algorithms.

Binaries in `src/bin/` can be run using `cargo run --bin <binary_name>`. For example:
```bash
cargo run --bin benchmark_scan_methods --features wgpu_backend
```

### 7.4. Backend Selection
As described in section [2.6. Backend Selection](#26-backend-selection).
The choice of backend significantly impacts performance, especially for larger models or sequences. `wgpu_backend` is recommended for GPU acceleration.

## 8. Development Status (Current Version)

### 8.a. Implemented Features ✅ **ALL WORKING**
- **LinOSS-IM Core**: Real-valued LinOSS-IM layer (`LinossLayer`) with configurable `delta_t`, state dimensions, and optional `D` matrix.
- **Model Hierarchy**: `LinossBlock` (LinossLayer + GLU + LayerNorm) and `FullLinossModel` (stacked blocks with projections and residuals).
- **Parallel Scans**: Multiple associative scan algorithms for sequence processing:
    - Sequential scan (baseline for `forward_step`).
    - Parallel recursive doubling scan.
    - Tree-based scan.
    - Work-efficient scan.
- **Backend Support**: ✅ **VERIFIED WORKING** - Both `ndarray` (CPU) and `wgpu` (GPU) backends compile and run successfully.
- **Configuration**: Struct-based configuration for all layers and models, deriving `burn::module::Config`.
- **Examples & Benchmarks**: ✅ **ALL EXAMPLES FIXED** - All examples compile and run successfully with both backends.
    - `sine_wave_visualization.rs` - Clean inference demonstration
    - `basic_usage.rs` - Minimal working example  
    - `compare_scan_methods.rs` - Performance comparison (optimized for reasonable execution time)
    - `chaotic_2d_linoss.rs` - Interactive TUI visualization
    - `damped_sine_response.rs` - Training with TUI
    - `flyLinoss.rs` - Tensor visualization TUI
    - `train_linoss.rs` - Full training pipeline
- **Naming Conventions**: Adherence to Rust `snake_case` naming, with comments linking to mathematical notation.
- **Documentation**: Comprehensive status tracking in `EXAMPLES_STATUS.md`.

### 8.b. Key Characteristics
- **Real-Valued**: Current implementation focuses on real-valued states and parameters.
- **Burn Integration**: Leverages Burn's tensor library, module system, and backend abstraction.
- **Modularity**: Code is organized into distinct modules for layers, blocks, and models.
- **✅ Stability**: All examples compile cleanly with Burn 0.17.1 and run successfully.
- **✅ Performance**: Optimized examples complete in reasonable time with proper computational complexity.

### 8.c. Recent Fixes (June 2025)
- **Fixed Dependencies**: Removed broken `burn-train` and `burn-dataset` dependencies.
- **Simplified Examples**: Replaced complex training frameworks with clean, working inference examples.
- **Performance Optimization**: Reduced computational complexity in performance benchmarks.
- **Backend Verification**: Confirmed both ndarray and wgpu backends work correctly.
- **Import Updates**: Fixed all import issues for Burn 0.17.1 compatibility.
- **Warning Cleanup**: Removed unused imports and dead code warnings.

### 8.d. Future Enhancements
(Refer to `linoss_development_directive.md` for a more detailed list)
- Implementation of LinOSS-IMEX variant.
- Support for complex-valued states and parameters (if deemed beneficial).
- More comprehensive test suites.
- Additional example applications (e.g., time-series forecasting, sequence classification).
- Further performance optimizations and benchmarking.

## 🌊 Featured: NeuroBreeze - Brain Dynamics Visualization

**NEW**: Real-time neural dynamics visualization with 3×16 spinning Lissajous intercommunication patterns!

```bash
cargo run --example pure_dlinoss_brain_dynamics
```

### Key Features:
- **🧠 3 Neural Regions**: Prefrontal Cortex, Default Mode Network, Thalamus
- **🌀 3×16 Signal Matrix**: Each region shows 16 intercommunication signals as spinning patterns
- **⚡ Velocity-Driven Animation**: 🔴🟠🟡🟢🔵🟣⚫⚪ = high velocity, ◐◓◑◒ = medium velocity
- **🔄 Bidirectional Connectivity**: Full 6×6 connectivity matrices between all dLinOSS blocks  
- **📡 Real-time Data Streaming**: JSON data streams through `/tmp/dlinoss_brain_pipe`
- **🎮 Interactive Controls**: Press 'p' to pause, 'd' for damping, '+/-' for coupling strength

This showcases dLinOSS (Damped Linear Oscillatory State-Space) models for brain dynamics modeling with beautiful real-time visualization of intercommunication between neural regions.

---

