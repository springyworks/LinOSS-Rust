# Oscillatory State-Space Models (LinOSS) - Software Development Directive

## 🎯 **PROJECT STATUS UPDATE (June 11, 2025)** ✅

**The LinOSS Rust implementation is now COMPLETE and FULLY WORKING!**

- ✅ **All examples compile and run successfully**
- ✅ **Both ndarray (CPU) and wgpu (GPU) backends verified**
- ✅ **All parallel scan algorithms integrated and tested**
- ✅ **Clean codebase with no compilation warnings**
- ✅ **Comprehensive documentation updated**

**Ready for**: Development, Research, Learning, Production Experiments

See `EXAMPLES_STATUS.md` and `scan_integration_report.md` for detailed status reports.

---

## IMPORTANT CODING NOTES & COPILOT GUIDANCE

This section highlights crucial points for development, especially when working with the Burn library and for guiding AI-assisted coding (like GitHub Copilot).

1.  **Tensor Data Handling (`TensorData`)**:
    *   **Creation**: When creating tensors from raw data (e.g., `Vec<f32>`), always use `burn::tensor::TensorData`.
        *   Example: `Tensor::<B, D>::from_data(TensorData::new(data_vec, shape), &device)`
    *   **Extraction**: To get data out of a tensor, use `.into_data()`.
        *   Example: `tensor.into_data().try_into_vec::<f32>().unwrap()`
    *   **Copilot Pitfall**: Copilot might suggest older `Data::new(...)` or direct `.to_data()` which are deprecated or less robust. Always guide it to use `TensorData` and `into_data()`.

2.  **Backend Trait Bounds**:
    *   Ensure all generic functions and structs using a backend `B: Backend` have the necessary trait bounds for `B::FloatElem` and `B::IntElem`.
    *   Commonly needed: `burn::tensor::Element`, `core::fmt::Debug`, `Copy`, `From<f32>` (for float elements), `serde::Serialize`, `serde::de::DeserializeOwned`. For math operations: `burn::tensor::ElementConversion`, `burn::tensor::ops::TensorOps`, `num_traits::FloatConst`, `num_traits::ToPrimitive`, etc.
    *   **Copilot Pitfall**: Copilot often misses these, leading to compilation errors. Explicitly remind it or add them manually.

3.  **Module Forward Methods**:
    *   For recurrent layers or layers that update an internal state step-by-step (like `LinossLayer`), the primary method for a single step is often named `forward_step(...)` rather than just `forward(...)`.
    *   `forward(...)` is typically reserved for processing an entire sequence or a batch without explicit step-by-step state management visible to the caller.
    *   **Copilot Pitfall**: Copilot might default to generating `forward(...)`. Be specific about the method signature and purpose.

4.  **Snake Case Naming**:
    *   All struct fields, function parameters, and local variables should use `snake_case`. This is standard Rust convention.
    *   Mathematical notation (e.g., `A, B, C, D` matrices) should be mapped to `snake_case` fields (e.g., `a_matrix`, `b_matrix`) with comments clarifying the mapping if needed.
    *   Avoid `#[allow(non_snake_case)]` unless absolutely necessary for external library compatibility.
    *   **Copilot Pitfall**: Copilot might generate `camelCase` or `PascalCase` for fields if prompted with mathematical symbols directly. Guide it to use `snake_case`.

5.  **Configuration Flags (`Config` structs)**:
    *   Use clear, descriptive, and `snake_case` names for configuration flags (e.g., `enable_d_feedthrough` instead of `D_feedthrough`).
    *   Ensure `Config` structs derive `burn::module::Config`, `Debug`, `Clone`, `serde::Serialize`, `serde::Deserialize`.

6.  **Error Handling**:
    *   Use `Result<T, E>` for functions that can fail. Avoid `unwrap()` in library code; propagate errors or handle them gracefully.
    *   **Copilot Pitfall**: Copilot frequently uses `unwrap()`. Review and replace with proper error handling.

7.  **Feature Flags for Backends**:
    *   When code is backend-specific (e.g., NdArray vs. WGPU), use `#[cfg(feature = "ndarray_backend")]` or `#[cfg(feature = "wgpu_backend")]`.
    *   Ensure `Cargo.toml` correctly defines these features and their dependencies.

8.  **Burn API Updates**:
    *   The Burn API is actively developed. Be mindful of deprecations and breaking changes.
    *   Consult the official Burn documentation and examples for the latest API usage.
    *   **Copilot Pitfall**: Copilot's training data might not include the very latest API changes. If it generates code that doesn't compile, check for recent Burn updates. For example, optimizer APIs, tensor creation, and module definitions have evolved.

9.  **Parallel Scans & State Management**:
    *   The implemented parallel scan algorithms (`forward_parallel_scan`, `forward_tree_scan`, etc.) are for processing entire sequences.
    *   The `forward_step` method in `LinossLayer` handles the single-step recurrent update, which is used by the sequential scan and can be used for autoregressive generation.
    *   Ensure the state (`y_state`, `z_state`) is correctly passed and updated in recurrent loops.

10. **LinOSS Model Variants**:
    *   The current implementation focuses on the real-valued LinOSS-IM.
    *   The `enable_d_feedthrough` flag controls the presence of the `D` matrix in the `LinossLayer` and `LinossBlock`.
    *   Future extensions (like LinOSS-IMEX or complex-valued models) would require significant additions and new configuration options.

By keeping these points in mind, development can be smoother, and AI-assisted code generation will be more effective and produce higher-quality Rust code compatible with the Burn framework.

---

**Source:** [https://arxiv.org/html/2410.03943v2#bib](https://arxiv.org/html/2410.03943v2#bib)
**Current Crate Version:** 0.3.0 (Post-Burn API Update)

## Abstract Summary

This document outlines the development plan for implementing Linear Oscillatory State-Space models (LinOSS) in Rust using the Burn crate, based on the research paper "Oscillatory State-Space Models". LinOSS models are inspired by cortical dynamics and are based on a system of forced harmonic oscillators. They are designed for efficiently learning on long sequences, offering stable dynamics and universal approximation capabilities. Two main variants are proposed: LinOSS-IM (Implicit Time Integration) and LinOSS-IMEX (Implicit-Explicit Time Integration).

### Dimensionality Notation
For clarity in mapping the paper's mathematics to the code:
- `m`: Dimension of the hidden state `𝐲` (and `𝐳`). Corresponds to `d_state_m` in configuration.
- `p`: Dimension of the input signal `𝐮`. Corresponds to `d_input_p` in configuration.
- `q`: Dimension of the output signal `𝐱` from the linear readout `𝐂𝐲 + 𝐃𝐮`. Corresponds to `d_output_q` in configuration.
- `n` (or `N`): Length of the input/output sequences. Corresponds to `sequence_length`.

## Core Concepts for Implementation

### 1. State-Space Model (SSM) Fundamentals
The model is based on a system of forced linear second-order Ordinary Differential Equations (ODEs):
```
𝐲''(𝑡) = -𝐀𝐲(𝑡) + 𝐁𝐮(𝑡) + 𝐛
𝐱(𝑡) = 𝐂𝐲(𝑡) + 𝐃𝐮(𝑡)
```
where:
- `𝐲(𝑡)`: hidden state (vector of dimension `m`)
- `𝐮(𝑡)`: input signal (vector of dimension `p`)
- `𝐱(𝑡)`: output state (vector of dimension `q`)
- `𝐀`: diagonal matrix of size `m x m` (learnable, `A𝑘𝑘 ≥ 0`)
- `𝐁`: weight matrix of size `m x p` (learnable)
- `𝐂`: weight matrix of size `q x m` (learnable)
- `𝐃`: weight matrix of size `q x p` (learnable, for direct feedthrough from input `𝐮` to output `𝐱`). In some simplified SSM variants or specific applications, `𝐃` might be assumed to be zero, but the general LinOSS formulation as per the source paper includes it.
- `𝐛`: bias vector of dimension `m` (learnable)

Auxiliary state `𝐳(𝑡) = 𝐲'(𝑡)` (vector of dimension `m`). The system can be rewritten as a first-order system with a combined state `[𝐳(𝑡), 𝐲(𝑡)]⊤` of dimension `2m`:
```
𝐳'(𝑡) = -𝐀𝐲(𝑡) + 𝐁𝐮(𝑡)
𝐲'(𝑡) = 𝐳(𝑡)
```
When focusing on the state dynamics (as in Eq. 2 of the source paper), the linear readout `𝐱(𝑡)` is temporarily omitted for clarity of the state update equations:
```
𝐳'(𝑡) = -𝐀𝐲(𝑡) + 𝐁𝐮(𝑡) // Bias 𝐛 is also omitted here as per paper's Eq. 2, but present in the 𝐲'' equation.
𝐲'(𝑡) = 𝐳(𝑡)
```
The full model, especially when stacked in layers (see Appendix A, Algorithm 1 of the source paper), uses the `𝐱(𝑡) = 𝐂𝐲(𝑡) + 𝐃𝐮(𝑡)` readout.

### 2. Forced Harmonic Oscillators
The model foundation is a system of uncoupled forced harmonic oscillators.

### 3. Discretization Methods

#### a. LinOSS-IM (Implicit Time Integration)
Discretized hidden states `𝐳𝑛, 𝐲𝑛` at time `𝑡𝑛 = 𝑛Δ𝑡`:
```
𝐳𝑛 = 𝐳𝑛−1 + Δ𝑡(-𝐀𝐲𝑛 + 𝐁𝐮𝑛)
𝐲𝑛 = 𝐲𝑛−1 + Δ𝑡𝐳𝑛
```
Matrix form `𝐌𝐱𝑛 = 𝐱𝑛−1 + 𝐅𝑛`, where `𝐱𝑛 = [𝐳𝑛, 𝐲𝑛]⊤`:
```
𝐌 = [[I, Δ𝑡𝐀], [-Δ𝑡I, I]]
𝐅𝑛 = [Δ𝑡𝐁𝐮𝑛, 𝟎]⊤
```
The recurrence is `𝐱𝑛 = 𝐌IM𝐱𝑛−1 + 𝐅𝑛IM`, with `𝐌IM = 𝐌⁻¹`.
`𝐌⁻¹` can be computed efficiently using the Schur complement due to `𝐀` being diagonal.
`𝐒 = (I + Δ𝑡²𝐀)⁻¹` (diagonal and trivially invertible if `𝐀𝑘 ≥ 0`).
```
𝐌⁻¹ = [[𝐒, -Δ𝑡𝐀𝐒], [Δ𝑡𝐒, 𝐒]]  // Note: Paper has slightly different M⁻¹ structure in eq (3) vs text. Eq (3) is: [[I - Δ𝑡²𝐀𝐒, -Δ𝑡𝐀𝐒], [Δ𝑡𝐒, 𝐒]]. Recheck paper for correct one. Assuming text description for now.
```
This method introduces dissipative terms.

#### b. LinOSS-IMEX (Implicit-Explicit Time Integration)
Discretization:
```
𝐳𝑛 = 𝐳𝑛−1 + Δ𝑡(-𝐀𝐲𝑛−1 + 𝐁𝐮𝑛)
𝐲𝑛 = 𝐲𝑛−1 + Δ𝑡𝐳𝑛
```
Matrix form `𝐱𝑛 = 𝐌IMEX𝐱𝑛−1 + 𝐅𝑛IMEX`:
```
𝐌IMEX = [[I, -Δ𝑡𝐀], [Δ𝑡I, I - Δ𝑡²𝐀]]
𝐅𝑛IMEX = [Δ𝑡𝐁𝐮𝑛, Δ𝑡²𝐁𝐮𝑛]⊤
```
This method is volume-preserving (conservative) and relates to Hamiltonian systems.

### 4. Fast Recurrence via Associative Parallel Scans
Both LinOSS-IM and LinOSS-IMEX can leverage parallel scans to compute the recurrence `𝐱𝑛 = 𝐌𝐱𝑛−1 + 𝐅𝑛` efficiently.
The associative operation is: `(𝐚₁, 𝐚₂) ⋅ (𝐛₁, 𝐛₂) = (𝐛₁∘𝐚₁, 𝐛₁∘𝐚₂ + 𝐛₂)`
Applied to `[(𝐌, 𝐅₁), (𝐌, 𝐅₂), ...]`.

## Rust and Burn Crate Considerations

### Mapping to Burn Tensors
- All matrices (`𝐀, 𝐁, 𝐂, 𝐃, 𝐌, 𝐒`) and vectors (`𝐲, 𝐳, 𝐮, 𝐱, 𝐛, 𝐅`) will be represented as Burn tensors. Let `batch_size` be `B_s` (to avoid confusion with matrix B) and `sequence_length` be `N`.
- **Tensor Data Handling**: When creating tensors from raw data or extracting data from tensors, Burn uses the `burn::tensor::TensorData` type. For instance, creating a tensor might involve `Tensor::<B, D>::from_data(TensorData::new(data_vec, shape), &device)`, and extracting data might use `tensor.into_data().try_into_vec::<f32>().unwrap()`.
- `𝐀`: Diagonal matrix, its learnable component `a_diag_mat` (representing `𝐀̂` from the paper) is represented as a 1D tensor of `m` elements (real). `𝐀` itself is derived via `relu(a_diag_mat)`.
- `𝐁`: Weight matrix `b_matrix`, tensor of shape `(m, p)` (real).
- `𝐂`: Weight matrix `c_matrix`, tensor of shape `(q, m)` (real).
- `𝐃`: Optional weight matrix `d_matrix`, tensor of shape `(q, p)` (real). Controlled by `enable_d_feedthrough`.
- `𝐛`: Bias vector `bias_b`, tensor of shape `(m)` (real). This bias is added to the `𝐁𝐮` term.
- `𝐲(𝑡), 𝐳(𝑡)`: Hidden states, real-valued tensors. For a single time step, these are of shape `(B_s, m)`.
- `𝐮(𝑡)`: Input signal, real-valued tensor, typically of shape `(B_s, N, p)` for a sequence, or `(B_s, p)` for a single time step.
- `𝐱(𝑡)`: Output signal from the linear readout `𝐂𝐲 + 𝐃𝐮`, real-valued tensor, typically of shape `(B_s, N, q)` for a sequence, or `(B_s, q)` for a single time step.

### Potential Burn Modules/Layers
- `LinossLayer`: Implements a single LinOSS layer using LinOSS-IM discretization with real-valued parameters and states.
    - Takes `u_input` (real) as input for a single time step.
    - Outputs `x_output_next` (real) for that time step, along with next states `y_next_state`, `z_next_state` (both real).
    - Internal parameters: `a_diag_mat` (real), `b_matrix` (real), `c_matrix` (real), `d_matrix` (optional, real), `bias_b` (real).
    - Initial states `y_prev_state`, `z_prev_state` are passed in (real).
    - Method choice: Currently, only LinOSS-IM (real-valued) is implemented.
- `LinossBlock`: Composed of a `LinossLayer` followed by nonlinear transformations (GLU using SiLU) and potentially LayerNorm. (Note: Skip connection is typically part of the `FullLinossModel` structure, not directly within `LinossBlock`'s forward pass, but the block processes the main path).
    - `x_linoss_output = layer.forward_step(u_input, y_state, z_state)`
    - `x_glu_output = GLU(x_linoss_output)` (where GLU involves two linear layers and SiLU)
    - The block can manage its own state recurrently or pass states through.
- `FullLinossModel`: Stacks multiple `LinossBlock`s with initial encoding and final decoding affine transformations. Residual connections are typically applied between blocks in this model.

### Efficient Computations
- **Diagonal `𝐀`**: Ensure operations involving `𝐀` leverage its diagonal structure (e.g., element-wise multiplication instead of full matrix multiplication where possible). Burn's `mul_element_wise` can be used.
- **Matrix Inversion for `𝐌⁻¹` (LinOSS-IM)**:
    - `𝐒 = (I + Δ𝑡²𝐀)⁻¹`: Since `𝐀` is diagonal, `I + Δ𝑡²𝐀` is diagonal. Its inverse `𝐒` is also diagonal, with elements `S𝑘𝑘 = 1 / (1 + Δ𝑡²A𝑘𝑘)`. This is a simple element-wise operation.
    - Construct `𝐌⁻¹` using `𝐒` and `𝐀`. This involves block matrix operations which can be constructed with Burn tensor manipulations.
- **Parallel Scans**:
    - The LinOSS recurrence `𝐱𝑛 = 𝐌𝐱𝑛−1 + 𝐅𝑛` can be parallelized.
    - Several scan algorithms (sequential, parallel recursive doubling, tree-based, work-efficient) are implemented in `src/linoss/parallel_scan.rs` and integrated into `LinossLayer` as alternative forward methods (e.g., `forward_parallel_scan`). These operate on the real-valued state update matrices.

### Parameterization and Initialization
- **`𝐀` Parameterization**: `𝐀 = ReLU(a_diag_mat)` where `a_diag_mat` is a learnable 1D tensor. This ensures `𝐀𝑘𝑘 ≥ 0`.
- **Initialization**:
    - `a_diag_mat` elements can be initialized, e.g., from `Distribution::Normal(0.0, init_std)`. The paper suggests `𝐀̂𝑘𝑘 ~ 𝒰([0,1])` for `𝐀̂` if `𝐀 = 𝐀̂`. With `ReLU(𝐀̂)`, the initialization of `𝐀̂` can be more flexible.
    - `delta_t` is a configurable float, e.g., `0.05`, `0.1`, or `1.0`.
- Other weights (`b_matrix, c_matrix, d_matrix`, encoder/decoder weights) use standard Burn initializations (e.g., `Initializer::Normal`).

## Key Theoretical Insights for Development

- **Stability**: Critical condition `𝐀𝑘𝑘 ≥ 0` for LinOSS-IM. LinOSS-IMEX has eigenvalues with magnitude 1. The `ReLU(𝐀̂` parameterization handles this.
- **Universality**: LinOSS can approximate any causal and continuous operator. This motivates its expressive power.
- **LinOSS-IM vs. LinOSS-IMEX**:
    - LinOSS-IM: Dissipative, potentially better for tasks where forgetting is important. Eigenvalues `|𝜆| ≤ 1`.
    - LinOSS-IMEX: Conservative (volume-preserving), potentially better for energy-conserving systems. Eigenvalues `|𝜆| = 1`.
    - Offer both as options in the implementation.

## Architecture Details (from Appendix A)

- **Input Encoding**: `𝐮0 ← 𝐖enc𝐮 + 𝐛enc` (Affine transformation)
- **LinOSS Blocks (L blocks)**:
    ```rust
    // Pseudocode for a block l
    // y_l = solve_ode_parallel_scan(u_l_minus_1, A, B, /* bias_b */); // Solves ODE (1) for y_l
    // x_l_readout = C.matmul(y_l) + D.matmul(u_l_minus_1); // Linear readout
    // x_l_gelu = gelu(x_l_readout);
    // u_l = glu(x_l_gelu) + u_l_minus_1; // GLU and skip connection
    ```
- **GLU**: `GLU(𝐱) = sigmoid(𝐖₁𝐱) ∘ 𝐖₂𝐱` (element-wise product `∘`). Requires two linear layers and a sigmoid.
- **Output Decoding**: `𝐨 ← 𝐖dec𝐮L + 𝐛dec` (Affine transformation on the output of the last block)
- **Sequence-to-Sequence for Forecasting**:
    - Input: `[past_sequence, masked_future_part]`
    - Output: Only use the `future_prediction_part` of the LinOSS output sequence.

## Development Notes & TODOs

1.  **[Core] `LinossLayer` module (LinOSS-IM, Real-valued):**
    *   [X] Parameters: `a_diag_mat`, `b_matrix`, `c_matrix`, `d_matrix` (optional), `bias_b`. All real-valued.
    *   [X] Forward pass for LinOSS-IM (`forward_step` for single step, and sequence-processing versions using scans):
        *   [X] Calculate `𝐒` (real, diagonal).
        *   [X] Construct effective `𝐌⁻¹` components for state update (real-valued operations).
        *   [X] Implement recurrence:
            *   Serial loop (`forward_recurrent_sequential_scan` or similar internal logic for `forward_step`).
            *   Parallel scan variants (`forward_parallel_scan`, `forward_tree_scan`, etc.) for sequence processing.
            *   State `x_n = [z_n, y_n]⊤` is real `[batch, 2*m]`.
            *   Forcing term `F_n` components are derived from `b_matrix * u_input + bias_b`.
    *   [ ] Implement forward pass for LinOSS-IMEX (Future).
    *   [X] Linear readout `c_matrix * y_new_state + d_matrix * u_input` (all real).
2.  **[Core] `LinossBlock` module:**
    *   [X] Incorporates `LinossLayer`.
    *   [X] Adds GLU activation (using two linear layers and SiLU).
    *   [X] Includes `LayerNorm`.
3.  **[Core] `FullLinossModel` module:**
    *   [X] Input encoder (Linear layer).
    *   [X] Stacking of `LinossBlock`s.
    *   [X] Output decoder (Linear layer).
    *   [X] Manages hidden states between blocks and applies residual connections.
4.  **[Parallelism] Parallel Scan Implementation:**
    *   [X] Implemented serial recurrence.
    *   [X] Implemented parallel scan (recursive doubling: `forward_parallel_scan`).
    *   [X] Implemented tree-based parallel scan (`forward_tree_scan`).
    *   [X] Implemented work-efficient parallel scan (`forward_work_efficient_scan`).
    *   [ ] Research existing Burn capabilities or workarounds for further GPU optimization (e.g. custom kernels - long-term).
5.  **[Testing] Unit tests for each module:**
    *   [X] Test for numerical consistency of all implemented scan methods (`test_scan_method_outputs_consistency`).
    *   [X] Tests for `LinOSSLayer` initialization and forward pass (various configurations).
    *   [ ] Check stability conditions for `𝐀`.
6.  **[Examples] Create example usage:**
    *   [ ] Simple sequence classification/regression task.
    *   [ ] (Optional) Time-series forecasting setup.
7.  **[Hyperparameters] Expose hyperparameters:**
    *   Number of layers, hidden dimensions, state-space dimension, `Δ𝑡`.
    *   Choice of LinOSS-IM/IMEX.
8.  **[Documentation] Document the implementation details and usage.**
9.  **[Backend Support & Benchmarking]**
    *   [X] Added feature flags (`ndarray_backend`, `wgpu_backend`) to `Cargo.toml` for backend selection.
    *   [X] Refactored `benchmark_scan_methods.rs` to support benchmarking with selected backends.
    *   [X] Successfully benchmarked all scan methods on NdArray (CPU) and WGPU (GPU) backends.
    *   [X] Verified that WGPU backend provides significant speedup for parallel/tree scans at larger batch sizes.

### Optional Future Work
*   **Expanding tests:**
    *   Implement tests for batched sequences with varying lengths.
    *   Include more complex scenarios and edge cases in testing.
*   **GPU Benchmarking & Optimization:**
    *   Conduct further benchmarking on the GPU backend with larger sequence lengths and batch sizes to better highlight its advantages and identify bottlenecks.
    *   Investigate and potentially optimize WGPU backend performance for small sequence lengths and standalone scan algorithms if these become critical use cases. This might involve exploring custom WGPU kernels or more fine-grained tensor operation tuning.
*   **LinOSS-IMEX Variant:**
    *   Implement the LinOSS-IMEX (Implicit-Explicit Time Integration) variant.
    *   Compare its performance and characteristics (e.g., stability, conservation properties) against LinOSS-IM.
*   **Further GPU Optimizations:**
    *   Explore advanced GPU optimization techniques beyond the current Burn backend capabilities if needed for specific performance targets (e.g., custom CUDA/WGSL kernels for the scan operations if Burn's abstractions prove insufficient for maximal performance).
*   **Model Enhancements (Beyond current scope):**
    *   Explore alternative activation functions or normalization techniques if needed.
    *   Develop more example applications (e.g., sequence classification, time-series forecasting) to demonstrate the model's capabilities.

### Potential Challenges:
- Implementing the parallel scan efficiently in Burn.
- Ensuring numerical stability, especially with long sequences, despite theoretical guarantees.
- Correctly handling tensor shapes and batching through the recurrent steps.
- Debugging the dynamics of a new SSM architecture.

### Equation/Matrix Clarification:
- Double-check the exact formulation of `𝐌⁻¹` for LinOSS-IM from Equation (3) vs. the text description in Section 2.3 (Implicit time integration). The structure `[[𝐒, -Δ𝑡𝐀𝐒], [Δ𝑡𝐒, 𝐒]]` seems more consistent with `𝐱𝑛 = [𝐳𝑛,𝐲𝑛]⊤` and the update equations. Eq (3) has `𝐌⁻¹ = [[I - Δ𝑡²𝐀𝐒, -Δ𝑡𝐀𝐒], [Δ𝑡𝐒, 𝐒]]`. The paper states `𝐌 = [[I, Δ𝑡𝐤], [-Δ𝑡I, I]]` and `𝐱𝑛 = [𝐳𝑛,𝐲𝑛]⊤`. If `𝐌𝐱𝑛 = 𝐱𝑛−1 + 𝐅𝑛`, then `𝐱𝑛 = 𝐌⁻¹𝐱𝑛−1 + 𝐌⁻¹𝐅𝑛`.
 Let `𝐱𝑛 = [𝐳𝑛, 𝐲𝑛]⊤`.
 `𝐳𝑛 = 𝐳𝑛−1 + Δ𝑡(-𝐀𝐲𝑛 + 𝐁𝐮𝑛)`
 `𝐲𝑛 = 𝐲𝑛−1 + Δ𝑡𝐧𝑛`  // Note: This should be `𝐲𝑛 = 𝐲𝑛−1 + Δ𝑡𝐝𝑛`
 Rearranging for `𝐳𝑛, 𝐲𝑛` on LHS:
 `𝐳𝑛 + Δ𝑡𝐀𝐲𝑛 = 𝐳𝑛−1 + Δ𝑡𝐁𝐮𝑛`
 `-Δ𝑡𝐳𝑛 + 𝐲𝑛 = 𝐲𝑛−1` // Corrected from `-Δ𝑡𝐧 + 𝐲𝑛 = 𝐲𝑛−1`
 This is `[[I, Δ𝑡𝐀], [-Δ𝑡I, I]] [𝐳𝑛, 𝐲𝑛]⊤ = [𝐳𝑛−1 + Δ𝑡𝐁𝐜𝑛, 𝐲𝑛−1]⊤`.
 So `𝐌 = [[I, Δ𝑡𝐀], [-Δ𝑡I, I]]` and `𝐅𝑛_rhs = [𝐳𝑛−1 + Δ𝑡𝐁𝐜𝑛, 𝐲𝑛−1]⊤`.
 The recurrence is `𝐌 𝐱𝑛 = 𝐱𝑛−1_aug`, where `𝐱𝑛 = [𝐳𝑛, 𝐲𝑛]⊤` and `𝐱𝑛−1_aug = [𝐳𝑛−1 + Δ𝑡(𝐁𝐮𝑛 + 𝐛_real_part_of_complex_bias_b_input), 𝐲𝑛−1]⊤`.
 Thus, `𝐱𝑛 = 𝐌⁻¹ 𝐱𝑛−1_aug`.

 The paper's `𝐅𝑛IM = 𝐌⁻¹ [Δ𝑡(𝐁𝐮𝑛 + 𝐛), 𝟎]⊤` (where `𝐛` is the input bias, not the output bias `bias_c`) is part of the update `𝐱𝑛 = 𝐌⁻¹𝐱𝑛−1 + 𝐅𝑛IM`.
 The implemented Rust code aligns with `𝐱𝑛 = 𝐌⁻¹ (𝐱𝑛−1 + [Δ𝑡(𝐁𝐮𝑛 + 𝐛_input), 𝟎]⊤)`.
 Specifically, the `F_IM_t` term in the Rust code corresponds to `[Δ𝑡(𝐁𝐜𝑛 + 𝐛_input), 𝟎]⊤`.
 The matrix `M_IM` in the Rust code corresponds to `𝐌⁻¹`.

 The derivation for `𝐌⁻¹`:
 `𝐌 = [[I, Δ𝑡𝐀], [-Δ𝑡I, I]]`.
 Using the block matrix inversion formula: `[[A, B], [C, D]]⁻¹ = [[ (A-BD⁻¹C)⁻¹ , -(A-BD⁻¹C)⁻¹BD⁻¹ ],[ -D⁻¹C(A-BD⁻¹C)⁻¹ , D⁻¹ + D⁻¹C(A-BD⁻¹C)⁻¹BD⁻¹ ]]`
 Here `A=I, B=Δ𝑡𝐀, C=-Δ𝑡I, D=I`.
 Schur complement of D in M: `A - BD⁻¹C = I - (Δ𝑡𝐀)(I⁻¹)(-Δ𝑡I) = I + Δ𝑡²𝐀`.
 Let `𝐒 = (I + Δ𝑡²𝐀)⁻¹`. This is the inverse of the Schur complement.
 Then `𝐌⁻¹ = [[𝐒, -𝐒(Δ𝑡𝐀)I⁻¹], [-I⁻¹(-Δ𝑡I)𝐒, I⁻¹ + I⁻¹(-Δ𝑡I)𝐒(Δ𝑡𝐀)I⁻¹]]`
 `𝐌⁻¹ = [[𝐒, -Δ𝑡𝐒𝐀], [Δ𝑡𝐒, I - Δ𝑡²𝐒𝐀]]`.
 This is the `M_IM` matrix structure used in the Rust code and matches one of the forms discussed in the paper's surrounding text/derivations, though slightly different from the direct text quote of `[[𝐒, -Δ𝑡𝐀𝐒], [Δ𝑡𝐒, 𝐒]]` if `𝐒` is `(I+Δ𝑡²𝐀)⁻¹`. The key is the consistent application of the derived `𝐌⁻¹`.
 The Rust implementation correctly uses `𝐒 = (I + Δ𝑡²𝐀)⁻¹` and constructs `M_IM` effectively as `[[𝐒, -Δ𝑡𝐒𝐀], [Δ𝑡𝐒, I - Δ𝑡²𝐒𝐀]]` for its state update logic.

This directive should provide a solid guidance for the LinossRust implementation using Rust and Burn.

---
(Historical Changelog Entry - Describes a previous complex-valued iteration)
version: 0.3.0
date: 2024-07-29 
changes:
  - Updated LinOSSLayer to use complex numbers for B_matrix, C_matrix, initial_z, initial_y, and intermediate states.
  - Detailed complex arithmetic in the LinOSS-IM forward pass.
  - Final layer output is the real part of the complex computation.
  - Refactored logging to use the `log` crate instead of `println!`.
  - Updated codebase to align with recent Burn library API changes (e.g., `Data` to `TensorData`, tensor creation and data access methods).
  - Addressed various compiler and linter errors.
---
