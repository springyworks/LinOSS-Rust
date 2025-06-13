# LinossRust Examples

This directory contains runnable Rust examples demonstrating various features, models, and utilities in the LinossRust project.

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
Shows a training loop with TUI for a damped sine response task.

### dlinoss_comparison.rs
Compares different D-LinOSS model variants on a benchmark task.

### flyLinoss.rs
Visualizes tensors and model states in a TUI.

---

For more details on each example, see the comments at the top of each file or run with `--help` if supported.

## Navigation

If your editor supports Markdown link navigation, you can jump to the section for a specific example by clicking the corresponding link in the index above.

If you right-click an example file and your editor supports it, you may be able to search for the section header (e.g., `### burn_iris_loader.rs`) in this README to jump directly to the relevant documentation.
