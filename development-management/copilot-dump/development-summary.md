
---
## [2025-06-12] Burn Tensor Data API Gotcha (LinossRust Recovery)

During the recovery and reconstruction of the LinossRust project, restoring the synthetic data pipeline for Burn 0.17 required careful handling of tensor creation from flat vectors. The Burn API expects:

- Use `TensorData::new(data, shape)` to create tensor data from a flat vector and shape.
- Use `Tensor::from_data(tensor_data, device)` to create the tensor.
- Direct use of `Tensor::from_floats(data, device)` with a flat vector and then reshaping can cause shape/rank panics if not used exactly as expected.

**Example Fix:**
```rust
let tensor_data = TensorData::new(data, [batch, seq_len, input_dim]);
let tensor = Tensor::<B, 3>::from_data(tensor_data, device);
```

**Lesson:**
- Always check the Burn version and API for tensor creation from raw data.
- If you see a panic about "Given dimensions differ from the tensor rank", check that you are using `TensorData::new` and `Tensor::from_data` with the correct shape and data length.

See also: `examples/synthetic_data_usage.rs` for a working reference.
