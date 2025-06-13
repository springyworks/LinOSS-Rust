# Synthetic Dataset Information
**Dataset Name**: small_test
**Generation Date**: 1749801741

## Dataset Characteristics
- **Samples**: 60
- **Features**: 3
- **Classes**: 2
- **Class Separation**: 1.50
- **Noise Level**: 0.30
- **Random Seed**: 123

## Data Generation Method
- **Class-dependent means**: Each class has different base feature values
- **Feature diversity**: Each feature has slight offset patterns
- **Deterministic noise**: Reproducible pseudo-random variations
- **Sinusoidal patterns**: Added for realistic data structure

## Files Generated
- `small_test.csv` - Human-readable CSV format
- `small_test_features.npy` - Features in NumPy array format
- `small_test_labels.npy` - Labels in NumPy array format
- `small_test_info.md` - This metadata file

## Usage
```rust
// Load in Rust with ndarray-npy
let features: Array2<f32> = read_npy("small_test_features.npy")?;
let labels: Array1<u8> = read_npy("small_test_labels.npy")?;
```

```python
# Load in Python with NumPy
import numpy as np
features = np.load('small_test_features.npy')
labels = np.load('small_test_labels.npy')
```
