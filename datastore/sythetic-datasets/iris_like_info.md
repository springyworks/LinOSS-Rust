# Synthetic Dataset Information
**Dataset Name**: iris_like
**Generation Date**: 1749801741

## Dataset Characteristics
- **Samples**: 150
- **Features**: 4
- **Classes**: 3
- **Class Separation**: 2.00
- **Noise Level**: 0.50
- **Random Seed**: 42

## Data Generation Method
- **Class-dependent means**: Each class has different base feature values
- **Feature diversity**: Each feature has slight offset patterns
- **Deterministic noise**: Reproducible pseudo-random variations
- **Sinusoidal patterns**: Added for realistic data structure

## Files Generated
- `iris_like.csv` - Human-readable CSV format
- `iris_like_features.npy` - Features in NumPy array format
- `iris_like_labels.npy` - Labels in NumPy array format
- `iris_like_info.md` - This metadata file

## Usage
```rust
// Load in Rust with ndarray-npy
let features: Array2<f32> = read_npy("iris_like_features.npy")?;
let labels: Array1<u8> = read_npy("iris_like_labels.npy")?;
```

```python
# Load in Python with NumPy
import numpy as np
features = np.load('iris_like_features.npy')
labels = np.load('iris_like_labels.npy')
```
