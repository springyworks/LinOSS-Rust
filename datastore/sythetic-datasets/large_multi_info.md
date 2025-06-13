# Synthetic Dataset Information
**Dataset Name**: large_multi
**Generation Date**: 1749801741

## Dataset Characteristics
- **Samples**: 300
- **Features**: 8
- **Classes**: 5
- **Class Separation**: 3.00
- **Noise Level**: 0.80
- **Random Seed**: 456

## Data Generation Method
- **Class-dependent means**: Each class has different base feature values
- **Feature diversity**: Each feature has slight offset patterns
- **Deterministic noise**: Reproducible pseudo-random variations
- **Sinusoidal patterns**: Added for realistic data structure

## Files Generated
- `large_multi.csv` - Human-readable CSV format
- `large_multi_features.npy` - Features in NumPy array format
- `large_multi_labels.npy` - Labels in NumPy array format
- `large_multi_info.md` - This metadata file

## Usage
```rust
// Load in Rust with ndarray-npy
let features: Array2<f32> = read_npy("large_multi_features.npy")?;
let labels: Array1<u8> = read_npy("large_multi_labels.npy")?;
```

```python
# Load in Python with NumPy
import numpy as np
features = np.load('large_multi_features.npy')
labels = np.load('large_multi_labels.npy')
```
