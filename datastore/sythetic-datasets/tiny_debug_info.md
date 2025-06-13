# Synthetic Dataset Information
**Dataset Name**: tiny_debug
**Generation Date**: 1749801741

## Dataset Characteristics
- **Samples**: 30
- **Features**: 2
- **Classes**: 3
- **Class Separation**: 1.00
- **Noise Level**: 0.20
- **Random Seed**: 789

## Data Generation Method
- **Class-dependent means**: Each class has different base feature values
- **Feature diversity**: Each feature has slight offset patterns
- **Deterministic noise**: Reproducible pseudo-random variations
- **Sinusoidal patterns**: Added for realistic data structure

## Files Generated
- `tiny_debug.csv` - Human-readable CSV format
- `tiny_debug_features.npy` - Features in NumPy array format
- `tiny_debug_labels.npy` - Labels in NumPy array format
- `tiny_debug_info.md` - This metadata file

## Usage
```rust
// Load in Rust with ndarray-npy
let features: Array2<f32> = read_npy("tiny_debug_features.npy")?;
let labels: Array1<u8> = read_npy("tiny_debug_labels.npy")?;
```

```python
# Load in Python with NumPy
import numpy as np
features = np.load('tiny_debug_features.npy')
labels = np.load('tiny_debug_labels.npy')
```
