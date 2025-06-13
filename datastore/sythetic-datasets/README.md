# Synthetic Datasets Directory

This directory is used for testing model parameters save/load functionality with synthetic (generated) datasets and maintains execution traces for debugging and verification.

## Purpose

- **Testing Environment**: Provides a controlled environment for testing model serialization
- **No External Dependencies**: Tests don't rely on external dataset files
- **Reproducible Results**: Synthetic data generation uses deterministic patterns
- **Fast Execution**: No I/O overhead for loading large datasets
- **Execution Tracking**: Maintains detailed traces of test runs and model parameters

## Usage

The directory is used by:
- `test_modelparams_synthetic.rs` - Comprehensive test suite for model parameters functionality
- `test_load_sample_modelparams.rs` - Utility to load and test saved sample model parameters
- `test_with_synthetic_dataset_files.rs` - Test using actual synthetic dataset files (like Iris)
- `generate_synthetic_datasets.rs` - Utility to generate new synthetic datasets
- `examine_traces.sh` - Script to examine execution traces and sample files

## Loading Synthetic Datasets

### In Rust
```rust
// Load like Iris dataset
use ndarray_npy::read_npy;
let features: ndarray::Array2<f32> = read_npy("datastore/sythetic-datasets/iris_like_features.npy")?;
let labels: ndarray::Array1<u8> = read_npy("datastore/sythetic-datasets/iris_like_labels.npy")?;
```

### In Python
```python
# Load with NumPy (compatible format)
import numpy as np
features = np.load('datastore/sythetic-datasets/iris_like_features.npy')
labels = np.load('datastore/sythetic-datasets/iris_like_labels.npy')
```

### Human Inspection
```bash
# View dataset structure
head datastore/sythetic-datasets/iris_like.csv

# View dataset metadata  
cat datastore/sythetic-datasets/iris_like_info.md
```

## Directory Structure

```
sythetic-datasets/
├── README.md                           # This file
├── examine_traces.sh                   # Trace examination utility
├── generate_synthetic_datasets.rs      # Dataset generation utility (in src/bin/)
├── execution-traces/                   # Detailed test execution logs
│   ├── test_execution_<timestamp>.md   # Markdown reports with test results
│   └── ...
├── sample_trained_model_<timestamp>.bin # Sample trained model parameters
├── <dataset_name>.csv                  # Human-readable dataset files
├── <dataset_name>_features.npy         # NumPy feature arrays
├── <dataset_name>_labels.npy           # NumPy label arrays
└── <dataset_name>_info.md              # Dataset metadata and usage info
```

## Available Synthetic Datasets

### Generated Datasets
- **`iris_like`**: 150 samples, 4 features, 3 classes (mimics Iris dataset structure)
- **`small_test`**: 60 samples, 3 features, 2 classes (compact testing)
- **`large_multi`**: 300 samples, 8 features, 5 classes (complex multi-class)
- **`tiny_debug`**: 30 samples, 2 features, 3 classes (minimal debugging)

Each dataset comes in multiple formats:
- **`.csv`** - Human-readable comma-separated values
- **`_features.npy`** - NumPy array of features (compatible with `ndarray-npy`)
- **`_labels.npy`** - NumPy array of labels (compatible with `ndarray-npy`)
- **`_info.md`** - Metadata including generation parameters and usage examples

## Test Coverage

The synthetic tests verify:
- ✅ Basic save/load functionality with bit-exact parameter preservation
- ✅ Training resumption (checkpoint functionality)  
- ✅ File handling edge cases (non-existent files, nested directories)
- ✅ Multiple save/load cycles without parameter drift
- ✅ Automatic directory creation for parameter files

## Execution Traces

### Trace Files (`execution-traces/*.md`)
Detailed markdown reports containing:
- **Test Configuration**: Dataset size, model architecture, training parameters
- **Test Results**: Pass/fail status for each test with detailed explanations
- **Performance Metrics**: Training loss progression, parameter preservation verification
- **Summary Statistics**: Success rates, total tests, execution timestamps

### Sample Model Parameters (`sample_trained_model_*.bin`)
Binary files containing trained model parameters from successful test runs:
- **Format**: Burn BinFileRecorder with FullPrecisionSettings
- **Architecture**: TinyMLP (4 input → 16 hidden → 3 output)
- **Training**: 15 epochs on synthetic classification dataset
- **Size**: ~733 bytes per file
- **Usage**: Can be loaded with `load_modelparams()` function

## Utilities

### `examine_traces.sh`
Script to examine and summarize execution traces:
```bash
./examine_traces.sh
```
Shows:
- List of execution traces with timestamps and success rates
- Sample model parameter files with sizes
- Usage instructions for viewing and loading traces

### `test_load_sample_modelparams.rs`
Test utility to verify saved model parameters:
```bash
cargo run --bin test_load_sample_modelparams
```
- Automatically finds newest sample model parameters
- Loads and runs inference to verify functionality
- Demonstrates proper usage of saved model parameters

## Configuration Options

In `test_modelparams_synthetic.rs`:
- `save_traces = true` - Enable execution trace saving
- `save_sample_modelparams = true` - Enable sample model parameter saving

## Temporary Files

This directory may temporarily contain `.bin` files during test execution:
- `test_synthetic_modelparams.bin` - Temporary test file (cleaned up)
- `test_checkpoint.bin` - Temporary checkpoint file (cleaned up)  
- `cycle_test.bin` - Temporary cycle test file (cleaned up)

Tests automatically clean up temporary files but preserve trace files.

## Generated Data Characteristics

The synthetic datasets feature:
- **Samples**: 120 (configurable)
- **Features**: 4 (configurable) 
- **Classes**: 3 (configurable)
- **Structure**: Class-dependent feature means with deterministic noise
- **Separability**: Designed for successful neural network training convergence

This approach ensures tests are:
- Self-contained and portable
- Fast to execute
- Deterministic and reproducible
- Independent of external data sources
