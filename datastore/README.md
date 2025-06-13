# LinossRust Datastore

This directory contains all datasets and data processing utilities for the LinossRust project.

## Directory Structure

```
datastore/
├── scripts/                    # Data processing utilities
│   └── download_and_convert_dataset.py
├── downloads-of-datasets/      # Raw downloaded datasets
├── processed-by-python/        # Python-processed datasets (.npy format)
├── processed-by-rust-for-burn/ # Rust/Burn-optimized datasets
└── sythetic-datasets/          # Generated synthetic datasets
```

## Data Processing Workflow

### 1. Download and Convert (Python)

Use the provided Python script to download datasets and convert them to `.npy` format:

```bash
# From project root
python3 datastore/scripts/download_and_convert_dataset.py --urls <dataset_url>

# Or process existing files in downloads-of-datasets/
python3 datastore/scripts/download_and_convert_dataset.py
```

**Features:**
- Downloads files from URLs with smart skipping
- Converts `.pkl`, `.csv`, and `.data` files to `.npy`
- Handles mixed-type data (features as float32, labels as uint8)
- Idempotent - skips conversion if output is newer than source
- Copies results to `processed-by-python/`

### 2. Rust/Burn Processing

Further processing for Burn framework happens in Rust code, with results stored in `processed-by-rust-for-burn/`.

## Supported Data Formats

**Input Formats:**
- `.pkl` files (Python pickle with numpy arrays)
- `.csv` files (with or without headers)
- `.data` files (CSV-like format)

**Output Format:**
- `.npy` files (NumPy binary format)
- Features: `float32` arrays
- Labels: `uint8` arrays
- Label mappings: `.json` files (when applicable)

## Example Usage

```bash
# Download and process Iris dataset
cd /path/to/LinossRust
python3 datastore/scripts/download_and_convert_dataset.py \
  --urls https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data

# Check what was created
ls -la datastore/processed-by-python/
```

## Data Processing Script Reference

### `download_and_convert_dataset.py`

**Options:**
- `--urls URL [URL ...]` - Download from URLs before processing
- `--input_dir DIR` - Directory to scan for data files (default: downloads-of-datasets)
- `--processed_dir DIR` - Where to copy .npy files (default: processed-by-python)
- `--force` - Force re-download existing files
- `--dry_run` - Show what would be done without making changes
- `--example` - Show usage examples

**Smart Features:**
- Skips downloads if file already exists (unless `--force`)
- Skips conversion if .npy is newer than source
- Handles subdirectories recursively
- Provides detailed logging and error reporting

## Integration with LinossRust

The processed `.npy` files are designed to work seamlessly with:
- `ndarray-npy` crate for loading in Rust
- Burn framework tensor creation
- LinossRust training examples

See `examples/burn_iris_loader.rs` for usage examples.