// Generate and save synthetic datasets for testing model parameters functionality
// Creates datasets in both CSV (human-readable) and NPY (efficient) formats

use ndarray::{Array1, Array2};
use ndarray_npy::write_npy;
use anyhow::Result;
use std::fs::File;
use std::io::Write;
use std::path::Path;

/// Generate synthetic classification dataset with controllable characteristics
fn generate_synthetic_classification_dataset(
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
    class_separation: f32,
    noise_level: f32,
    random_seed: u64,
) -> (Array2<f32>, Array1<u8>) {
    // Use a simple deterministic pseudo-random generator for reproducibility
    let mut rng_state = random_seed;
    
    // Simple LCG (Linear Congruential Generator) for deterministic randomness
    let mut next_random = move || {
        rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        (rng_state / 65536) % 32768
    };
    
    let mut features = Array2::<f32>::zeros((n_samples, n_features));
    let mut labels = Array1::<u8>::zeros(n_samples);
    
    for i in 0..n_samples {
        let class = i % n_classes;
        labels[i] = class as u8;
        
        // Generate features with class-dependent means and controlled separation
        for j in 0..n_features {
            // Base value depends on class (creates separation)
            let class_mean = (class as f32) * class_separation;
            
            // Feature-dependent offset (creates feature diversity)
            let feature_offset = (j as f32) * 0.3;
            
            // Deterministic "noise" based on sample and feature indices
            let noise_seed = (i * 1000 + j * 100 + class * 10) as u64;
            let mut noise_rng = noise_seed;
            noise_rng = noise_rng.wrapping_mul(1103515245).wrapping_add(12345);
            let noise = ((noise_rng % 10000) as f32 / 10000.0 - 0.5) * noise_level;
            
            // Add some sinusoidal patterns for more interesting structure
            let pattern = ((i as f32 * 0.1 + j as f32 * 0.05).sin() + 
                          (i as f32 * 0.07 + class as f32).cos()) * 0.2;
            
            features[[i, j]] = class_mean + feature_offset + noise + pattern;
        }
    }
    
    (features, labels)
}

/// Save dataset in CSV format for human inspection
fn save_dataset_csv(
    features: &Array2<f32>,
    labels: &Array1<u8>,
    filepath: &str,
    dataset_name: &str,
) -> Result<()> {
    let mut file = File::create(filepath)?;
    
    // Write header with metadata
    writeln!(file, "# Synthetic Dataset: {}", dataset_name)?;
    writeln!(file, "# Samples: {}, Features: {}, Classes: {}", 
             features.shape()[0], features.shape()[1], 
             labels.iter().max().unwrap_or(&0) + 1)?;
    writeln!(file, "# Format: feature1,feature2,...,featureN,class")?;
    writeln!(file)?;
    
    // Write header row
    write!(file, "# ")?;
    for i in 0..features.shape()[1] {
        write!(file, "feature{},", i + 1)?;
    }
    writeln!(file, "class")?;
    
    // Write data rows
    for i in 0..features.shape()[0] {
        for j in 0..features.shape()[1] {
            write!(file, "{:.6},", features[[i, j]])?;
        }
        writeln!(file, "{}", labels[i])?;
    }
    
    println!("✅ CSV dataset saved to: {}", filepath);
    Ok(())
}

/// Save dataset in NPY format for efficient loading
fn save_dataset_npy(
    features: &Array2<f32>,
    labels: &Array1<u8>,
    features_path: &str,
    labels_path: &str,
) -> Result<()> {
    write_npy(features_path, features)?;
    write_npy(labels_path, labels)?;
    
    println!("✅ NPY features saved to: {}", features_path);
    println!("✅ NPY labels saved to: {}", labels_path);
    Ok(())
}

/// Generate dataset metadata file
fn save_dataset_info(
    filepath: &str,
    dataset_name: &str,
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
    class_separation: f32,
    noise_level: f32,
    random_seed: u64,
) -> Result<()> {
    let mut file = File::create(filepath)?;
    
    writeln!(file, "# Synthetic Dataset Information")?;
    writeln!(file, "**Dataset Name**: {}", dataset_name)?;
    writeln!(file, "**Generation Date**: {}", 
             std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs())?;
    writeln!(file)?;
    writeln!(file, "## Dataset Characteristics")?;
    writeln!(file, "- **Samples**: {}", n_samples)?;
    writeln!(file, "- **Features**: {}", n_features)?;
    writeln!(file, "- **Classes**: {}", n_classes)?;
    writeln!(file, "- **Class Separation**: {:.2}", class_separation)?;
    writeln!(file, "- **Noise Level**: {:.2}", noise_level)?;
    writeln!(file, "- **Random Seed**: {}", random_seed)?;
    writeln!(file)?;
    writeln!(file, "## Data Generation Method")?;
    writeln!(file, "- **Class-dependent means**: Each class has different base feature values")?;
    writeln!(file, "- **Feature diversity**: Each feature has slight offset patterns")?;
    writeln!(file, "- **Deterministic noise**: Reproducible pseudo-random variations")?;
    writeln!(file, "- **Sinusoidal patterns**: Added for realistic data structure")?;
    writeln!(file)?;
    writeln!(file, "## Files Generated")?;
    writeln!(file, "- `{}.csv` - Human-readable CSV format", dataset_name)?;
    writeln!(file, "- `{}_features.npy` - Features in NumPy array format", dataset_name)?;
    writeln!(file, "- `{}_labels.npy` - Labels in NumPy array format", dataset_name)?;
    writeln!(file, "- `{}_info.md` - This metadata file", dataset_name)?;
    writeln!(file)?;
    writeln!(file, "## Usage")?;
    writeln!(file, "```rust")?;
    writeln!(file, "// Load in Rust with ndarray-npy")?;
    writeln!(file, "let features: Array2<f32> = read_npy(\"{}_features.npy\")?;", dataset_name)?;
    writeln!(file, "let labels: Array1<u8> = read_npy(\"{}_labels.npy\")?;", dataset_name)?;
    writeln!(file, "```")?;
    writeln!(file)?;
    writeln!(file, "```python")?;
    writeln!(file, "# Load in Python with NumPy")?;
    writeln!(file, "import numpy as np")?;
    writeln!(file, "features = np.load('{}_features.npy')", dataset_name)?;
    writeln!(file, "labels = np.load('{}_labels.npy')", dataset_name)?;
    writeln!(file, "```")?;
    
    println!("✅ Dataset info saved to: {}", filepath);
    Ok(())
}

fn main() -> Result<()> {
    println!("🏭 Synthetic Dataset Generator");
    println!("=============================");
    
    let output_dir = "datastore/sythetic-datasets";
    std::fs::create_dir_all(output_dir)?;
    
    // Dataset configurations to generate
    let datasets = vec![
        ("iris_like", 150, 4, 3, 2.0, 0.5, 42),
        ("small_test", 60, 3, 2, 1.5, 0.3, 123),
        ("large_multi", 300, 8, 5, 3.0, 0.8, 456),
        ("tiny_debug", 30, 2, 3, 1.0, 0.2, 789),
    ];
    
    for (name, n_samples, n_features, n_classes, class_separation, noise_level, seed) in datasets {
        println!("\n📊 Generating dataset: {}", name);
        println!("   Samples: {}, Features: {}, Classes: {}", n_samples, n_features, n_classes);
        
        // Generate the dataset
        let (features, labels) = generate_synthetic_classification_dataset(
            n_samples, n_features, n_classes, class_separation, noise_level, seed
        );
        
        // Save in multiple formats
        let csv_path = format!("{}/{}.csv", output_dir, name);
        let features_npy_path = format!("{}/{}_features.npy", output_dir, name);
        let labels_npy_path = format!("{}/{}_labels.npy", output_dir, name);
        let info_path = format!("{}/{}_info.md", output_dir, name);
        
        // Save files
        save_dataset_csv(&features, &labels, &csv_path, name)?;
        save_dataset_npy(&features, &labels, &features_npy_path, &labels_npy_path)?;
        save_dataset_info(&info_path, name, n_samples, n_features, n_classes, 
                         class_separation, noise_level, seed)?;
        
        // Show dataset statistics
        let class_counts = {
            let mut counts = vec![0; n_classes];
            for &label in labels.iter() {
                counts[label as usize] += 1;
            }
            counts
        };
        
        println!("   Class distribution: {:?}", class_counts);
        
        // Show feature ranges
        let feature_ranges: Vec<(f32, f32)> = (0..n_features)
            .map(|j| {
                let column = features.column(j);
                let min = column.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                let max = column.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                (min, max)
            })
            .collect();
        
        println!("   Feature ranges:");
        for (j, (min, max)) in feature_ranges.iter().enumerate() {
            println!("     Feature {}: [{:.3}, {:.3}]", j + 1, min, max);
        }
    }
    
    println!("\n📁 All synthetic datasets generated in: {}", output_dir);
    println!("   Use these datasets for:");
    println!("   - 🔍 Manual inspection (CSV files)");
    println!("   - 🚀 Fast loading in tests (NPY files)");
    println!("   - 📚 Understanding data characteristics (info files)");
    println!("   - 🧪 Reproducible experiments (fixed seeds)");
    
    Ok(())
}
