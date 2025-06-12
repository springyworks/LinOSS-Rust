// src/data/python_converter.rs
// Utilities for converting Python LinOSS datasets to Rust-native formats

use std::path::Path;
use serde::{Deserialize, Serialize};
use crate::data::formats::{ArrayData, DataFormat};
use crate::data::{DatasetMeta, DataSplits};

/// Python dataset structure (as typically stored in pickle files)
/// This represents the expected structure from the Python LinOSS codebase
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PythonDatasetInfo {
    pub collection: String,
    pub dataset_name: String,
    pub has_presplit: bool,
}

/// Convert a Python LinOSS dataset directory to Rust format
pub fn convert_python_linoss_dataset(
    python_data_dir: &Path,
    collection: &str,
    dataset_name: &str,
    output_dir: &Path,
    target_format: DataFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Converting Python LinOSS dataset: {}/{}", collection, dataset_name);
    
    let source_dir = python_data_dir
        .join("processed")
        .join(collection)
        .join(dataset_name);
    
    let output_dataset_dir = output_dir
        .join("processed")
        .join(collection)
        .join(dataset_name);
    
    std::fs::create_dir_all(&output_dataset_dir)?;
    
    // Determine if this dataset has pre-splits
    let has_presplit = source_dir.join("X_train.pkl").exists();
    
    if has_presplit {
        convert_presplit_dataset(&source_dir, &output_dataset_dir, target_format)?;
    } else {
        convert_unified_dataset(&source_dir, &output_dataset_dir, target_format)?;
    }
    
    println!("✓ Conversion complete: {}", output_dataset_dir.display());
    Ok(())
}

/// Convert a dataset with pre-defined splits (X_train.pkl, y_train.pkl, etc.)
fn convert_presplit_dataset(
    source_dir: &Path,
    output_dir: &Path,
    target_format: DataFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    let split_files = [
        ("X_train.pkl", "y_train.pkl", "train"),
        ("X_val.pkl", "y_val.pkl", "val"),
        ("X_test.pkl", "y_test.pkl", "test"),
    ];
    
    let mut all_data = Vec::new();
    let mut all_labels = Vec::new();
    let mut train_indices = Vec::new();
    let mut val_indices = Vec::new();
    let mut test_indices = Vec::new();
    
    let mut current_idx = 0;
    
    for (x_file, y_file, split_name) in &split_files {
        let x_path = source_dir.join(x_file);
        let y_path = source_dir.join(y_file);
        
        if x_path.exists() && y_path.exists() {
            // Load pickle files (placeholder - would use actual pickle deserialization)
            let (x_data, y_data) = load_pickle_data_placeholder(&x_path, &y_path)?;
            
            let n_samples = x_data.shape[0];
            match *split_name {
                "train" => train_indices.extend(current_idx..current_idx + n_samples),
                "val" => val_indices.extend(current_idx..current_idx + n_samples),
                "test" => test_indices.extend(current_idx..current_idx + n_samples),
                _ => {}
            }
            
            all_data.push(x_data);
            all_labels.push(y_data);
            current_idx += n_samples;
        }
    }
    
    if all_data.is_empty() {
        return Err("No valid split files found".into());
    }
    
    // Combine all data
    let combined_data = combine_array_data(all_data)?;
    let combined_labels = combine_array_data(all_labels)?;
    
    // Create metadata
    let meta = DatasetMeta {
        name: source_dir.file_name().unwrap().to_string_lossy().to_string(),
        collection: source_dir.parent().unwrap().file_name().unwrap().to_string_lossy().to_string(),
        n_samples: combined_data.shape[0],
        n_timesteps: combined_data.shape[1],
        n_features: combined_data.shape[2],
        n_classes: if combined_labels.shape.len() > 1 {
            Some(combined_labels.shape[1])
        } else {
            None
        },
        has_splits: true,
        format_version: "1.0".to_string(),
    };
    
    // Create splits
    let splits = DataSplits {
        train_indices,
        val_indices,
        test_indices,
    };
    
    // Save everything
    save_converted_dataset(output_dir, &combined_data, &combined_labels, &meta, Some(&splits), target_format)?;
    
    Ok(())
}

/// Convert a dataset with unified data/labels files
fn convert_unified_dataset(
    source_dir: &Path,
    output_dir: &Path,
    target_format: DataFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    let data_path = source_dir.join("data.pkl");
    let labels_path = source_dir.join("labels.pkl");
    
    if !data_path.exists() || !labels_path.exists() {
        return Err("Required data.pkl and labels.pkl files not found".into());
    }
    
    // Load pickle files
    let (data, labels) = load_pickle_data_placeholder(&data_path, &labels_path)?;
    
    // Create metadata
    let meta = DatasetMeta {
        name: source_dir.file_name().unwrap().to_string_lossy().to_string(),
        collection: source_dir.parent().unwrap().file_name().unwrap().to_string_lossy().to_string(),
        n_samples: data.shape[0],
        n_timesteps: data.shape[1],
        n_features: data.shape[2],
        n_classes: if labels.shape.len() > 1 {
            Some(labels.shape[1])
        } else {
            None
        },
        has_splits: false,
        format_version: "1.0".to_string(),
    };
    
    // Check for original splits
    let splits = if source_dir.join("original_idxs.pkl").exists() {
        // Load original indices (placeholder)
        load_original_splits_placeholder(&source_dir.join("original_idxs.pkl"))?
    } else {
        None
    };
    
    save_converted_dataset(output_dir, &data, &labels, &meta, splits.as_ref(), target_format)?;
    
    Ok(())
}

/// Placeholder function for loading pickle data
/// In a real implementation, this would use serde-pickle-rs with proper NumPy array handling
fn load_pickle_data_placeholder(
    x_path: &Path,
    y_path: &Path,
) -> Result<(ArrayData, ArrayData), Box<dyn std::error::Error>> {
    // This is a placeholder implementation
    // Real implementation would:
    // 1. Use serde-pickle-rs to deserialize pickle files
    // 2. Handle NumPy array structures specifically
    // 3. Convert to our ArrayData format
    
    println!("Loading (placeholder): {} and {}", x_path.display(), y_path.display());
    
    // For now, return dummy data to demonstrate the structure
    let dummy_x_data = ArrayData {
        shape: vec![100, 50, 3], // 100 samples, 50 timesteps, 3 features
        dtype: "f32".to_string(),
        data: vec![0.0; 100 * 50 * 3],
    };
    
    let dummy_y_data = ArrayData {
        shape: vec![100, 2], // 100 samples, 2 classes
        dtype: "f32".to_string(),
        data: vec![0.0; 100 * 2],
    };
    
    Ok((dummy_x_data, dummy_y_data))
}

/// Placeholder function for loading original split indices
fn load_original_splits_placeholder(
    splits_path: &Path,
) -> Result<Option<DataSplits>, Box<dyn std::error::Error>> {
    println!("Loading original splits (placeholder): {}", splits_path.display());
    
    // Placeholder splits
    Ok(Some(DataSplits {
        train_indices: (0..70).collect(),
        val_indices: (70..85).collect(),
        test_indices: (85..100).collect(),
    }))
}

/// Combine multiple ArrayData structures
fn combine_array_data(arrays: Vec<ArrayData>) -> Result<ArrayData, Box<dyn std::error::Error>> {
    if arrays.is_empty() {
        return Err("No arrays to combine".into());
    }
    
    // Check that all arrays have the same shape except for the first dimension
    let first_shape = &arrays[0].shape;
    for arr in &arrays[1..] {
        if arr.shape[1..] != first_shape[1..] {
            return Err("Arrays have incompatible shapes for combining".into());
        }
    }
    
    // Calculate new shape
    let total_samples: usize = arrays.iter().map(|a| a.shape[0]).sum();
    let mut new_shape = first_shape.clone();
    new_shape[0] = total_samples;
    
    // Combine data
    let mut combined_data = Vec::new();
    for arr in &arrays { // Borrow instead of consuming
        combined_data.extend(arr.data.iter().cloned()); // Clone the data
    }
    
    Ok(ArrayData {
        shape: new_shape,
        dtype: arrays[0].dtype.clone(),
        data: combined_data,
    })
}

/// Save the fully converted dataset
fn save_converted_dataset(
    output_dir: &Path,
    data: &ArrayData,
    labels: &ArrayData,
    meta: &DatasetMeta,
    splits: Option<&DataSplits>,
    target_format: DataFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    // Save metadata
    let meta_file = std::fs::File::create(output_dir.join("meta.json"))?;
    serde_json::to_writer_pretty(meta_file, meta)?;
    
    // Save splits if present
    if let Some(splits) = splits {
        let splits_file = std::fs::File::create(output_dir.join("splits.json"))?;
        serde_json::to_writer_pretty(splits_file, splits)?;
    }
    
    // Save data and labels in target format
    let data_ext = match target_format {
        DataFormat::Json => "json",
        DataFormat::Parquet => "parquet",
        DataFormat::Pickle => "pkl",
        _ => return Err("Unsupported target format".into()),
    };
    
    // Save data
    match target_format {
        DataFormat::Json => {
            let data_file = std::fs::File::create(output_dir.join(format!("data.{}", data_ext)))?;
            serde_json::to_writer_pretty(data_file, data)?;
            
            let labels_file = std::fs::File::create(output_dir.join(format!("labels.{}", data_ext)))?;
            serde_json::to_writer_pretty(labels_file, labels)?;
        }
        DataFormat::Pickle => {
            let data_serialized = serde_pickle_rs::to_vec(data, Default::default())?;
            std::fs::write(output_dir.join(format!("data.{}", data_ext)), data_serialized)?;
            
            let labels_serialized = serde_pickle_rs::to_vec(labels, Default::default())?;
            std::fs::write(output_dir.join(format!("labels.{}", data_ext)), labels_serialized)?;
        }
        _ => return Err("Format not implemented for saving".into()),
    }
    
    println!("✓ Saved dataset files in {} format", data_ext);
    Ok(())
}

/// Create a CLI tool for converting Python datasets
pub fn convert_python_datasets_cli() -> Result<(), Box<dyn std::error::Error>> {
    println!("LinOSS Dataset Converter (Rust)");
    println!("================================");
    
    // These would be command-line arguments in a real CLI
    let python_data_dir = Path::new("../linoss_kos/data_dir");
    let output_dir = Path::new("datastore");
    let target_format = DataFormat::Json; // Could be configurable
    
    if !python_data_dir.exists() {
        println!("Python data directory not found: {}", python_data_dir.display());
        println!("Please ensure the Python LinOSS project is available at the expected location.");
        return Ok(());
    }
    
    // Find available datasets
    let processed_dir = python_data_dir.join("processed");
    if !processed_dir.exists() {
        println!("No processed directory found in Python project");
        return Ok(());
    }
    
    // Scan for collections
    for collection_entry in std::fs::read_dir(&processed_dir)? {
        let collection_path = collection_entry?.path();
        if collection_path.is_dir() {
            let collection_name = collection_path.file_name().unwrap().to_string_lossy();
            println!("Found collection: {}", collection_name);
            
            // Scan for datasets in this collection
            for dataset_entry in std::fs::read_dir(&collection_path)? {
                let dataset_path = dataset_entry?.path();
                if dataset_path.is_dir() {
                    let dataset_name = dataset_path.file_name().unwrap().to_string_lossy();
                    println!("  Found dataset: {}", dataset_name);
                    
                    // Convert this dataset
                    match convert_python_linoss_dataset(
                        python_data_dir,
                        &collection_name,
                        &dataset_name,
                        output_dir,
                        target_format,
                    ) {
                        Ok(_) => println!("  ✓ Converted successfully"),
                        Err(e) => println!("  ✗ Conversion failed: {}", e),
                    }
                }
            }
        }
    }
    
    println!("\nConversion process completed!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    
    #[test]
    fn test_combine_array_data() {
        let arr1 = ArrayData {
            shape: vec![2, 3, 1],
            dtype: "f32".to_string(),
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        };
        
        let arr2 = ArrayData {
            shape: vec![1, 3, 1],
            dtype: "f32".to_string(),
            data: vec![7.0, 8.0, 9.0],
        };
        
        let combined = combine_array_data(vec![arr1, arr2]).unwrap();
        assert_eq!(combined.shape, vec![3, 3, 1]);
        assert_eq!(combined.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    }
}
