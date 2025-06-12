// src/data/formats.rs
// Data format definitions and utilities for LinOSS datasets

use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataFormat {
    Numpy,
    Pickle,
    Csv,
    Json,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArrayData {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl ArrayData {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self { data, shape }
    }
}

// Stub for inspect_data_file
pub fn inspect_data_file<P: AsRef<std::path::Path>>(_path: P) -> Result<(), String> {
    println!("[inspect_data_file] Called on {:?} (stub)", _path.as_ref());
    Ok(())
}
