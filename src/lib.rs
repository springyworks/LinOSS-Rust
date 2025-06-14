// Removed the invalid `html_extra_files` attribute
pub mod linoss;
pub mod data;
pub mod checkpoint;
pub mod visualization;
pub type Vector = nalgebra::DVector<f64>;
pub type Matrix = nalgebra::DMatrix<f64>;

#[derive(Debug)]
pub enum LinossError {
    DimensionMismatch(String),
    Other(String),
}

impl std::fmt::Display for LinossError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LinossError::DimensionMismatch(s) => write!(f, "Dimension Mismatch: {}", s),
            LinossError::Other(s) => write!(f, "Linoss Error: {}", s),
        }
    }
}

impl std::error::Error for LinossError {}