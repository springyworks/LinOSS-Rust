// LinOSS Rust Library
// This library provides Linear Oscillatory State Space (LinOSS) models using the Burn framework

pub mod linoss;

// Re-export commonly used types for convenience
pub use linoss::{
    model::{FullLinossModel, FullLinossModelConfig},
    block::{LinossBlock, LinossBlockConfig},
    layer::{LinossLayer, LinossLayerConfig},
};

// Error type for LinOSS operations
#[derive(Debug)]
pub enum LinossError {
    InvalidDimensions(String),
    DimensionMismatch(String),
    ComputationError(String),
    ConfigurationError(String),
}

impl std::fmt::Display for LinossError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LinossError::InvalidDimensions(msg) => write!(f, "Invalid dimensions: {}", msg),
            LinossError::DimensionMismatch(msg) => write!(f, "Dimension mismatch: {}", msg),
            LinossError::ComputationError(msg) => write!(f, "Computation error: {}", msg),
            LinossError::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
        }
    }
}

impl std::error::Error for LinossError {}
