//! linoss_rust: Core LinOSS Models and Utilities
//!
//! This library provides the main LinOSS (Linear Oscillatory State-Space) models, layers, and utilities for deep learning and control applications.
//!
//! # Features
//! - D-LinOSS and LinOSS layers
//! - Model configuration and parameterization
//! - Utilities for simulation, training, and analysis
//!
//! # Example
//! ```rust
//! use linoss_rust::linoss::dlinoss_layer::DLinossLayer;
//! // ...
//! ```

/// Visualization and GUI tools (optional, requires `visualization` feature)
#[cfg(feature = "visualization")]
pub mod visualization;

/// Core LinOSS models and layers
pub mod linoss;
/// Data utilities and dataset handling
pub mod data;
/// Model checkpointing and serialization
pub mod checkpoint;
/// Analysis and evaluation tools
pub mod analysis;

/// Convenient type alias for dynamic vectors
pub type Vector = nalgebra::DVector<f64>;
/// Convenient type alias for dynamic matrices
pub type Matrix = nalgebra::DMatrix<f64>;

/// Main error type for LinOSS operations
#[derive(Debug)]
pub enum LinossError {
    /// Dimension mismatch error
    DimensionMismatch(String),
    /// Other error
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

// Re-export main modules for convenience
pub use linoss::*;