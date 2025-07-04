pub mod activation;
// pub mod burn_activations;  // Temporarily disabled - enum constraints
// pub mod simple_dlinoss;  // Temporarily disabled - tensor API issues
pub mod production_dlinoss;
pub mod block;
pub mod layer;
pub mod layer_fixed;
pub mod layers;
pub mod model;
pub mod vis_utils;
pub mod dlinoss_layer;
pub mod dlinoss_layer_optimized;
// pub mod dlinoss_block;  // Temporarily disabled - enum constraints
// pub mod dlinoss_model;  // Temporarily disabled - enum constraints
pub mod parallel_scan;

// Re-export core LinOSS components
pub use layer::{LinossLayer, LinossLayerConfig};
pub use layer_fixed::{FixedLinossLayer, FixedLinossLayerConfig};
pub use model::{FullLinossModel, LinossOutput};

// Re-export D-LinOSS components
pub use dlinoss_layer::{DLinossLayer, DLinossLayerConfig};
pub use dlinoss_layer_optimized::{OptimizedDLinossLayer, OptimizedDLinossConfig};

// Re-export production D-LinOSS (recommended for use)
pub use production_dlinoss::{
    ProductionDLinossConfig, ProductionSSMLayer, ProductionDLinossBlock, ProductionDLinossModel,
    ReluActivation, GeluActivation, TanhActivation, ActivationFunction,
    create_production_dlinoss_classifier, create_production_vanilla_linoss,
};

// Re-export activation functions
pub use activation::GLU;