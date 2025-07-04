//! Enhanced Activation Functions for D-LinOSS using Burn Framework
//! 
//! This module showcases Burn's comprehensive activation function library
//! and demonstrates automatic kernel fusion capabilities for optimal performance.

use burn::{
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig, Relu, Gelu, SwiGlu, SwiGluConfig, LeakyRelu, LeakyReluConfig, PRelu, PReluConfig},
    tensor::{backend::Backend, Tensor, activation},
};

/// Enhanced GLU implementation using Burn's activation functions
#[derive(Module, Debug)]
pub struct BurnGLU<B: Backend> {
    linear: Linear<B>,
}

impl<B: Backend> BurnGLU<B> {
    pub fn new(input_features: usize, output_features: usize, device: &B::Device) -> Self {
        let linear = LinearConfig::new(input_features, output_features * 2).init(device);
        Self { linear }
    }

    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let projected = self.linear.forward(input);
        burn_glu(projected)
    }
}

/// GLU function using Burn's tensor operations
pub fn burn_glu<B: Backend, const D: usize>(input: Tensor<B, D>) -> Tensor<B, D> {
    let last_dim = input.dims()[D - 1];
    let output_size = last_dim / 2;
    
    let a = input.clone().narrow(D - 1, 0, output_size);
    let b = input.narrow(D - 1, output_size, output_size);
    
    // GLU: a * sigmoid(b)
    a * activation::sigmoid(b)
}

/// Comprehensive activation function showcase for Burn framework
#[derive(Module, Debug)]
pub enum BurnActivation<B: Backend> {
    /// Rectified Linear Unit - Simple and effective
    Relu(Relu),
    /// Gaussian Error Linear Unit - Smooth and differentiable
    Gelu(Gelu),
    /// Leaky Rectified Linear Unit - Prevents dead neurons
    LeakyRelu(LeakyRelu),
    /// Parametric Rectified Linear Unit - Learnable negative slope
    PRelu(PRelu<B>),
    /// Swish Gated Linear Unit - State-of-the-art for transformers
    SwiGlu(SwiGlu<B>),
    /// Hyperbolic Tangent - Classic bounded activation
    Tanh,
    /// Sigmoid - Bounded between 0 and 1
    Sigmoid,
    /// Swish/SiLU - Smooth and unbounded above
    Silu,
    /// Mish - Self-regularized smooth activation
    Mish,
    /// Hard Sigmoid - Computationally efficient sigmoid approximation
    HardSigmoid,
    /// Custom GELU - Demonstrates Burn's kernel fusion
    CustomGelu,
    /// Fused Complex Activation - Shows advanced optimization
    FusedComplex,
}

impl<B: Backend> BurnActivation<B> {
    /// Create ReLU activation (recommended for vanilla LinOSS)
    pub fn relu() -> Self {
        Self::Relu(Relu::new())
    }
    
    /// Create GELU activation (recommended for D-LinOSS)
    pub fn gelu() -> Self {
        Self::Gelu(Gelu::new())
    }
    
    /// Create Leaky ReLU with custom negative slope
    pub fn leaky_relu(negative_slope: f64, device: &B::Device) -> Self {
        let config = LeakyReluConfig::new().with_negative_slope(negative_slope);
        Self::LeakyRelu(config.init(device))
    }
    
    /// Create Parametric ReLU with learnable parameters
    pub fn prelu(num_parameters: usize, device: &B::Device) -> Self {
        let config = PReluConfig::new(num_parameters);
        Self::PRelu(config.init(device))
    }
    
    /// Create SwiGLU activation (state-of-the-art for attention mechanisms)
    pub fn swiglu(input_dim: usize, output_dim: usize, device: &B::Device) -> Self {
        let config = SwiGluConfig::new(input_dim, output_dim);
        Self::SwiGlu(config.init(device))
    }
    
    /// Create Tanh activation
    pub fn tanh() -> Self {
        Self::Tanh
    }
    
    /// Create Sigmoid activation
    pub fn sigmoid() -> Self {
        Self::Sigmoid
    }
    
    /// Create Swish/SiLU activation
    pub fn silu() -> Self {
        Self::Silu
    }
    
    /// Create Mish activation
    pub fn mish() -> Self {
        Self::Mish
    }
    
    /// Create Hard Sigmoid activation
    pub fn hard_sigmoid() -> Self {
        Self::HardSigmoid
    }
    
    /// Create custom GELU demonstrating kernel fusion
    pub fn custom_gelu() -> Self {
        Self::CustomGelu
    }
    
    /// Create fused complex activation
    pub fn fused_complex() -> Self {
        Self::FusedComplex
    }
    
    /// Forward pass through the chosen activation function
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        match self {
            Self::Relu(relu) => relu.forward(input),
            Self::Gelu(gelu) => gelu.forward(input),
            Self::LeakyRelu(leaky_relu) => leaky_relu.forward(input),
            Self::PRelu(prelu) => prelu.forward(input),
            Self::SwiGlu(swiglu) => swiglu.forward(input),
            Self::Tanh => activation::tanh(input),
            Self::Sigmoid => activation::sigmoid(input),
            Self::Silu => activation::silu(input),
            Self::Mish => activation::mish(input),
            Self::HardSigmoid => activation::hard_sigmoid(input),
            Self::CustomGelu => custom_gelu_kernel_fusion(input),
            Self::FusedComplex => fused_complex_activation(input),
        }
    }
}

/// Custom GELU implementation showcasing Burn's automatic kernel fusion
/// 
/// This function demonstrates how Burn automatically optimizes multiple tensor operations
/// into a single, efficient kernel when possible, providing significant performance benefits.
pub fn custom_gelu_kernel_fusion<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    // GELU(x) = x * Φ(x) where Φ(x) is the CDF of standard normal distribution
    // High-precision approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    
    let sqrt_2_over_pi = (2.0 / std::f64::consts::PI).sqrt();
    let gelu_coefficient = 0.044715;
    
    // These operations will be automatically fused by Burn for optimal performance
    let x_cubed = x.clone().powf_scalar(3.0);
    let polynomial_term = x.clone() + x_cubed * gelu_coefficient;
    let scaled_term = polynomial_term * sqrt_2_over_pi;
    let tanh_result = activation::tanh(scaled_term);
    let cdf_approximation = (tanh_result + 1.0) * 0.5;
    
    x * cdf_approximation
}

/// Demonstrates advanced kernel fusion with a complex multi-component activation
/// 
/// This showcases Burn's ability to fuse complex operations including:
/// - Multiple activation functions
/// - Polynomial operations  
/// - Weighted combinations
/// - All optimized into efficient kernels automatically
pub fn fused_complex_activation<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    // Complex activation: α₁*x*sigmoid(x) + α₂*tanh(βx) + α₃*x³ + α₄*GELU(x)
    // Burn will automatically fuse these operations for maximum efficiency
    
    let alpha1 = 0.7;
    let alpha2 = 0.2;
    let alpha3 = 0.05;
    let alpha4 = 0.05;
    let beta = 0.5;
    
    // Component 1: Swish-like (x * sigmoid(x))
    let swish_component = x.clone() * activation::sigmoid(x.clone()) * alpha1;
    
    // Component 2: Scaled tanh
    let tanh_component = activation::tanh(x.clone() * beta) * alpha2;
    
    // Component 3: Cubic term for high-frequency modeling
    let cubic_component = x.clone().powf_scalar(3.0) * alpha3;
    
    // Component 4: GELU for smooth gradients
    let gelu_component = activation::gelu(x) * alpha4;
    
    // Burn will optimize this entire computation chain
    swish_component + tanh_component + cubic_component + gelu_component
}

/// Performance benchmark function demonstrating Burn's optimization capabilities
/// 
/// This function can be used to compare the performance of different activation functions
/// and see how Burn's automatic kernel fusion improves execution speed.
pub fn activation_benchmark<B: Backend>(
    input: Tensor<B, 3>,
    activation_type: &str,
    device: &B::Device,
) -> Tensor<B, 3> {
    match activation_type {
        "relu" => {
            let activation = BurnActivation::<B>::relu();
            activation.forward(input)
        },
        "gelu" => {
            let activation = BurnActivation::<B>::gelu();
            activation.forward(input)
        },
        "custom_gelu" => {
            let activation = BurnActivation::<B>::custom_gelu();
            activation.forward(input)
        },
        "fused_complex" => {
            let activation = BurnActivation::<B>::fused_complex();
            activation.forward(input)
        },
        "swiglu" => {
            let hidden_dim = input.dims()[2];
            let activation = BurnActivation::<B>::swiglu(hidden_dim, hidden_dim, device);
            activation.forward(input)
        },
        _ => {
            // Default to GELU
            let activation = BurnActivation::<B>::gelu();
            activation.forward(input)
        }
    }
}

/// Configuration for selecting optimal activation functions for different D-LinOSS variants
#[derive(Config, Debug)]
pub struct ActivationConfig {
    /// Primary activation function name
    pub primary_activation: String,
    /// GLU variant ("standard", "swiglu", "burn_glu")
    pub glu_variant: String,
    /// Leaky ReLU negative slope (if applicable)
    pub leaky_slope: f64,
    /// Enable custom optimizations
    pub enable_fusion: bool,
    /// Performance mode ("speed", "memory", "balanced")
    pub performance_mode: String,
}

impl ActivationConfig {
    /// Optimal configuration for D-LinOSS models
    pub fn dlinoss_optimal() -> Self {
        Self {
            primary_activation: "gelu".to_string(),
            glu_variant: "swiglu".to_string(),
            leaky_slope: 0.01,
            enable_fusion: true,
            performance_mode: "speed".to_string(),
        }
    }
    
    /// Configuration for vanilla LinOSS models
    pub fn vanilla_linoss() -> Self {
        Self {
            primary_activation: "relu".to_string(),
            glu_variant: "standard".to_string(),
            leaky_slope: 0.01,
            enable_fusion: true,
            performance_mode: "balanced".to_string(),
        }
    }
    
    /// Experimental high-performance configuration
    pub fn experimental() -> Self {
        Self {
            primary_activation: "fused_complex".to_string(),
            glu_variant: "burn_glu".to_string(),
            leaky_slope: 0.02,
            enable_fusion: true,
            performance_mode: "speed".to_string(),
        }
    }
    
    /// Memory-efficient configuration for large models
    pub fn memory_efficient() -> Self {
        Self {
            primary_activation: "relu".to_string(),
            glu_variant: "standard".to_string(),
            leaky_slope: 0.01,
            enable_fusion: false,
            performance_mode: "memory".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::{backend::ndarray::NdArray, tensor::Distribution};

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_burn_glu() {
        let device = Default::default();
        let glu = BurnGLU::<TestBackend>::new(64, 32, &device);
        
        let input = Tensor::<TestBackend, 3>::random([2, 10, 64], Distribution::Normal(0.0, 1.0), &device);
        let output = glu.forward(input);
        
        assert_eq!(output.dims(), [2, 10, 32]);
        println!("✓ Burn GLU test passed");
    }

    #[test]
    fn test_all_burn_activations() {
        let device = Default::default();
        let input = Tensor::<TestBackend, 2>::random([4, 16], Distribution::Normal(0.0, 1.0), &device);
        
        // Test all activation functions
        let activations = [
            BurnActivation::<TestBackend>::relu(),
            BurnActivation::<TestBackend>::gelu(),
            BurnActivation::<TestBackend>::leaky_relu(0.01, &device),
            BurnActivation::<TestBackend>::tanh(),
            BurnActivation::<TestBackend>::sigmoid(),
            BurnActivation::<TestBackend>::silu(),
            BurnActivation::<TestBackend>::mish(),
            BurnActivation::<TestBackend>::hard_sigmoid(),
            BurnActivation::<TestBackend>::custom_gelu(),
            BurnActivation::<TestBackend>::fused_complex(),
        ];
        
        for (i, activation) in activations.iter().enumerate() {
            let output = activation.forward(input.clone());
            assert_eq!(output.dims(), input.dims());
            println!("✓ Activation {} test passed", i);
        }
        
        println!("✓ All Burn activation functions tested successfully!");
    }
    
    #[test]
    fn test_kernel_fusion_demo() {
        let device = Default::default();
        let input = Tensor::<TestBackend, 3>::random([2, 8, 32], Distribution::Normal(0.0, 1.0), &device);
        
        // Test custom GELU with kernel fusion
        let custom_output = custom_gelu_kernel_fusion(input.clone());
        assert_eq!(custom_output.dims(), input.dims());
        
        // Test fused complex activation
        let fused_output = fused_complex_activation(input.clone());
        assert_eq!(fused_output.dims(), input.dims());
        
        println!("✓ Kernel fusion demo test passed - Burn optimizes these automatically!");
    }
    
    #[test]
    fn test_activation_benchmark() {
        let device = Default::default();
        let input = Tensor::<TestBackend, 3>::random([4, 16, 64], Distribution::Normal(0.0, 1.0), &device);
        
        let test_cases = ["relu", "gelu", "custom_gelu", "fused_complex"];
        
        for activation_type in test_cases.iter() {
            let output = activation_benchmark(input.clone(), activation_type, &device);
            assert_eq!(output.dims(), input.dims());
            println!("✓ Benchmark test for {} passed", activation_type);
        }
    }
    
    #[test]
    fn test_activation_configs() {
        let dlinoss_config = ActivationConfig::dlinoss_optimal();
        let vanilla_config = ActivationConfig::vanilla_linoss();
        let experimental_config = ActivationConfig::experimental();
        let memory_config = ActivationConfig::memory_efficient();
        
        assert_eq!(dlinoss_config.primary_activation, "gelu");
        assert_eq!(vanilla_config.primary_activation, "relu");
        assert_eq!(experimental_config.primary_activation, "fused_complex");
        assert_eq!(memory_config.performance_mode, "memory");
        
        println!("✓ All activation configurations tested successfully!");
    }
    
    #[test]
    fn test_gelu_variants() {
        let device = Default::default();
        let input = Tensor::<TestBackend, 2>::random([4, 8], Distribution::Normal(0.0, 1.0), &device);
        
        // Compare built-in GELU with custom implementation
        let builtin_output = activation::gelu(input.clone());
        let custom_output = custom_gelu_kernel_fusion(input);
        
        // Both should have same dimensions
        assert_eq!(builtin_output.dims(), custom_output.dims());
        
        println!("✓ GELU variants comparison test passed");
        println!("  Built-in GELU: Highly optimized by Burn framework");
        println!("  Custom GELU: Demonstrates kernel fusion capabilities");
    }
}
