# Reference Implementations

This directory contains references to the original Python implementations of LinOSS and dLinOSS for comparison and debugging purposes.

## GitHub Repositories

### LinOSS Python Repository
- **Repository**: [LinOSS Python Implementation](https://github.com/[username]/linoss-python)
- **Paper**: arXiv:2410.03943 - "Oscillatory State-Space Models"
- **Description**: Original Python implementation of Linear Oscillatory State-Space (LinOSS) models
- **Key Features**:
  - Core LinOSS layer implementation
  - Training loops and optimization
  - Example usage on various datasets
  - Mathematical formulation reference

### dLinOSS Python Repository  
- **Repository**: [dLinOSS Python Implementation](https://github.com/[username]/dlinoss-python)
- **Paper**: arXiv:2505.12171 - "Damped Linear Oscillatory State-Space Models"
- **Description**: Original Python implementation of Damped Linear Oscillatory State-Space (dLinOSS) models
- **Key Features**:
  - Damped oscillator formulation
  - A-matrix parameterization with damping
  - Enhanced stability for long sequences
  - Performance improvements over LinOSS

## Local Development Copies

The local workspace also contains development copies:
- `/home/rustuser/pyth/linoss_kos/` - LinOSS Python reference copy
- `/home/rustuser/pyth/dlinoss/` - dLinOSS Python reference copy

## How to Use These References

### 1. Clone the Repositories
```bash
# Clone LinOSS Python implementation
git clone https://github.com/[username]/linoss-python.git linoss-python/

# Clone dLinOSS Python implementation  
git clone https://github.com/[username]/dlinoss-python.git dlinoss-python/
```

### 2. Development Workflow
1. **Compare layer implementations** - Study the core LinOSS/dLinOSS layer logic
2. **Check initialization** - See how parameters are initialized in Python
3. **Examine training loops** - Compare learning rates, optimizers, loss functions
4. **Test on simple functions** - Validate behavior on identity, scaling, sine waves
5. **Debug specific issues** - Use Python implementations to validate expected behavior

### 3. Key Comparison Points
- Mathematical formulation differences
- Parameter initialization strategies  
- Training stability approaches
- Performance optimization techniques
- Test case implementations

## Purpose

These reference implementations help us:

1. **Debug our Rust implementation** - Compare exact algorithm details
2. **Understand the mathematical formulation** - See how the papers translate to code
3. **Validate our analytical tests** - Ensure our test functions match the expected behavior
4. **Fix initialization and training issues** - Copy working hyperparameters and initialization strategies

## Usage

After cloning the repositories:

1. **Compare layer implementations** - Look at the core LinOSS/dLinOSS layer logic
2. **Check initialization** - See how parameters are initialized 
3. **Examine training loops** - Compare learning rates, optimizers, loss functions
4. **Test on simple functions** - See how they handle identity, scaling, sine waves, etc.
5. **Debug specific issues** - Use Python implementations to validate expected behavior

## Key Files to Check

### LinOSS Python Implementation
- Core layer implementation
- Model architecture 
- Training loops
- Example usage
- Test cases

### dLinOSS Python Implementation  
- Damped oscillator formulation
- A-matrix parameterization
- Damping coefficient handling
- Training stability

## Research Papers & Citations

### LinOSS (Linear Oscillatory State-Space Models)
```bibtex
@article{linoss2024,
    title={Oscillatory State-Space Models}, 
    author={[Authors]},
    journal={arXiv preprint arXiv:2410.03943},
    year={2024}
}
```

### dLinOSS (Damped Linear Oscillatory State-Space Models)
```bibtex
@article{dlinoss2024,
    title={Damped Linear Oscillatory State-Space Models},
    author={[Authors]}, 
    journal={arXiv preprint arXiv:2505.12171},
    year={2024}
}
```

## Integration with LinossRust

This Rust implementation aims to:
- **Replicate the mathematical formulations** from both papers
- **Optimize performance** using Rust and the Burn deep learning framework
- **Extend functionality** with real-time visualization and profiling
- **Maintain compatibility** with the Python reference implementations

## Validation Strategy

1. **Unit Tests**: Compare individual layer outputs with Python implementations
2. **Integration Tests**: Validate full model training on simple analytical functions  
3. **Performance Tests**: Benchmark against Python implementations
4. **Numerical Accuracy**: Ensure mathematical equivalence within floating-point precision
