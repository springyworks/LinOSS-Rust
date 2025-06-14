# Reference Implementations

This directory contains the original Python implementations of LinOSS and dLinOSS for comparison and debugging purposes.

## Structure

- `linoss-python/` - Original LinOSS Python implementation (arXiv:2410.03943)
- `dlinoss-python/` - Original dLinOSS Python implementation (arXiv:2505.12171)

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

## Next Steps

1. Clone the repositories into this directory
2. Run their example scripts to understand expected behavior
3. Create minimal Python test scripts for our analytical functions
4. Compare outputs with our Rust implementation
5. Fix the identified issues in our Rust code
