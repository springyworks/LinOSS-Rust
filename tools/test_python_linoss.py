#!/usr/bin/env python3
"""
Simple Python test script for LinOSS analytical functions
This will help us understand the expected behavior for our Rust implementation
"""

import sys
import os

# Add the LinOSS repository to Python path
linoss_path = '/home/rustuser/pyth/linoss_kos'
if linoss_path not in sys.path:
    sys.path.insert(0, linoss_path)

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr

# Import LinOSS - suppress Pylance warnings since path is set dynamically
from models.LinOSS import LinOSSLayer  # type: ignore
print("Successfully imported LinOSSLayer")

def test_analytical_functions():
    """Test LinOSS on simple analytical functions"""
    
    # Test parameters (matching our Rust test)
    seq_len = 20
    d_model = 8
    key = jr.PRNGKey(42)
    
    # Create LinOSS layer
    linoss_layer = LinOSSLayer(
        ssm_size=d_model // 2,  # Half the model size for complex numbers
        H=1,  # Single input/output channel
        discretization='IM',
        key=key
    )
    
    # Generate test data (same as Rust)
    inputs = np.linspace(0, 1, seq_len)
    
    # Test functions
    test_cases = {
        'identity': inputs,
        'scaling': 2.0 * inputs, 
        'step': np.where(inputs > 0.5, 1.0, 0.0),
        'sine': np.sin(2 * np.pi * inputs),
        'exponential': np.exp(-2 * inputs),
        'quadratic': inputs ** 2
    }
    
    print("=== Python LinOSS Analytical Test ===")
    print(f"Input: {inputs[:5]}")
    
    for func_name, targets in test_cases.items():
        print(f"\n=== Testing {func_name} ===")
        print(f"Target: {targets[:5]}")
        
        # Reshape for LinOSS (L, H) format
        input_seq = inputs.reshape(-1, 1)
        target_seq = targets.reshape(-1, 1)
        
        # Forward pass
        outputs = linoss_layer(input_seq)
        
        # Compute loss
        loss = jnp.mean((outputs - target_seq) ** 2)
        
        print(f"Initial Output: {outputs[:5].flatten()}")
        print(f"Initial Loss: {loss:.6f}")
        
        # Note: This is just the initial forward pass without training
        # For a fair comparison, we'd need to implement training here too
        
if __name__ == "__main__":
    test_analytical_functions()
