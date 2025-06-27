#!/bin/bash
set -e

echo "=== Testing basic WASM build without Burn ==="

# Add WASM target
rustup target add wasm32-unknown-unknown

# Set RUSTFLAGS for getrandom WASM compatibility (force the configuration)
export RUSTFLAGS="--cfg getrandom_backend=\"wasm_js\" --cfg web_sys_unstable_apis"

# Build with no features (just basic WASM)
echo "Building with NO features (testing getrandom fix)..."
echo "RUSTFLAGS: $RUSTFLAGS"
wasm-pack build --target web --dev --no-default-features

echo "✅ Basic WASM build test completed!"
