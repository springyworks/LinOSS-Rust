#!/bin/bash

echo "🔍 D-LinOSS WASM Getrandom Diagnostic"
echo "====================================="

# Check current getrandom setup
echo ""
echo "1. Current getrandom dependency in Cargo.toml:"
echo "----------------------------------------------"
grep -A2 -B2 "getrandom" Cargo.toml || echo "No getrandom found in Cargo.toml"

echo ""
echo "2. Checking Cargo.lock for getrandom versions:"
echo "---------------------------------------------"
if [ -f Cargo.lock ]; then
    grep -A5 "name = \"getrandom\"" Cargo.lock || echo "No getrandom found in Cargo.lock"
else
    echo "No Cargo.lock found"
fi

echo ""
echo "3. Testing basic WASM target compilation:"
echo "----------------------------------------"
echo "Creating minimal test..."

# Create minimal test
cat > src/test_minimal.rs << 'EOF'
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn test_minimal() {
    // Just test basic WASM compilation
}
EOF

# Try to compile for WASM target
echo "Compiling for wasm32-unknown-unknown..."
if cargo check --target wasm32-unknown-unknown --lib 2>&1 | head -10; then
    echo "✅ Basic WASM compilation works"
else
    echo "❌ Basic WASM compilation failed"
fi

echo ""
echo "4. Testing getrandom specifically:"
echo "---------------------------------"

# Create a minimal getrandom test
cat > /tmp/getrandom_test.rs << 'EOF'
use getrandom::getrandom;

fn main() {
    let mut buf = [0u8; 32];
    getrandom(&mut buf).unwrap();
}
EOF

cat > /tmp/Cargo_getrandom.toml << 'EOF'
[package]
name = "getrandom-test"
version = "0.1.0"
edition = "2021"

[dependencies]
getrandom = { version = "0.2", features = ["js"] }
EOF

echo "Testing getrandom with WASM target..."
if cargo check --manifest-path /tmp/Cargo_getrandom.toml --target wasm32-unknown-unknown 2>&1; then
    echo "✅ getrandom WASM compatibility OK"
else
    echo "❌ getrandom WASM issue detected"
fi

# Cleanup
rm -f src/test_minimal.rs /tmp/getrandom_test.rs /tmp/Cargo_getrandom.toml

echo ""
echo "5. Recommended fixes:"
echo "--------------------"
echo "If getrandom fails:"
echo "  - Add: getrandom = { version = \"0.2\", features = [\"js\"] }"
echo "  - Set RUSTFLAGS: --cfg getrandom_backend=\"wasm_js\""
echo "  - Check local Burn clone for getrandom version conflicts"

echo ""
echo "6. Environment info:"
echo "-------------------"
echo "rustc version: $(rustc --version)"
echo "wasm-pack version: $(wasm-pack --version 2>/dev/null || echo 'not installed')"
echo "WASM target installed: $(rustup target list --installed | grep wasm32 || echo 'no')"
