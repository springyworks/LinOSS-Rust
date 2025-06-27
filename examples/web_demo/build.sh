#!/bin/bash
set -e

echo "=== Building D-LinOSS Web Demo with Burn WASM Support ==="

# Add WASM target
rustup target add wasm32-unknown-unknown

# Install wasm-pack if not already installed
if ! command -v wasm-pack &> /dev/null; then
    echo "Installing wasm-pack..."
    cargo install wasm-pack
fi

# Build mode selection (following copilot.md guidance)
BUILD_MODE=${1:-debug}

# Set RUSTFLAGS for getrandom WASM compatibility (force the configuration)
export RUSTFLAGS="--cfg getrandom_backend=\"wasm_js\" --cfg web_sys_unstable_apis"

case "$BUILD_MODE" in
    "debug")
        echo "Building DEBUG mode (fast ~2 seconds, large ~5MB)..."
        wasm-pack build --target web --dev --features linoss
        ;;
    "release")
        echo "Building RELEASE mode (slow ~12 seconds, small ~1.5MB)..."
        wasm-pack build --target web --release --features linoss
        ;;
    "minimal")
        echo "Building MINIMAL mode (no D-LinOSS, just Burn tensors)..."
        wasm-pack build --target web --dev --features minimal-burn
        ;;
    "burn")
        echo "Building BURN mode (Burn tensors + autodiff, no D-LinOSS)..."
        wasm-pack build --target web --dev --features burn
        ;;
    *)
        echo "Usage: $0 [debug|release|minimal|burn]"
        echo "  debug   - Fast build for development with full D-LinOSS (default)"
        echo "  release - Optimized build for production with full D-LinOSS"
        echo "  minimal - Basic Burn tensors without autodiff or D-LinOSS"
        echo "  burn    - Burn tensors + autodiff without D-LinOSS"
        exit 1
        ;;
esac

echo "✅ Build completed!"
echo "📁 WASM files generated in pkg/"
echo "🌐 Open index.html in a web server to test"
