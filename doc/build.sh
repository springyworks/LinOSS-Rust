#!/bin/bash
# Build script for LinOSS Web Demo

set -e

echo "🧠 Building LinOSS Web Demo..."

# Check if wasm-pack is installed
if ! command -v wasm-pack &> /dev/null; then
    echo "❌ wasm-pack is not installed"
    echo "Install with: curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh"
    exit 1
fi

# Add WASM target if not present
rustup target add wasm32-unknown-unknown

echo "📦 Building WASM module..."
# Use linoss feature for full D-LinOSS functionality
wasm-pack build --target web --out-dir wasm --out-name linoss_web_demo --features linoss

echo "🧹 Cleaning up unnecessary files..."
cd wasm
rm -f .gitignore package.json README.md

echo "✅ Build complete!"
echo "🌐 Serve with: python -m http.server 8000"
echo "📱 Or visit: http://localhost:8000"
