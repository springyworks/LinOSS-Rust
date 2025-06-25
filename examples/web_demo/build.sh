#!/bin/bash

# Build script for D-LinOSS WASM Demo
echo "🔥 Building D-LinOSS WASM Demo..."

# Check if wasm-pack is installed
if ! command -v wasm-pack &> /dev/null; then
    echo "❌ wasm-pack not found. Installing..."
    curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
fi

# Check if we have the wasm32-unknown-unknown target
if ! rustup target list --installed | grep -q "wasm32-unknown-unknown"; then
    echo "📦 Installing wasm32-unknown-unknown target..."
    rustup target add wasm32-unknown-unknown
fi

# Build the WASM package
echo "🔨 Building WASM package..."

# Set optimization flags like Burn does
export RUSTFLAGS="-C embed-bitcode=yes -C codegen-units=1 -C opt-level=3 --cfg web_sys_unstable_apis --cfg getrandom_backend=\"wasm_js\""

wasm-pack build --target web --out-dir pkg --dev

if [ $? -eq 0 ]; then
    echo "✅ WASM build successful!"
    echo "🌐 Starting development server..."
    
    # Start a simple HTTP server
    if command -v python3 &> /dev/null; then
        python3 -m http.server 8083
    elif command -v python &> /dev/null; then
        python -m http.server 8083
    else
        echo "❌ Python not found. Please install Python or serve the files manually."
        echo "   The WASM demo is ready in the current directory."
        echo "   Open index.html in a web server to view the demo."
    fi
else
    echo "❌ WASM build failed. Please check the error messages above."
    exit 1
fi
