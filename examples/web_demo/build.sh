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

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf pkg/

# Determine build mode
BUILD_MODE=${1:-basic}

case $BUILD_MODE in
    "burn")
        echo "🔥 Building WASM package with Burn support..."
        echo "   Using local Burn clone: /home/rustuser/projects/from_github/burn"
        
        # Build with Burn feature enabled (skip wasm-opt optimization)
        wasm-pack build --target web --out-dir pkg --features burn --no-opt
        ;;
    "minimal-burn")
        echo "🔥 Building WASM package with minimal Burn support..."
        echo "   Using local Burn clone: /home/rustuser/projects/from_github/burn"
        
        # Build with minimal Burn feature enabled (skip wasm-opt optimization)
        wasm-pack build --target web --out-dir pkg --features minimal-burn --no-opt
        ;;
    "basic")
        echo "🔧 Building basic WASM package (no Burn)..."
        
        # Build without Burn (skip wasm-opt optimization)
        wasm-pack build --target web --out-dir pkg --no-opt
        ;;
    *)
        echo "❌ Unknown build mode: $BUILD_MODE"
        echo "   Usage: $0 [basic|burn|minimal-burn]"
        exit 1
        ;;
esac

if [ $? -eq 0 ]; then
    echo "✅ WASM build successful! ($BUILD_MODE mode)"
    echo "🌐 Starting development server..."
    
    # Update index.html based on build mode
    if [ "$BUILD_MODE" = "burn" ]; then
        echo "📝 Creating Burn-enabled index.html..."
        cat > index_burn.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>D-LinOSS Burn WASM Demo</title>
    <style>
        html, body { margin: 0; padding: 0; height: 100%; }
        canvas { display: block; width: 100%; height: 100%; background: #1a1a1a; }
        .loading { 
            position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
            color: white; font-family: Arial, sans-serif; 
        }
    </style>
</head>
<body>
    <div class="loading" id="loading">🔥 Loading D-LinOSS Burn Neural Dynamics...</div>
    <canvas id="linoss_canvas"></canvas>
    
    <script type="module">
        import init, { main_burn } from './pkg/linoss_web_demo.js';
        
        async function run() {
            await init();
            document.getElementById('loading').style.display = 'none';
            main_burn();
        }
        
        run().catch(console.error);
    </script>
</body>
</html>
EOF
    elif [ "$BUILD_MODE" = "minimal-burn" ]; then
        echo "📝 Creating minimal Burn index.html..."
        cat > index_minimal_burn.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Minimal Burn Tensor WASM Test</title>
    <style>
        html, body { margin: 0; padding: 0; height: 100%; }
        canvas { display: block; width: 100%; height: 100%; background: #1a1a1a; }
        .loading { 
            position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
            color: white; font-family: Arial, sans-serif; 
        }
    </style>
</head>
<body>
    <div class="loading" id="loading">🔥 Loading Minimal Burn Tensor Test...</div>
    <canvas id="linoss_canvas"></canvas>
    
    <script type="module">
        import init, { main_minimal_burn } from './pkg/linoss_web_demo.js';
        
        async function run() {
            await init();
            document.getElementById('loading').style.display = 'none';
            main_minimal_burn();
        }
        
        run().catch(console.error);
    </script>
</body>
</html>
EOF
        echo "   Created index_minimal_burn.html for minimal Burn test"
    else
        echo "📝 Using basic index.html..."
    fi
    
    # Start a simple HTTP server
    if command -v python3 &> /dev/null; then
        echo "🚀 Starting server on http://localhost:8083"
        echo "   Basic demo: http://localhost:8083/index.html"
        if [ "$BUILD_MODE" = "burn" ]; then
            echo "   Burn demo:  http://localhost:8083/index_burn.html"
        elif [ "$BUILD_MODE" = "minimal-burn" ]; then
            echo "   Minimal Burn demo: http://localhost:8083/index_minimal_burn.html"
        fi
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
    echo "💡 Common issues:"
    echo "   - getrandom dependency conflicts (check Cargo.toml)"
    echo "   - Local Burn clone path: /home/rustuser/projects/from_github/burn"
    echo "   - Try: cargo clean && $0 basic"
    exit 1
fi
