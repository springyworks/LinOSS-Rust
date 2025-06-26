#!/bin/bash

# 🔥 Burn Profiler MaxGraph Launch Script
# ======================================
# This script launches the complete live burn profiler system:
# 1. Burn Profiler Demo (D-LinOSS simulation)
# 2. WebSocket Bridge Server
# 3. Opens MaxGraph web interface

set -e

echo "🔥 Burn Profiler MaxGraph - Live Neural Dynamics Visualization"
echo "=============================================================="

# Check dependencies
echo "🔍 Checking dependencies..."

# Check if Rust/Cargo is available
if ! command -v cargo &> /dev/null; then
    echo "❌ Cargo not found. Please install Rust: https://rustup.rs/"
    exit 1
fi

# Check if mkfifo is available
if ! command -v mkfifo &> /dev/null; then
    echo "❌ mkfifo not found. This tool requires a Unix-like system with named pipes."
    exit 1
fi

# Build the project
echo "🔨 Building LinossRust project..."
cd "$(dirname "$0")"/..

if ! cargo build --release --example burn_profiler_demo --bin burn_profiler_bridge; then
    echo "❌ Build failed. Please check the error messages above."
    exit 1
fi

echo "✅ Build successful!"

# Function to cleanup background processes
cleanup() {
    echo ""
    echo "🧹 Cleaning up background processes..."
    jobs -p | xargs -r kill
    # Clean up HTTP server
    if [[ ! -z "$HTTP_SERVER_PID" ]]; then
        kill $HTTP_SERVER_PID 2>/dev/null || true
    fi
    # Clean up named pipe
    rm -f /tmp/dlinoss_brain_pipe
    echo "✅ Cleanup complete"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Start WebSocket bridge server in background
echo "🌐 Starting WebSocket bridge server..."
cargo run --release --bin burn_profiler_bridge &
BRIDGE_PID=$!

# Wait a moment for server to start
sleep 2

# Start the neural simulation in background
echo "🧠 Starting D-LinOSS neural dynamics simulation..."
cargo run --release --example burn_profiler_demo &
DEMO_PID=$!

# Wait another moment for pipe to be created
sleep 3

# Start HTTP server for proper web interface serving
echo "🌐 Starting HTTP server for web interface..."
cd ../typscr
python3 -m http.server 8081 &
HTTP_SERVER_PID=$!
cd - > /dev/null

# Wait for HTTP server to start
sleep 2

# Open the web interface in external browser (VS Code Simple Browser doesn't work)
echo "🎯 Opening MaxGraph web interface in external browser..."

WEB_URL="http://localhost:8081/burn_profiler_maxgraph.html"

if command -v xdg-open &> /dev/null; then
    xdg-open "$WEB_URL"
elif command -v open &> /dev/null; then
    open "$WEB_URL"
elif command -v firefox &> /dev/null; then
    firefox "$WEB_URL" &
elif command -v google-chrome &> /dev/null; then
    google-chrome "$WEB_URL" &
elif command -v chromium &> /dev/null; then
    chromium "$WEB_URL" &
else
    echo "⚠️ Could not automatically open browser."
    echo "   Please manually open: $WEB_URL"
fi

echo ""
echo "🎮 Live Burn Profiler is now running!"
echo "======================================"
echo "📡 WebSocket Server: ws://localhost:8080"
echo "🧠 Neural Simulation: Running with instrumentation"
echo "🌐 Web Interface: http://localhost:8081/burn_profiler_maxgraph.html"
echo "🌐 HTTP Server: localhost:8081"
echo ""
echo "📋 What you should see:"
echo "   • Real-time neural region activity visualization"
echo "   • Animated tensor operations around brain regions (rotating pink dots)"
echo "   • Live performance metrics (FPS, memory usage, operations)"
echo "   • Interactive network with dynamic node colors and connections"
echo ""
echo "✅ External browser should have opened automatically!"
echo "   (VS Code Simple Browser doesn't work - external browser required)"
echo ""
echo "⚠️  If the web interface doesn't connect automatically:"
echo "   1. Check that you see 3 colored circles with rotating dots"
echo "   2. Look for 'Connected' status in the control panel"
echo "   3. If issues persist, check browser console for errors"
echo ""
echo "Press Ctrl+C to stop all components..."

# Wait for background processes
wait

echo "🏁 All components have stopped."
