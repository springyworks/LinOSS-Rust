#!/bin/bash

# D-LinOSS Web Demo Server
echo "🧠 D-LinOSS Web Demo Server"
echo "=========================="

# Check if we're in the right directory
if [ ! -f "pkg/linoss_web_demo.js" ]; then
    echo "❌ Error: pkg/linoss_web_demo.js not found!"
    echo "Please run ./build.sh first to build the WASM module"
    exit 1
fi

# Show available files
echo "📁 Available demo file:"
echo "  - index.html (D-LinOSS neural dynamics demo)"
echo ""

# Default to index.html
DEFAULT_FILE="index.html"
PORT=${1:-8000}

echo "🚀 Starting server on port $PORT"
echo "📂 Serving from: $(pwd)"
echo "🌐 Open: http://localhost:$PORT"
echo "🏠 Default page: $DEFAULT_FILE"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start Python HTTP server
if command -v python3 >/dev/null 2>&1; then
    python3 -m http.server $PORT
elif command -v python >/dev/null 2>&1; then
    python -m http.server $PORT
else
    echo "❌ Error: Python not found! Please install Python to run the server."
    exit 1
fi