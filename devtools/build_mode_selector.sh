#!/bin/bash
# Build mode selector for LinossRust development

set -e

echo "🦀 LinossRust Build Mode Selector"
echo "================================="
echo

# Function to time and build
time_build() {
    local profile=$1
    local description=$2
    echo "⏱️  Testing $description..."
    start_time=$(date +%s.%N)
    cargo build --profile "$profile" --quiet
    end_time=$(date +%s.%N)
    duration=$(echo "$end_time - $start_time" | bc -l)
    printf "   ✅ Completed in %.2f seconds\n" "$duration"
    
    # Check binary size
    if [ -f "target/$profile/examples/enhanced_brain_dynamics" ]; then
        size=$(ls -lh "target/$profile/examples/enhanced_brain_dynamics" | awk '{print $5}')
        echo "   📦 Binary size: $size"
    fi
    echo
}

# Clean previous builds for fair comparison
echo "🧹 Cleaning previous builds..."
cargo clean --quiet
echo

# Test different build modes
echo "📊 Build Performance Comparison:"
echo

time_build "dev" "Debug Build (fastest compilation)"
time_build "dev-opt" "Development Optimized (balanced)"
time_build "fast-release" "Fast Release (good optimization)"
time_build "release" "Full Release (maximum optimization)"

echo "🎯 Recommendations:"
echo "==================="
echo "• Use 'dev' for rapid iteration and debugging"
echo "• Use 'dev-opt' for development with some performance"
echo "• Use 'fast-release' for testing performance without long waits"
echo "• Use 'release' only for final builds and benchmarks"
echo
echo "💡 Quick commands:"
echo "   cargo build                    # Debug (fastest)"
echo "   cargo build --profile dev-opt  # Development optimized"
echo "   cargo build --profile fast-release  # Quick release"
echo "   cargo build --release          # Full optimization"
echo
