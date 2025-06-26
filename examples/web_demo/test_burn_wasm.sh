#!/bin/bash

echo "🧪 Testing D-LinOSS WASM + Burn Integration"
echo "=========================================="

# Function to test build and report results
test_build() {
    local mode=$1
    echo ""
    echo "🔧 Testing $mode build..."
    echo "-------------------------"
    
    # Clean first
    cargo clean --target-dir target/
    rm -rf pkg/
    
    # Attempt build
    if ./build.sh $mode 2>&1 | head -20; then
        echo "✅ $mode build completed"
        
        # Check if WASM files were generated
        if [ -f "pkg/linoss_web_demo.js" ] && [ -f "pkg/linoss_web_demo_bg.wasm" ]; then
            echo "✅ WASM artifacts generated successfully"
            
            # Check file sizes
            local js_size=$(stat -c%s "pkg/linoss_web_demo.js" 2>/dev/null || echo "unknown")
            local wasm_size=$(stat -c%s "pkg/linoss_web_demo_bg.wasm" 2>/dev/null || echo "unknown")
            echo "   JS size: $js_size bytes"
            echo "   WASM size: $wasm_size bytes"
            
            return 0
        else
            echo "❌ WASM artifacts not found"
            return 1
        fi
    else
        echo "❌ $mode build failed"
        return 1
    fi
}

# Function to check getrandom issue specifically
check_getrandom() {
    echo ""
    echo "🔍 Checking getrandom dependency..."
    echo "--------------------------------"
    
    # Try to compile just the getrandom features
    echo '[dependencies]' > /tmp/test_getrandom.toml
    echo 'getrandom = { version = "0.2", features = ["js"] }' >> /tmp/test_getrandom.toml
    
    if cargo check --target wasm32-unknown-unknown --manifest-path /tmp/test_getrandom.toml 2>/dev/null; then
        echo "✅ getrandom WASM compatibility OK"
    else
        echo "❌ getrandom WASM compatibility issue detected"
        echo "💡 This is the classic Burn+WASM problem!"
    fi
    
    rm -f /tmp/test_getrandom.toml
}

# Function to verify Burn clone
check_burn_clone() {
    echo ""
    echo "🔍 Checking Burn clone..."
    echo "------------------------"
    
    local burn_path="/home/rustuser/projects/from_github/burn"
    
    if [ -d "$burn_path" ]; then
        echo "✅ Burn clone found at $burn_path"
        
        if [ -f "$burn_path/crates/burn/Cargo.toml" ]; then
            echo "✅ Burn main crate structure OK"
            
            # Check version
            local version=$(grep '^version' "$burn_path/crates/burn/Cargo.toml" | head -1 | cut -d'"' -f2)
            echo "   Version: $version"
        else
            echo "❌ Burn crate structure incorrect"
        fi
    else
        echo "❌ Burn clone not found at $burn_path"
        echo "💡 Clone with: git clone https://github.com/tracel-ai/burn.git"
    fi
}

# Main test sequence
echo "Starting comprehensive test..."

# 1. Check prerequisites
check_burn_clone
check_getrandom

# 2. Test basic build (should always work)
echo ""
echo "🎯 Phase 1: Basic Build Test"
echo "============================"
if test_build "basic"; then
    BASIC_OK=true
else
    BASIC_OK=false
fi

# 3. Test Burn build (this is what we're investigating)
echo ""
echo "🎯 Phase 2: Burn Build Test"
echo "==========================="
if test_build "burn"; then
    BURN_OK=true
else
    BURN_OK=false
fi

# 4. Results summary
echo ""
echo "📊 Test Results Summary"
echo "======================"
echo "Basic build: $([ $BASIC_OK = true ] && echo "✅ PASS" || echo "❌ FAIL")"
echo "Burn build:  $([ $BURN_OK = true ] && echo "✅ PASS" || echo "❌ FAIL")"

if [ $BURN_OK = true ]; then
    echo ""
    echo "🎉 SUCCESS! Burn+WASM integration working!"
    echo "   You can now run: ./build.sh burn"
    echo "   Then open: http://localhost:8083/index_burn.html"
elif [ $BASIC_OK = true ]; then
    echo ""
    echo "⚠️  Partial success: Basic WASM works, Burn has issues"
    echo "   This confirms the getrandom dependency problem"
    echo "   Solutions to try:"
    echo "   1. Update Burn clone: cd $burn_path && git pull"
    echo "   2. Check burn/Cargo.toml getrandom features"
    echo "   3. Try different Burn backend features"
else
    echo ""
    echo "❌ Both builds failed - fundamental WASM setup issue"
    echo "   Check wasm-pack installation and target setup"
fi

# Cleanup
rm -rf pkg/ target/
echo ""
echo "🧹 Cleanup completed"
