#!/bin/bash

echo "=== LinOSS Rust Examples Comprehensive Test ==="
echo "Testing compilation and execution of all examples"
echo

# Quick compile test examples
quick_examples=(
    "basic_usage"
    "sine_wave_visualization"
    "compare_scan_methods"
    "parallel_scan_test"
)

# TUI examples (run briefly then exit)
tui_examples=(
    "chaotic_2d_linoss"
    "damped_sine_response" 
    "flyLinoss"
    "train_linoss"
)

echo "=== Testing Quick Examples ==="
for example in "${quick_examples[@]}"; do
    echo -n "Testing $example... "
    timeout 10s cargo run --example "$example" --features ndarray_backend > /tmp/test_$example.txt 2>&1
    exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "✓ PASS"
    elif [ $exit_code -eq 124 ]; then
        echo "⚠ TIMEOUT (possibly still running)"
    else
        echo "✗ FAIL (exit code: $exit_code)"
        echo "Error output:"
        cat /tmp/test_$example.txt | head -5 | sed 's/^/  /'
    fi
done

echo
echo "=== Testing TUI Examples (Brief Run) ==="
for example in "${tui_examples[@]}"; do
    echo -n "Testing $example... "
    timeout 3s cargo run --example "$example" --features ndarray_backend > /tmp/test_$example.txt 2>&1
    exit_code=$?
    
    # For TUI examples, timeout is expected (means they started successfully)
    if [ $exit_code -eq 124 ]; then
        echo "✓ PASS (runs, exited on timeout)"
    elif [ $exit_code -eq 0 ]; then
        echo "✓ PASS (completed quickly)"
    else
        echo "✗ FAIL (exit code: $exit_code)"
        echo "Error output:"
        cat /tmp/test_$example.txt | head -5 | sed 's/^/  /'
    fi
done

echo
echo "=== Testing Compilation Only ==="
echo -n "Compiling all examples... "
cargo check --examples --features ndarray_backend > /tmp/compile_test.txt 2>&1
if [ $? -eq 0 ]; then
    echo "✓ ALL EXAMPLES COMPILE"
else
    echo "✗ COMPILATION ERRORS"
    cat /tmp/compile_test.txt
fi

echo
echo "=== Testing with WGPU Backend ==="
echo -n "Testing sine_wave_visualization with WGPU... "
timeout 10s cargo run --example sine_wave_visualization --features wgpu_backend > /tmp/test_wgpu.txt 2>&1
if [ $? -eq 0 ]; then
    echo "✓ WGPU BACKEND WORKS"
else
    echo "⚠ WGPU ISSUES (check /tmp/test_wgpu.txt)"
fi

echo
echo "=== Summary ==="
echo "LinOSS examples test completed. Check individual results above."
