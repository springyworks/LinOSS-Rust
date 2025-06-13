#!/bin/bash
# filepath: /home/rustuser/rustdev/LinossRust/datastore/sythetic-datasets/examine_traces.sh
# Utility script to examine the execution traces and model parameters

echo "🔍 Examining Synthetic Dataset Traces"
echo "===================================="

# Check synthetic dataset files
echo ""
echo "📊 Synthetic Dataset Files:"
echo "---------------------------"
dataset_count=0
for csv_file in *.csv; do
    if [ -f "$csv_file" ]; then
        dataset_name=$(basename "$csv_file" .csv)
        echo "📄 $dataset_name dataset:"
        
        # Extract info from CSV header
        if grep -q "Samples:" "$csv_file"; then
            info_line=$(grep "Samples:" "$csv_file")
            echo "   └─ $info_line" | sed 's/# //'
        fi
        
        # Check for corresponding files
        files_found=""
        for ext in "_features.npy" "_labels.npy" "_info.md"; do
            if [ -f "${dataset_name}${ext}" ]; then
                files_found="$files_found ${ext#_}"
            fi
        done
        echo "   └─ Files: $files_found"
        
        dataset_count=$((dataset_count + 1))
    fi
done

if [ $dataset_count -eq 0 ]; then
    echo "   No synthetic dataset files found"
fi

# Check execution traces
echo ""
echo "📝 Execution Traces:"
echo "-------------------"
if [ -d "execution-traces" ]; then
    for trace_file in execution-traces/*.md; do
        if [ -f "$trace_file" ]; then
            timestamp=$(basename "$trace_file" .md | sed 's/test_execution_//')
            echo "📄 $trace_file (Unix timestamp: $timestamp)"
            
            # Show summary from the trace
            if grep -q "Success Rate" "$trace_file"; then
                success_rate=$(grep "Success Rate" "$trace_file" | sed 's/.*Success Rate.*: //')
                total_tests=$(grep "Total Tests" "$trace_file" | sed 's/.*Total Tests.*: //')
                passed_tests=$(grep "Passed" "$trace_file" | sed 's/.*Passed.*: //')
                echo "   └─ Tests: $passed_tests/$total_tests passed ($success_rate)"
            fi
        fi
    done
else
    echo "   No execution traces directory found"
fi

# Check sample model parameters
echo ""
echo "💾 Sample Trained Model Parameters:"
echo "-----------------------------------"
sample_count=0
for model_file in sample_trained_model_*.bin; do
    if [ -f "$model_file" ]; then
        timestamp=$(basename "$model_file" .bin | sed 's/sample_trained_model_//')
        file_size=$(stat -f%z "$model_file" 2>/dev/null || stat -c%s "$model_file" 2>/dev/null || echo "unknown")
        echo "🧠 $model_file (Unix timestamp: $timestamp, Size: ${file_size} bytes)"
        sample_count=$((sample_count + 1))
    fi
done

if [ $sample_count -eq 0 ]; then
    echo "   No sample model parameter files found"
fi

# Summary
echo ""
echo "📊 Summary:"
echo "-----------"
trace_count=$(find execution-traces -name "*.md" 2>/dev/null | wc -l || echo 0)
echo "📝 Execution traces: $trace_count"
echo "🧠 Sample model files: $sample_count"
echo "📊 Synthetic datasets: $dataset_count"

echo ""
echo "💡 Usage:"
echo "  - View trace details: cat execution-traces/test_execution_<timestamp>.md"
echo "  - Load model parameters in Rust: load_modelparams(..., \"sample_trained_model_<timestamp>.bin\")"
echo "  - Load synthetic dataset: read_npy(\"iris_like_features.npy\")"
echo "  - View dataset info: cat iris_like_info.md"
echo "  - Inspect CSV data: head iris_like.csv"
echo "  - Clean old traces: rm execution-traces/test_execution_*.md sample_trained_model_*.bin"
echo "  - Generate new datasets: cargo run --bin generate_synthetic_datasets"
