#!/bin/bash
# LinOSS Rust Project Status Checker
# Automatically tests examples and updates status

set -e

PROJECT_ROOT="/home/rustuser/projects/active/LinossRust"
STATUS_FILE="$PROJECT_ROOT/PROJECT_STATUS.md"
LOG_FILE="$PROJECT_ROOT/example_test_log.txt"

cd "$PROJECT_ROOT"

echo "🔍 LinOSS Rust Example Status Checker"
echo "======================================"
echo "Date: $(date)"
echo "" > "$LOG_FILE"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to test example compilation
test_example() {
    local example_name="$1"
    local example_file="examples/${example_name}.rs"
    
    if [ ! -f "$example_file" ]; then
        echo -e "${RED}❌ MISSING: $example_name${NC}"
        echo "MISSING: $example_name" >> "$LOG_FILE"
        return 1
    fi
    
    echo -n "Testing $example_name... "
    if cargo check --example "$example_name" 2>/dev/null; then
        echo -e "${GREEN}✅ COMPILES${NC}"
        echo "COMPILES: $example_name" >> "$LOG_FILE"
        return 0
    else
        echo -e "${RED}❌ FAILS${NC}"
        echo "FAILS: $example_name" >> "$LOG_FILE"
        # Store detailed error
        cargo check --example "$example_name" 2>> "$LOG_FILE"
        return 1
    fi
}

# Function to check if example runs (basic check)
test_example_run() {
    local example_name="$1"
    echo -n "Quick run test for $example_name... "
    
    # Try to run with timeout (5 seconds) and capture if it starts
    if timeout 5s cargo run --example "$example_name" 2>/dev/null | head -n 5 >/dev/null; then
        echo -e "${GREEN}✅ RUNS${NC}"
        echo "RUNS: $example_name" >> "$LOG_FILE"
        return 0
    else
        echo -e "${YELLOW}⚠️ RUN-ISSUES${NC}"
        echo "RUN-ISSUES: $example_name" >> "$LOG_FILE"
        return 1
    fi
}

# Main status check
echo "🧪 Testing Example Compilation"
echo "==============================="

# List of examples to test (from PROJECT_STATUS.md "NEEDS-TEST" section)
EXAMPLES_TO_TEST=(
    "dlinoss_comprehensive_showcase"
    "dlinoss_oscillator_demo" 
    "dlinoss_response_visualizer"
    "dlinoss_time_series"
    "dlinoss_vs_vanilla_comparison"
    "burn_demo"
    "dlinoss_simple_mnist"
)

# Track results
WORKING_EXAMPLES=()
BROKEN_EXAMPLES=()
MISSING_EXAMPLES=()

# Test each example
for example in "${EXAMPLES_TO_TEST[@]}"; do
    if test_example "$example"; then
        WORKING_EXAMPLES+=("$example")
    else
        if [ -f "examples/${example}.rs" ]; then
            BROKEN_EXAMPLES+=("$example")
        else
            MISSING_EXAMPLES+=("$example")
        fi
    fi
done

echo ""
echo "📊 Results Summary"
echo "=================="
echo -e "${GREEN}✅ Working Examples (${#WORKING_EXAMPLES[@]}):${NC}"
for ex in "${WORKING_EXAMPLES[@]}"; do
    echo "   - $ex"
done

echo -e "${RED}❌ Broken Examples (${#BROKEN_EXAMPLES[@]}):${NC}"
for ex in "${BROKEN_EXAMPLES[@]}"; do
    echo "   - $ex"
done

echo -e "${YELLOW}❓ Missing Examples (${#MISSING_EXAMPLES[@]}):${NC}"
for ex in "${MISSING_EXAMPLES[@]}"; do
    echo "   - $ex"
done

# Update git with findings
echo ""
echo "📝 Updating Git Status"
echo "======================"

# Update .gitattributes based on test results
{
    echo "# Auto-updated by status checker $(date)"
    echo "# Working examples (compile successfully)"
    for ex in "${WORKING_EXAMPLES[@]}"; do
        echo "examples/${ex}.rs status=working-compile"
    done
    
    echo "# Broken examples (compilation fails)"  
    for ex in "${BROKEN_EXAMPLES[@]}"; do
        echo "examples/${ex}.rs status=broken-compile"
    done
    
    echo "# Missing examples"
    for ex in "${MISSING_EXAMPLES[@]}"; do
        echo "examples/${ex}.rs status=missing"
    done
} >> .gitattributes.auto

echo "🎯 Next Steps"
echo "============="
echo "1. Review detailed log: cat $LOG_FILE"
echo "2. Fix broken examples or mark as stale"
echo "3. Update PROJECT_STATUS.md with results"
echo "4. Consider removing missing/broken examples"

# Generate git commands for cleanup
echo ""
echo "🛠️ Suggested Git Commands"
echo "========================="
if [ ${#BROKEN_EXAMPLES[@]} -gt 0 ]; then
    echo "# Move broken examples to OLD/ directory:"
    for ex in "${BROKEN_EXAMPLES[@]}"; do
        echo "git mv examples/${ex}.rs examples/OLD/"
    done
fi

echo ""
echo "✅ Status check complete! See $LOG_FILE for details."
