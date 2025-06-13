#!/bin/bash
# AI Performance Status Bar
# Quick status check with visual indicators

# Colors for terminal output
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color
BOLD='\033[1m'

echo -e "${BOLD}🤖 AI Performance Status${NC}"
echo "=========================="

# Run the Python script and capture output
status_output=$(python3 ai_performance_advisor.py --status 2>/dev/null)

if [ $? -eq 0 ]; then
    # Extract the status line
    status_line=$(echo "$status_output" | head -n 1)
    
    # Color code based on status
    if [[ $status_line == *"OPTIMAL"* ]]; then
        echo -e "${GREEN}${status_line}${NC}"
        echo -e "${GREEN}✅ BEST TIME FOR AI DEVELOPMENT${NC}"
    elif [[ $status_line == *"PEAK"* ]]; then
        echo -e "${RED}${status_line}${NC}"
        echo -e "${RED}🚫 AVOID AI-INTENSIVE TASKS${NC}"
    else
        echo -e "${YELLOW}${status_line}${NC}"
        echo -e "${YELLOW}⚡ NORMAL AI USAGE${NC}"
    fi
    
    echo ""
    echo "$status_output" | tail -n +2
else
    echo "❌ Could not get AI performance status"
fi

echo ""
echo "💡 Quick Access Commands:"
echo "   ai-status     # This status check"
echo "   ai-watch      # Continuous monitoring"
echo "   ai-update     # Update Copilot settings"
