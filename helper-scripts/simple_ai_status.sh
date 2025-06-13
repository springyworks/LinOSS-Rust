#!/bin/bash
# Simple AI Status Check
cd /home/rustuser/rustdev/LinossRust
echo "🤖 Current AI Performance Status:"
echo "================================="
python3 ai_performance_advisor.py --status
echo ""
echo "💡 To see this anytime, run: ./simple_ai_status.sh"
