# AI Performance Advisory - Shell Aliases
# Add these to your ~/.bashrc or ~/.zshrc for instant access

# Quick status check with visual indicators
alias ai-status='cd /home/rustuser/rustdev/LinossRust && ./ai_status_bar.sh'

# Continuous monitoring mode
alias ai-watch='cd /home/rustuser/rustdev/LinossRust && python3 ai_performance_advisor.py --watch'

# Update Copilot settings (dry run first, then apply)
alias ai-update-dry='cd /home/rustuser/rustdev/LinossRust && python3 ai_performance_advisor.py --update-settings --dry-run'
alias ai-update='cd /home/rustuser/rustdev/LinossRust && python3 ai_performance_advisor.py --update-settings'

# Quick VS Code task access
alias ai-vscode='cd /home/rustuser/rustdev/LinossRust && code . && python3 ai_performance_advisor.py --status'

echo "🎯 AI Performance Advisory Aliases Loaded!"
echo "Available commands:"
echo "  ai-status      # Quick status with colors"
echo "  ai-watch       # Continuous monitoring" 
echo "  ai-update-dry  # Preview settings changes"
echo "  ai-update      # Apply optimal settings"
echo "  ai-vscode      # Open VS Code with status"
