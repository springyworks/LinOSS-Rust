# 🚀 AI Performance Advisory - Quick Start Guide

## Instant Setup (2 minutes)

### 1. Check Current Status
```bash
python3 ai_performance_advisor.py --status
```

### 2. Start Monitoring (Optional)
```bash
python3 ai_performance_advisor.py --watch
```
*Press Ctrl+C to stop*

### 3. Update Copilot Settings
```bash
# Preview changes first
python3 ai_performance_advisor.py --update-settings --dry-run

# Apply changes
python3 ai_performance_advisor.py --update-settings
```

## VS Code Integration (1 minute)

1. **Open Command Palette**: `Ctrl+Shift+P`
2. **Type**: "Tasks: Run Task"
3. **Select**: "AI Performance Status"

### Available VS Code Tasks:
- **AI Performance Status** - Quick check
- **Start AI Performance Monitoring** - Background watch
- **Update Copilot Settings** - Apply optimal settings
- **AI-Optimized Development Session** - Complete workflow

## Performance Windows (Remember These)

| Time (CET) | Status | Action |
|------------|--------|---------|
| 🟢 05:00-08:00 | OPTIMAL | Complex AI development |
| 🔴 08:00-10:00 | PEAK | Avoid AI, do reviews |
| 🟡 10:00-13:00 | MODERATE | Normal development |
| 🔴 13:00-15:00 | PEAK | Avoid AI, do testing |
| 🟡 15:00-19:00 | MODERATE | Normal development |
| 🔴 19:00-21:00 | PEAK | Avoid AI, do docs |
| 🟢 21:00-23:00 | OPTIMAL | Complex AI development |
| 🟡 23:00-05:00 | MODERATE | Normal development |

**Weekends**: 🟢 OPTIMAL from 06:00-22:00

## Daily Workflow

### Morning (Check First Thing)
```bash
python3 ai_performance_advisor.py --status
```

### Before Intensive AI Work
```bash
python3 ai_performance_advisor.py --update-settings
```

### When Performance Changes
- Watch for window title updates: `[AI: OPTIMAL]`
- VS Code shows current status in title bar

## Emergency Quick Reference

### If AI is Slow/Unresponsive:
1. Check status: `python3 ai_performance_advisor.py --status`
2. If status is 🔴 PEAK → Switch to non-AI tasks
3. If status is 🟢 OPTIMAL → Check internet/VS Code restart

### Best Development Times:
- **Early Morning**: 05:00-08:00 (CET)
- **Late Evening**: 21:00-23:00 (CET) 
- **Weekends**: Almost anytime 06:00-22:00 (CET)

### Worst Development Times:
- **European Morning Rush**: 08:00-10:00 (CET)
- **Lunch Break**: 13:00-15:00 (CET)
- **Evening Peak**: 19:00-21:00 (CET)

---

*💡 Pro Tip: Add `alias ai-status="python3 ai_performance_advisor.py --status"` to your shell profile for instant access!*
