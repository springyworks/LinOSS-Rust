#!/usr/bin/env python3
"""
AI Performance Advisory for VS Code/GitHub Copilot
===================================================

This script monitors AI server performance patterns and provides development
timing recommendations based on the European morning peak load analysis
completed during the LinossRust project development.

Features:
- Real-time performance window detection
- Copilot settings recommendations
- Workflow optimization suggestions
- Integration with VS Code workspace settings

Usage:
    python ai_performance_advisor.py [--watch] [--config] [--status]
"""

import json
import datetime
import argparse
import sys
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

@dataclass
class PerformanceWindow:
    """Represents a performance window with timing and recommendations."""
    name: str
    description: str
    time_ranges: List[Dict]
    recommendations: List[str]
    copilot_settings: Dict

@dataclass
class AIPerformanceStatus:
    """Current AI performance status and recommendations."""
    current_window: str
    description: str
    recommendations: List[str]
    copilot_settings: Dict
    next_optimal_time: Optional[datetime.datetime]
    time_until_optimal: Optional[str]

class AIPerformanceAdvisor:
    """Main class for AI performance monitoring and advisory."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or ".vscode/copilot-performance.json"
        self.timezone = "Europe/Amsterdam"
        self.config = self._load_config()
        
    def _load_config(self) -> Dict:
        """Load performance configuration from JSON file."""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"⚠️  Configuration file not found: {self.config_path}")
            return self._get_default_config()
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing configuration file: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration if file is not available."""
        return {
            "aiPerformanceMonitoring": {
                "enabled": True,
                "timezone": "Europe/Amsterdam",
                "performanceWindows": {
                    "peak": {
                        "description": "High AI server load - reduced performance expected",
                        "timeRanges": [
                            {"start": "08:00", "end": "10:00", "days": ["mon", "tue", "wed", "thu", "fri"]},
                            {"start": "13:00", "end": "15:00", "days": ["mon", "tue", "wed", "thu", "fri"]},
                            {"start": "19:00", "end": "21:00", "days": ["mon", "tue", "wed", "thu", "fri"]}
                        ],
                        "recommendations": [
                            "Consider pausing intensive AI-assisted development",
                            "Focus on code review, documentation, or testing",
                            "Use offline development activities"
                        ]
                    },
                    "optimal": {
                        "description": "Low AI server load - excellent performance expected",
                        "timeRanges": [
                            {"start": "05:00", "end": "08:00", "days": ["mon", "tue", "wed", "thu", "fri"]},
                            {"start": "21:00", "end": "23:00", "days": ["mon", "tue", "wed", "thu", "fri"]},
                            {"start": "06:00", "end": "22:00", "days": ["sat", "sun"]}
                        ],
                        "recommendations": [
                            "Ideal time for complex AI-assisted development",
                            "Schedule architecture decisions and refactoring",
                            "Use advanced Copilot Chat features"
                        ]
                    },
                    "moderate": {
                        "description": "Standard AI server load - normal performance",
                        "timeRanges": [
                            {"start": "10:00", "end": "13:00", "days": ["mon", "tue", "wed", "thu", "fri"]},
                            {"start": "15:00", "end": "19:00", "days": ["mon", "tue", "wed", "thu", "fri"]},
                            {"start": "23:00", "end": "05:00", "days": ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]}
                        ],
                        "recommendations": [
                            "Standard development workflow",
                            "Regular AI assistance for coding tasks"
                        ]
                    }
                }
            }
        }
    
    def get_current_performance_window(self) -> str:
        """Determine the current performance window based on time and day."""
        now = datetime.datetime.now()
        current_time = now.strftime("%H:%M")
        current_day = now.strftime("%a").lower()
        
        windows = self.config["aiPerformanceMonitoring"]["performanceWindows"]
        
        for window_name, window_config in windows.items():
            for time_range in window_config["timeRanges"]:
                if current_day in time_range["days"]:
                    start_time = datetime.datetime.strptime(time_range["start"], "%H:%M").time()
                    end_time = datetime.datetime.strptime(time_range["end"], "%H:%M").time()
                    current_time_obj = datetime.datetime.strptime(current_time, "%H:%M").time()
                    
                    # Handle overnight ranges
                    if start_time > end_time:
                        if current_time_obj >= start_time or current_time_obj <= end_time:
                            return window_name
                    else:
                        if start_time <= current_time_obj <= end_time:
                            return window_name
        
        return "moderate"  # Default fallback
    
    def get_status(self) -> AIPerformanceStatus:
        """Get current AI performance status and recommendations."""
        current_window = self.get_current_performance_window()
        window_config = self.config["aiPerformanceMonitoring"]["performanceWindows"][current_window]
        
        # Find next optimal time
        next_optimal = self._find_next_optimal_time()
        time_until_optimal = self._calculate_time_until(next_optimal) if next_optimal else None
        
        return AIPerformanceStatus(
            current_window=current_window,
            description=window_config["description"],
            recommendations=window_config["recommendations"],
            copilot_settings=window_config.get("copilotSettings", {}),
            next_optimal_time=next_optimal,
            time_until_optimal=time_until_optimal
        )
    
    def _find_next_optimal_time(self) -> Optional[datetime.datetime]:
        """Find the next optimal performance window."""
        now = datetime.datetime.now()
        optimal_ranges = self.config["aiPerformanceMonitoring"]["performanceWindows"]["optimal"]["timeRanges"]
        
        # Check today first
        for time_range in optimal_ranges:
            if now.strftime("%a").lower() in time_range["days"]:
                start_time = datetime.datetime.combine(
                    now.date(),
                    datetime.datetime.strptime(time_range["start"], "%H:%M").time()
                )
                if start_time > now:
                    return start_time
        
        # Check next 7 days
        for days_ahead in range(1, 8):
            future_date = now + datetime.timedelta(days=days_ahead)
            future_day = future_date.strftime("%a").lower()
            
            for time_range in optimal_ranges:
                if future_day in time_range["days"]:
                    return datetime.datetime.combine(
                        future_date.date(),
                        datetime.datetime.strptime(time_range["start"], "%H:%M").time()
                    )
        
        return None
    
    def _calculate_time_until(self, target_time: datetime.datetime) -> str:
        """Calculate human-readable time until target."""
        now = datetime.datetime.now()
        diff = target_time - now
        
        days = diff.days
        hours, remainder = divmod(diff.seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        
        parts = []
        if days > 0:
            parts.append(f"{days} day{'s' if days > 1 else ''}")
        if hours > 0:
            parts.append(f"{hours} hour{'s' if hours > 1 else ''}")
        if minutes > 0:
            parts.append(f"{minutes} minute{'s' if minutes > 1 else ''}")
        
        return ", ".join(parts) if parts else "less than a minute"
    
    def print_status(self):
        """Print current AI performance status to console."""
        status = self.get_status()
        
        # Status indicator
        indicators = {
            "optimal": "🟢",
            "moderate": "🟡", 
            "peak": "🔴"
        }
        
        indicator = indicators.get(status.current_window, "⚪")
        
        print(f"\n{indicator} AI Performance Status: {status.current_window.upper()}")
        print(f"📋 {status.description}")
        
        print(f"\n💡 Recommendations:")
        for i, rec in enumerate(status.recommendations, 1):
            print(f"   {i}. {rec}")
        
        if status.next_optimal_time:
            print(f"\n⏰ Next optimal window: {status.next_optimal_time.strftime('%A, %B %d at %H:%M')}")
            print(f"   (in {status.time_until_optimal})")
        
        if status.copilot_settings:
            print(f"\n⚙️  Recommended Copilot settings:")
            for key, value in status.copilot_settings.items():
                print(f"   {key}: {value}")
    
    def update_vscode_settings(self, dry_run: bool = True):
        """Update VS Code settings based on current performance window."""
        status = self.get_status()
        settings_path = ".vscode/settings.json"
        
        if not os.path.exists(settings_path):
            print(f"❌ VS Code settings file not found: {settings_path}")
            return
        
        try:
            with open(settings_path, 'r') as f:
                settings = json.load(f)
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing VS Code settings: {e}")
            return
        
        # Update Copilot settings
        updated = False
        for key, value in status.copilot_settings.items():
            if settings.get(key) != value:
                settings[key] = value
                updated = True
                print(f"📝 {'Would update' if dry_run else 'Updated'} {key}: {value}")
        
        # Add performance window indicator
        new_title = f"${{dirty}}${{activeEditorShort}}${{separator}}${{rootName}}${{separator}}[AI: {status.current_window.upper()}]"
        if settings.get("window.title") != new_title:
            settings["window.title"] = new_title
            updated = True
            print(f"📝 {'Would update' if dry_run else 'Updated'} window title with performance indicator")
        
        if updated and not dry_run:
            with open(settings_path, 'w') as f:
                json.dump(settings, f, indent=4)
            print(f"✅ VS Code settings updated successfully")
        elif not updated:
            print(f"✅ VS Code settings already optimal for current window")
    
    def watch_mode(self):
        """Run in continuous monitoring mode."""
        print("🔍 Starting AI Performance Advisory watch mode...")
        print("Press Ctrl+C to stop")
        
        last_window = None
        
        try:
            while True:
                import time
                current_window = self.get_current_performance_window()
                
                if current_window != last_window:
                    print(f"\n⏰ {datetime.datetime.now().strftime('%H:%M:%S')} - Performance window changed!")
                    self.print_status()
                    print("\n" + "="*60)
                    last_window = current_window
                
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            print("\n👋 AI Performance Advisory stopped")

def main():
    """Main entry point for the AI Performance Advisory script."""
    parser = argparse.ArgumentParser(
        description="AI Performance Advisory for VS Code/GitHub Copilot",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--watch", 
        action="store_true", 
        help="Run in continuous monitoring mode"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to configuration file (default: .vscode/copilot-performance.json)"
    )
    parser.add_argument(
        "--status", 
        action="store_true", 
        help="Show current status and exit"
    )
    parser.add_argument(
        "--update-settings", 
        action="store_true", 
        help="Update VS Code settings based on current performance window"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Show what would be changed without actually updating files"
    )
    
    args = parser.parse_args()
    
    advisor = AIPerformanceAdvisor(config_path=args.config)
    
    if args.watch:
        advisor.watch_mode()
    elif args.update_settings:
        advisor.update_vscode_settings(dry_run=args.dry_run)
    else:
        advisor.print_status()

if __name__ == "__main__":
    main()
