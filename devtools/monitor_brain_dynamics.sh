#!/bin/bash

# Helper script to monitor dLinOSS brain dynamics instrumentation in real-time

LOG_FILE="/tmp/dlinoss_brain_dynamics.log"
PIPE_FILE="/tmp/dlinoss_brain_pipe"

echo "=== dLinOSS Brain Dynamics Instrumentation Monitor ==="
echo "Log file: $LOG_FILE"
echo "Data pipe: $PIPE_FILE"
echo ""

# Function to monitor log file
monitor_log() {
    echo "--- Monitoring log file (press Ctrl+C to stop) ---"
    if [ -f "$LOG_FILE" ]; then
        tail -f "$LOG_FILE" | while IFS= read -r line; do
            timestamp=$(echo "$line" | cut -d',' -f1)
            json_data=$(echo "$line" | cut -d',' -f2-)
            
            # Extract key metrics from JSON
            sim_time=$(echo "$json_data" | jq -r '.simulation_time // "N/A"' 2>/dev/null || echo "N/A")
            total_energy=$(echo "$json_data" | jq -r '.system_stats.total_energy // "N/A"' 2>/dev/null || echo "N/A") 
            coupling=$(echo "$json_data" | jq -r '.system_stats.coupling_strength // "N/A"' 2>/dev/null || echo "N/A")
            
            printf "[%s] SimTime: %s, Energy: %s, Coupling: %s\n" \
                   "$(date -d "@$timestamp" '+%H:%M:%S' 2>/dev/null || echo "$timestamp")" \
                   "$sim_time" "$total_energy" "$coupling"
        done
    else
        echo "Log file not found. Make sure the simulation is running."
    fi
}

# Function to monitor data pipe
monitor_pipe() {
    echo "--- Monitoring data pipe (press Ctrl+C to stop) ---"
    if [ -p "$PIPE_FILE" ]; then
        cat "$PIPE_FILE" | while IFS= read -r line; do
            # Extract and display key information
            sim_time=$(echo "$line" | jq -r '.simulation_time // "N/A"' 2>/dev/null || echo "N/A")
            region_count=$(echo "$line" | jq -r '.regions | length' 2>/dev/null || echo "N/A")
            
            # Get position of first region (Prefrontal Cortex)
            pos=$(echo "$line" | jq -r '.regions[0].position // "N/A"' 2>/dev/null || echo "N/A")
            activity=$(echo "$line" | jq -r '.regions[0].activity_magnitude // "N/A"' 2>/dev/null || echo "N/A")
            
            printf "[PIPE] SimTime: %s, Regions: %s, PFC pos: %s, activity: %s\n" \
                   "$sim_time" "$region_count" "$pos" "$activity"
        done
    else
        echo "Data pipe not found. Make sure the simulation is running and mkfifo is available."
    fi
}

# Function to show current status
show_status() {
    echo "--- Current Status ---"
    if [ -f "$LOG_FILE" ]; then
        echo "✓ Log file exists ($(wc -l < "$LOG_FILE") lines)"
        echo "  Latest entry:"
        tail -n 1 "$LOG_FILE" | cut -d',' -f2- | jq -r '
          "  SimTime: \(.simulation_time // "N/A")s, " +
          "Regions: \(.regions | length), " +
          "Total Energy: \(.system_stats.total_energy // "N/A")"
        ' 2>/dev/null || echo "  (Unable to parse JSON)"
    else
        echo "✗ Log file not found"
    fi
    
    if [ -p "$PIPE_FILE" ]; then
        echo "✓ Data pipe exists"
    else
        echo "✗ Data pipe not found"
    fi
    
    if pgrep -f "pure_dlinoss_brain_dynamics" > /dev/null; then
        echo "✓ Simulation is running (PID: $(pgrep -f pure_dlinoss_brain_dynamics))"
    else
        echo "✗ Simulation not detected"
    fi
}

# Function to analyze log data
analyze_log() {
    echo "--- Log Analysis ---"
    if [ -f "$LOG_FILE" ]; then
        echo "Total log entries: $(wc -l < "$LOG_FILE")"
        echo "Time range:"
        first_time=$(head -n 1 "$LOG_FILE" | cut -d',' -f2- | jq -r '.simulation_time // 0' 2>/dev/null || echo "0")
        last_time=$(tail -n 1 "$LOG_FILE" | cut -d',' -f2- | jq -r '.simulation_time // 0' 2>/dev/null || echo "0")
        echo "  Start: ${first_time}s"
        echo "  End: ${last_time}s"
        echo "  Duration: $(echo "$last_time - $first_time" | bc 2>/dev/null || echo "N/A")s"
    else
        echo "No log file found."
    fi
}

# Command line interface
case "${1:-status}" in
    "log")
        monitor_log
        ;;
    "pipe")
        monitor_pipe
        ;;
    "status")
        show_status
        ;;
    "analyze")
        analyze_log
        ;;
    "help")
        echo "Usage: $0 [command]"
        echo "Commands:"
        echo "  status   - Show current instrumentation status (default)"
        echo "  log      - Monitor log file in real-time"
        echo "  pipe     - Monitor data pipe in real-time"
        echo "  analyze  - Analyze accumulated log data"
        echo "  help     - Show this help"
        ;;
    *)
        echo "Unknown command: $1"
        echo "Use '$0 help' for usage information."
        exit 1
        ;;
esac
