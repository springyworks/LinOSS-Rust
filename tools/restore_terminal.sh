#!/bin/bash
# Script to restore terminal after TUI app crashes
echo "Restoring terminal..."
reset
stty sane
echo "Terminal restored!"
