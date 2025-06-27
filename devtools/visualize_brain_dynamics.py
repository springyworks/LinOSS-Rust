#!/usr/bin/env python3
"""
Real-time dLinOSS Brain Dynamics Data Visualizer
Reads from the instrumentation log file and plots neural activity in real-time
"""

import json
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import numpy as np
import os

LOG_FILE = "/tmp/dlinoss_brain_dynamics.log"
HISTORY_SIZE = 200  # Number of points to keep in memory

class BrainDynamicsVisualizer:
    def __init__(self):
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Data storage
        self.time_data = deque(maxlen=HISTORY_SIZE)
        self.energy_data = deque(maxlen=HISTORY_SIZE)
        self.region_positions = {
            'Prefrontal Cortex': {'x': deque(maxlen=HISTORY_SIZE), 'y': deque(maxlen=HISTORY_SIZE), 'z': deque(maxlen=HISTORY_SIZE)},
            'Default Mode Network': {'x': deque(maxlen=HISTORY_SIZE), 'y': deque(maxlen=HISTORY_SIZE), 'z': deque(maxlen=HISTORY_SIZE)},
            'Thalamus': {'x': deque(maxlen=HISTORY_SIZE), 'y': deque(maxlen=HISTORY_SIZE), 'z': deque(maxlen=HISTORY_SIZE)}
        }
        self.region_activities = {
            'Prefrontal Cortex': deque(maxlen=HISTORY_SIZE),
            'Default Mode Network': deque(maxlen=HISTORY_SIZE),
            'Thalamus': deque(maxlen=HISTORY_SIZE)
        }
        
        # Plot setup
        self.setup_plots()
        
        # File position tracking
        self.file_pos = 0
        
    def setup_plots(self):
        # Plot 1: Total Energy over time
        self.ax1.set_title('Total Neural Energy')
        self.ax1.set_xlabel('Simulation Time (s)')
        self.ax1.set_ylabel('Energy')
        self.ax1.grid(True)
        
        # Plot 2: Region Activities
        self.ax2.set_title('Neural Region Activities')
        self.ax2.set_xlabel('Simulation Time (s)')
        self.ax2.set_ylabel('Activity Magnitude')
        self.ax2.grid(True)
        
        # Plot 3: 3D Trajectory (X-Y projection)
        self.ax3.set_title('Neural Trajectories (X-Y)')
        self.ax3.set_xlabel('X Position')
        self.ax3.set_ylabel('Y Position')
        self.ax3.grid(True)
        
        # Plot 4: Phase Space (Position vs Velocity)
        self.ax4.set_title('Phase Space (Pos vs Vel)')
        self.ax4.set_xlabel('Position Magnitude')
        self.ax4.set_ylabel('Velocity Magnitude')
        self.ax4.grid(True)
    
    def read_new_data(self):
        """Read new data from the log file"""
        if not os.path.exists(LOG_FILE):
            return
            
        try:
            with open(LOG_FILE, 'r') as f:
                f.seek(self.file_pos)
                
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                        
                    try:
                        parts = line.split(',', 1)
                        if len(parts) == 2:
                            timestamp_str, json_str = parts
                            data = json.loads(json_str)
                            self.process_data_point(data)
                    except (json.JSONDecodeError, ValueError) as e:
                        print(f"Error parsing line: {e}")
                        continue
                        
                self.file_pos = f.tell()
                
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"Error reading file: {e}")
    
    def process_data_point(self, data):
        """Process a single data point"""
        sim_time = data['simulation_time']
        total_energy = data['system_stats']['total_energy']
        
        self.time_data.append(sim_time)
        self.energy_data.append(total_energy)
        
        # Process region data
        for region_data in data['regions']:
            name = region_data['name']
            if name in self.region_positions:
                pos = region_data['position']
                vel = region_data['velocity']
                activity = region_data['activity_magnitude']
                
                self.region_positions[name]['x'].append(pos[0])
                self.region_positions[name]['y'].append(pos[1])
                self.region_positions[name]['z'].append(pos[2])
                self.region_activities[name].append(activity)
    
    def update_plots(self, frame):
        """Update all plots with new data"""
        self.read_new_data()
        
        if len(self.time_data) == 0:
            return
        
        # Clear all plots
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        
        # Reapply setup
        self.setup_plots()
        
        time_array = np.array(self.time_data)
        
        # Plot 1: Total Energy
        if len(self.energy_data) > 0:
            self.ax1.plot(time_array, self.energy_data, 'b-', linewidth=2, label='Total Energy')
            self.ax1.legend()
        
        # Plot 2: Region Activities
        colors = ['cyan', 'yellow', 'magenta']
        for i, (name, activities) in enumerate(self.region_activities.items()):
            if len(activities) > 0:
                self.ax2.plot(time_array, activities, color=colors[i], linewidth=2, label=name)
        self.ax2.legend()
        
        # Plot 3: 3D Trajectories (X-Y projection)
        for i, (name, positions) in enumerate(self.region_positions.items()):
            if len(positions['x']) > 0:
                x_data = np.array(positions['x'])
                y_data = np.array(positions['y'])
                self.ax3.plot(x_data, y_data, color=colors[i], alpha=0.7, linewidth=1, label=name)
                if len(x_data) > 0:
                    self.ax3.scatter(x_data[-1], y_data[-1], color=colors[i], s=50, zorder=5)
        self.ax3.legend()
        
        # Plot 4: Phase Space
        for i, (name, positions) in enumerate(self.region_positions.items()):
            if len(positions['x']) > 1:
                # Calculate position and velocity magnitudes
                x_data = np.array(positions['x'])
                y_data = np.array(positions['y'])
                z_data = np.array(positions['z'])
                
                pos_mag = np.sqrt(x_data**2 + y_data**2 + z_data**2)
                
                # Calculate velocity magnitude from position differences
                vel_mag = np.sqrt(np.diff(x_data)**2 + np.diff(y_data)**2 + np.diff(z_data)**2)
                
                if len(vel_mag) > 0:
                    self.ax4.scatter(pos_mag[1:], vel_mag, color=colors[i], alpha=0.6, s=20, label=name)
        self.ax4.legend()
        
        plt.tight_layout()
    
    def start(self):
        """Start the real-time visualization"""
        ani = animation.FuncAnimation(self.fig, self.update_plots, interval=100, blit=False)
        plt.show()
        return ani

if __name__ == "__main__":
    print("dLinOSS Brain Dynamics Real-time Visualizer")
    print(f"Reading from: {LOG_FILE}")
    print("Close the plot window to exit...")
    
    visualizer = BrainDynamicsVisualizer()
    ani = visualizer.start()
