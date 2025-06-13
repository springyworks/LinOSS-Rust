#!/usr/bin/env python3
"""
Brain Dynamics Analysis Tool
Analyzes the JSON data from the Lorenz brain dynamics simulation
to understand why the system evolves from chaotic to static behavior.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any

def load_dynamics_data(filename: str) -> List[Dict[str, Any]]:
    """Load the brain dynamics JSON data."""
    with open(filename, 'r') as f:
        return json.load(f)

def extract_positions(data: List[Dict[str, Any]], region_name: str) -> np.ndarray:
    """Extract position data for a specific brain region."""
    positions = []
    for timestep in data:
        for region in timestep['regions']:
            if region['name'] == region_name:
                positions.append(region['position'])
                break
    return np.array(positions)

def calculate_energy_over_time(data: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """Calculate total energy for each region over time."""
    energy_data = {}
    
    for timestep in data:
        for region in timestep['regions']:
            name = region['name']
            if name not in energy_data:
                energy_data[name] = []
            energy_data[name].append(region['energy'])
    
    return {name: np.array(values) for name, values in energy_data.items()}

def calculate_velocity_magnitude(data: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """Calculate velocity magnitude for each region over time."""
    velocity_data = {}
    
    for timestep in data:
        for region in timestep['regions']:
            name = region['name']
            if name not in velocity_data:
                velocity_data[name] = []
            
            velocity = np.array(region['velocity'])
            magnitude = np.linalg.norm(velocity)
            velocity_data[name].append(magnitude)
    
    return {name: np.array(values) for name, values in velocity_data.items()}

def analyze_attractor_convergence(data: List[Dict[str, Any]]):
    """Analyze why the system converges from chaos to static points."""
    
    print("🧠 BRAIN DYNAMICS ANALYSIS: From Chaos to Equilibrium")
    print("=" * 60)
    
    # Extract time points
    times = [step['time'] for step in data]
    
    # Analyze each brain region
    regions = ['Prefrontal_Cortex', 'Default_Mode_Network', 'Thalamus']
    
    for region_name in regions:
        print(f"\n📊 ANALYSIS: {region_name.replace('_', ' ')}")
        print("-" * 40)
        
        # Extract positions and velocities
        positions = extract_positions(data, region_name)
        
        if len(positions) == 0:
            continue
            
        # Calculate distance from origin over time
        distances = np.linalg.norm(positions, axis=1)
        
        # Calculate velocity magnitudes
        velocity_data = calculate_velocity_magnitude(data)
        velocities = velocity_data.get(region_name, np.array([]))
        
        # Energy analysis
        energy_data = calculate_energy_over_time(data)
        energies = energy_data.get(region_name, np.array([]))
        
        # Print key statistics
        print(f"Initial position: ({positions[0][0]:.3f}, {positions[0][1]:.3f}, {positions[0][2]:.3f})")
        print(f"Final position:   ({positions[-1][0]:.3f}, {positions[-1][1]:.3f}, {positions[-1][2]:.3f})")
        print(f"Initial velocity magnitude: {velocities[0]:.6f}")
        print(f"Final velocity magnitude:   {velocities[-1]:.6f}")
        print(f"Initial energy: {energies[0]:.3f}")
        print(f"Final energy:   {energies[-1]:.3f}")
        
        # Calculate convergence metrics
        pos_change = np.linalg.norm(positions[-1] - positions[0])
        velocity_decay = (velocities[0] - velocities[-1]) / velocities[0] * 100
        energy_change = (energies[0] - energies[-1]) / energies[0] * 100
        
        print(f"Total position change: {pos_change:.3f}")
        print(f"Velocity decay: {velocity_decay:.2f}%")
        print(f"Energy dissipation: {energy_change:.2f}%")
        
        # Find when system becomes "static" (velocity < 0.01)
        static_threshold = 0.01
        static_indices = np.where(velocities < static_threshold)[0]
        if len(static_indices) > 0:
            static_time = times[static_indices[0]]
            print(f"⚡ System becomes static at t = {static_time:.2f}s")
        
    return times, positions, velocities, energies

def compare_with_without_dlinoss():
    """Compare dynamics with and without dLinOSS influence."""
    print("\n🔬 COMPARING: Pure Lorenz vs dLinOSS-Influenced Dynamics")
    print("=" * 60)
    
    # Load both datasets
    no_dlinoss = load_dynamics_data('/home/rustuser/rustdev/LinossRust/logs/brain_dynamics_no_dlinoss.json')
    with_dlinoss = load_dynamics_data('/home/rustuser/rustdev/LinossRust/logs/brain_dynamics_with_dlinoss.json')
    
    for region_name in ['Prefrontal_Cortex', 'Default_Mode_Network', 'Thalamus']:
        print(f"\n📈 {region_name.replace('_', ' ')}:")
        
        # Extract final velocities
        no_dlinoss_vel = calculate_velocity_magnitude(no_dlinoss)[region_name][-1]
        with_dlinoss_vel = calculate_velocity_magnitude(with_dlinoss)[region_name][-1]
        
        # Extract final positions
        no_dlinoss_pos = extract_positions(no_dlinoss, region_name)[-1]
        with_dlinoss_pos = extract_positions(with_dlinoss, region_name)[-1]
        
        print(f"  Pure Lorenz final velocity: {no_dlinoss_vel:.6f}")
        print(f"  dLinOSS final velocity:     {with_dlinoss_vel:.6f}")
        print(f"  Damping effect: {((no_dlinoss_vel - with_dlinoss_vel) / no_dlinoss_vel * 100):.2f}%")

def explain_convergence_reasons():
    """Explain the mathematical reasons for convergence."""
    print("\n🎯 WHY DOES THE SYSTEM CONVERGE TO STATIC POINTS?")
    print("=" * 60)
    print("""
🔹 NUMERICAL INTEGRATION EFFECTS:
   • Small time step (dt = 0.01) with accumulated rounding errors
   • Finite precision arithmetic causes energy dissipation
   • Each calculation introduces tiny numerical damping

🔹 dLinOSS NEURAL DAMPING:
   • Neural network adds adaptive perturbations
   • These perturbations can act as implicit damping
   • Network learns to stabilize the chaotic dynamics

🔹 COUPLING BETWEEN REGIONS:
   • Inter-regional coupling can create energy sinks
   • Synchronization effects reduce total system energy
   • Coupled oscillators tend toward phase-locked states

🔹 LORENZ SYSTEM SENSITIVITY:
   • Lorenz attractors are sensitive to parameter changes
   • Small changes can shift from chaotic to fixed-point behavior
   • Initial conditions and perturbations matter enormously

🔹 COMPUTATIONAL CONSTRAINTS:
   • Discrete time steps vs continuous differential equations
   • Limited floating-point precision
   • Accumulation of truncation errors over time
""")

def main():
    """Main analysis function."""
    print("🚀 STARTING BRAIN DYNAMICS ANALYSIS...")
    
    # Analyze convergence behavior
    try:
        no_dlinoss_data = load_dynamics_data('/home/rustuser/rustdev/LinossRust/logs/brain_dynamics_no_dlinoss.json')
        times, positions, velocities, energies = analyze_attractor_convergence(no_dlinoss_data)
        
        # Compare with/without dLinOSS
        compare_with_without_dlinoss()
        
        # Explain the mathematical reasons
        explain_convergence_reasons()
        
        print(f"\n✅ Analysis complete! Processed {len(times)} timesteps of brain dynamics.")
        
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find data files. {e}")
        print("💡 Make sure you've run the brain_dynamics_analyzer first!")

if __name__ == "__main__":
    main()
