# Filename: mechanical_energy_solver.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arrow
from mpl_toolkits.mplot3d import Axes3D
import os

def calculate_kinetic_energy(mass, velocity):
    """
    Calculate the kinetic energy of an object.
    
    Parameters:
    -----------
    mass : float
        Mass of the object in kilograms (kg)
    velocity : float or numpy.ndarray
        Velocity of the object in meters per second (m/s)
        Can be a scalar or vector quantity
    
    Returns:
    --------
    float
        Kinetic energy in joules (J)
    """
    if isinstance(velocity, np.ndarray):
        v_squared = np.sum(velocity**2)
    else:
        v_squared = velocity**2
    
    return 0.5 * mass * v_squared

def calculate_gravitational_potential_energy(mass, height, gravity=9.8):
    """
    Calculate the gravitational potential energy of an object.
    
    Parameters:
    -----------
    mass : float
        Mass of the object in kilograms (kg)
    height : float
        Height of the object above the reference level in meters (m)
    gravity : float, optional
        Acceleration due to gravity in m/s², default is 9.8 m/s²
    
    Returns:
    --------
    float
        Gravitational potential energy in joules (J)
    """
    return mass * gravity * height

def calculate_elastic_potential_energy(spring_constant, displacement):
    """
    Calculate the elastic potential energy stored in a spring.
    
    Parameters:
    -----------
    spring_constant : float
        Spring constant in newtons per meter (N/m)
    displacement : float
        Displacement of the spring from its equilibrium position in meters (m)
    
    Returns:
    --------
    float
        Elastic potential energy in joules (J)
    """
    return 0.5 * spring_constant * displacement**2

def calculate_total_mechanical_energy(kinetic_energy, potential_energies):
    """
    Calculate the total mechanical energy of a system.
    
    Parameters:
    -----------
    kinetic_energy : float
        Kinetic energy in joules (J)
    potential_energies : list or float
        List of potential energies or a single potential energy value in joules (J)
    
    Returns:
    --------
    float
        Total mechanical energy in joules (J)
    """
    if isinstance(potential_energies, list):
        total_potential = sum(potential_energies)
    else:
        total_potential = potential_energies
    
    return kinetic_energy + total_potential

def solve_final_velocity_conservation(mass, initial_velocity, initial_height, final_height=0, gravity=9.8):
    """
    Solve for the final velocity of an object using conservation of mechanical energy.
    
    Parameters:
    -----------
    mass : float
        Mass of the object in kilograms (kg)
    initial_velocity : float or numpy.ndarray
        Initial velocity of the object in meters per second (m/s)
    initial_height : float
        Initial height of the object in meters (m)
    final_height : float, optional
        Final height of the object in meters (m), default is 0 (ground level)
    gravity : float, optional
        Acceleration due to gravity in m/s², default is 9.8 m/s²
    
    Returns:
    --------
    float
        Final velocity of the object in meters per second (m/s)
    """
    # Calculate initial energies
    if isinstance(initial_velocity, np.ndarray):
        initial_ke = calculate_kinetic_energy(mass, initial_velocity)
        initial_speed = np.sqrt(np.sum(initial_velocity**2))
    else:
        initial_ke = calculate_kinetic_energy(mass, initial_velocity)
        initial_speed = abs(initial_velocity)
    
    initial_pe = calculate_gravitational_potential_energy(mass, initial_height, gravity)
    initial_total_energy = calculate_total_mechanical_energy(initial_ke, initial_pe)
    
    # Calculate final potential energy
    final_pe = calculate_gravitational_potential_energy(mass, final_height, gravity)
    
    # Conservation of energy: initial_total_energy = final_ke + final_pe
    final_ke = initial_total_energy - final_pe
    
    # Solve for final velocity
    final_velocity = np.sqrt(2 * final_ke / mass)
    
    return final_velocity

def solve_pendulum_motion(mass, length, initial_angle, initial_angular_velocity=0, gravity=9.8, time_points=100, total_time=5):
    """
    Solve the motion of a simple pendulum using energy conservation principles.
    
    Parameters:
    -----------
    mass : float
        Mass of the pendulum bob in kilograms (kg)
    length : float
        Length of the pendulum in meters (m)
    initial_angle : float
        Initial angle of the pendulum from vertical in radians
    initial_angular_velocity : float, optional
        Initial angular velocity in radians per second (rad/s), default is 0
    gravity : float, optional
        Acceleration due to gravity in m/s², default is 9.8 m/s²
    time_points : int, optional
        Number of time points to calculate, default is 100
    total_time : float, optional
        Total time to simulate in seconds, default is 5
    
    Returns:
    --------
    tuple
        (times, angles, angular_velocities, x_positions, y_positions, energies)
        times: array of time points
        angles: array of pendulum angles
        angular_velocities: array of angular velocities
        x_positions: array of x-coordinates of the pendulum bob
        y_positions: array of y-coordinates of the pendulum bob
        energies: dictionary containing kinetic, potential, and total energy arrays
    """
    # For small angles, we can use the simple harmonic motion approximation
    # For larger angles, we would need numerical integration (not implemented here)
    
    # Calculate the period of the pendulum
    period = 2 * np.pi * np.sqrt(length / gravity)
    
    # Create time array
    times = np.linspace(0, total_time, time_points)
    
    # Calculate angular position and velocity
    if abs(initial_angle) < 0.1:  # Small angle approximation
        omega = np.sqrt(gravity / length)
        angles = initial_angle * np.cos(omega * times)
        angular_velocities = -initial_angle * omega * np.sin(omega * times)
    else:
        # For larger angles, we use conservation of energy to approximate
        # This is a simplified approach and not fully accurate for large angles
        max_height = length * (1 - np.cos(initial_angle))
        angles = np.zeros_like(times)
        angular_velocities = np.zeros_like(times)
        
        for i, t in enumerate(times):
            # Approximate angle using conservation of energy
            phase = (t % period) / period
            if phase < 0.5:
                # Moving from max to min
                progress = phase * 2
                angles[i] = initial_angle * np.cos(np.pi * progress)
            else:
                # Moving from min to max
                progress = (phase - 0.5) * 2
                angles[i] = -initial_angle * np.cos(np.pi * progress)
            
            # Approximate angular velocity using energy conservation
            height = length * (1 - np.cos(angles[i]))
            potential_energy = mass * gravity * height
            kinetic_energy = mass * gravity * max_height - potential_energy
            
            if kinetic_energy < 0:  # Numerical error
                kinetic_energy = 0
                
            speed = np.sqrt(2 * kinetic_energy / mass)
            angular_velocities[i] = speed / length * (1 if angles[i] < angles[i-1] else -1) if i > 0 else 0
    
    # Calculate positions
    x_positions = length * np.sin(angles)
    y_positions = -length * np.cos(angles)
    
    # Calculate energies
    kinetic_energies = 0.5 * mass * (length * angular_velocities)**2
    potential_energies = mass * gravity * (length - length * np.cos(angles))
    total_energies = kinetic_energies + potential_energies
    
    energies = {
        'kinetic': kinetic_energies,
        'potential': potential_energies,
        'total': total_energies
    }
    
    return times, angles, angular_velocities, x_positions, y_positions, energies

def visualize_pendulum_motion(times, x_positions, y_positions, energies, save_path=None):
    """
    Visualize the motion of a pendulum and its energy components.
    
    Parameters:
    -----------
    times : numpy.ndarray
        Array of time points
    x_positions : numpy.ndarray
        Array of x-coordinates of the pendulum bob
    y_positions : numpy.ndarray
        Array of y-coordinates of the pendulum bob
    energies : dict
        Dictionary containing 'kinetic', 'potential', and 'total' energy arrays
    save_path : str, optional
        Path to save the visualization, if None, the plot is displayed
    
    Returns:
    --------
    None
    """
    # Set up fonts for plot
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot pendulum trajectory
    ax1.plot(x_positions, y_positions, 'b-', alpha=0.3)
    ax1.plot([0, x_positions[0]], [0, y_positions[0]], 'k-')
    ax1.plot(0, 0, 'ko', markersize=5)
    ax1.plot(x_positions[0], y_positions[0], 'bo', markersize=10)
    
    # Set equal aspect ratio
    ax1.set_aspect('equal')
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.set_title('Pendulum Trajectory')
    ax1.grid(True)
    
    # Plot energy components
    ax2.plot(times, energies['kinetic'], 'r-', label='Kinetic Energy')
    ax2.plot(times, energies['potential'], 'g-', label='Potential Energy')
    ax2.plot(times, energies['total'], 'b-', label='Total Energy')
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Energy (J)')
    ax2.set_title('Energy Components')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def visualize_energy_conservation(initial_state, final_state, save_path=None):
    """
    Visualize the conservation of energy between initial and final states.
    
    Parameters:
    -----------
    initial_state : dict
        Dictionary containing initial state information:
        - 'height': height in meters
        - 'velocity': velocity in m/s
        - 'mass': mass in kg
        - 'kinetic_energy': kinetic energy in joules
        - 'potential_energy': potential energy in joules
        - 'total_energy': total energy in joules
    final_state : dict
        Dictionary containing final state information (same keys as initial_state)
    save_path : str, optional
        Path to save the visualization, if None, the plot is displayed
    
    Returns:
    --------
    None
    """
    # Set up fonts for plot
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot initial and final states
    states = ['Initial', 'Final']
    heights = [initial_state['height'], final_state['height']]
    velocities = [initial_state['velocity'], final_state['velocity']]
    
    # Create a simple visualization of the object at different heights
    max_height = max(heights) + 0.1
    ax1.set_xlim(-0.5, 1.5)
    ax1.set_ylim(0, max_height * 1.2)
    
    # Draw ground
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=2)
    
    # Draw objects
    for i, (state, height, velocity) in enumerate(zip(states, heights, velocities)):
        circle = Circle((i, height), 0.1, color='blue', alpha=0.7)
        ax1.add_patch(circle)
        
        # Add velocity vector
        if velocity != 0:
            arrow_length = 0.2 * velocity / max(velocities)
            arrow = Arrow(i, height, 0, -arrow_length, width=0.05, color='red')
            ax1.add_patch(arrow)
        
        # Add labels
        ax1.text(i, height + 0.15, f"{state}\nv={velocity:.2f} m/s", 
                 ha='center', va='bottom')
    
    ax1.set_xlabel('State')
    ax1.set_ylabel('Height (m)')
    ax1.set_title('Object Position and Velocity')
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(states)
    ax1.grid(True)
    
    # Plot energy components
    energy_types = ['Kinetic Energy', 'Potential Energy', 'Total Energy']
    initial_energies = [initial_state['kinetic_energy'], 
                        initial_state['potential_energy'], 
                        initial_state['total_energy']]
    final_energies = [final_state['kinetic_energy'], 
                      final_state['potential_energy'], 
                      final_state['total_energy']]
    
    x = np.arange(len(energy_types))
    width = 0.35
    
    ax2.bar(x - width/2, initial_energies, width, label='Initial')
    ax2.bar(x + width/2, final_energies, width, label='Final')
    
    ax2.set_ylabel('Energy (J)')
    ax2.set_title('Energy Conservation')
    ax2.set_xticks(x)
    ax2.set_xticklabels(energy_types)
    ax2.legend()
    
    # Add value labels on bars
    for i, v in enumerate(initial_energies):
        ax2.text(i - width/2, v + 0.1, f"{v:.2f}", ha='center')
    
    for i, v in enumerate(final_energies):
        ax2.text(i + width/2, v + 0.1, f"{v:.2f}", ha='center')
    
    plt.tight_layout()
    
    if save_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def main():
    """
    Main function to demonstrate the use of the mechanical energy solver.
    Solves the problem: "Starting with an initial speed of 5.00 m/s at a height of 0.300 m, 
    a 1.50 kg ball swings downward. Using the principle of conservation of mechanical energy, 
    find the speed of the 1.50 kg ball just before impact."
    """
    # Problem parameters
    mass = 1.50  # kg
    initial_velocity = 5.00  # m/s
    initial_height = 0.300  # m
    final_height = 0.0  # m (ground level)
    gravity = 9.8  # m/s²
    
    # Calculate initial energies
    initial_ke = calculate_kinetic_energy(mass, initial_velocity)
    initial_pe = calculate_gravitational_potential_energy(mass, initial_height, gravity)
    initial_total_energy = calculate_total_mechanical_energy(initial_ke, initial_pe)
    
    # Solve for final velocity using conservation of energy
    final_velocity = solve_final_velocity_conservation(
        mass, initial_velocity, initial_height, final_height, gravity
    )
    
    # Calculate final energies
    final_ke = calculate_kinetic_energy(mass, final_velocity)
    final_pe = calculate_gravitational_potential_energy(mass, final_height, gravity)
    final_total_energy = calculate_total_mechanical_energy(final_ke, final_pe)
    
    # Print results
    print("\nMechanical Energy Conservation Analysis")
    print("=======================================")
    print(f"Mass: {mass} kg")
    print(f"Gravity: {gravity} m/s²")
    print("\nInitial State:")
    print(f"  Height: {initial_height} m")
    print(f"  Velocity: {initial_velocity} m/s")
    print(f"  Kinetic Energy: {initial_ke:.2f} J")
    print(f"  Potential Energy: {initial_pe:.2f} J")
    print(f"  Total Energy: {initial_total_energy:.2f} J")
    print("\nFinal State:")
    print(f"  Height: {final_height} m")
    print(f"  Velocity: {final_velocity:.2f} m/s")
    print(f"  Kinetic Energy: {final_ke:.2f} J")
    print(f"  Potential Energy: {final_pe:.2f} J")
    print(f"  Total Energy: {final_total_energy:.2f} J")
    print("\nEnergy Conservation Check:")
    print(f"  Energy Difference: {final_total_energy - initial_total_energy:.6f} J")
    print(f"  Relative Error: {abs((final_total_energy - initial_total_energy)/initial_total_energy)*100:.6f}%")
    
    # Visualize the energy conservation
    initial_state = {
        'height': initial_height,
        'velocity': initial_velocity,
        'mass': mass,
        'kinetic_energy': initial_ke,
        'potential_energy': initial_pe,
        'total_energy': initial_total_energy
    }
    
    final_state = {
        'height': final_height,
        'velocity': final_velocity,
        'mass': mass,
        'kinetic_energy': final_ke,
        'potential_energy': final_pe,
        'total_energy': final_total_energy
    }
    
    # Create directory for images if it doesn't exist
    os.makedirs("./images", exist_ok=True)
    
    # Visualize energy conservation
    visualize_energy_conservation(
        initial_state, 
        final_state, 
        save_path="./images/energy_conservation.png"
    )
    
    print("\nVisualization saved to './images/energy_conservation.png'")

if __name__ == "__main__":
    main()