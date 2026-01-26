# Filename: circular_motion_solver.py

import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

def calculate_conical_pendulum_velocity(length, angle, gravity=9.8):
    """
    Calculate the velocity of a conical pendulum (object moving in a horizontal circle).
    
    This function computes the velocity of an object in conical pendulum motion,
    where the object moves in a horizontal circle while suspended by a string or rod
    that makes an angle with the vertical. The calculation is based on the balance
    between the horizontal component of tension (providing centripetal force) and
    the vertical component balancing gravity.
    
    Parameters:
    -----------
    length : float
        Length of the string or rod (m)
    angle : float
        Angle between the string and the vertical (radians)
    gravity : float, optional
        Acceleration due to gravity (m/s²), default is 9.8 m/s²
    
    Returns:
    --------
    float
        Velocity of the object (m/s)
    
    Notes:
    ------
    The formula is derived from force balance in a conical pendulum:
    - Vertical: T*cos(θ) = mg
    - Horizontal: T*sin(θ) = mv²/r
    - Where r = L*sin(θ) is the radius of the circular path
    """
    # Ensure angle is within valid range
    if angle <= 0 or angle >= np.pi/2:
        raise ValueError("Angle must be between 0 and π/2 radians (exclusive)")
    
    # Calculate velocity using the derived formula
    velocity = np.sqrt(gravity * length * np.sin(angle)**2 / np.cos(angle))
    
    return velocity

def calculate_circular_motion_parameters(velocity, radius, mass=1.0):
    """
    Calculate various parameters for an object in uniform circular motion.
    
    Parameters:
    -----------
    velocity : float
        Tangential velocity of the object (m/s)
    radius : float
        Radius of the circular path (m)
    mass : float, optional
        Mass of the object (kg), default is 1.0 kg
    
    Returns:
    --------
    dict
        Dictionary containing calculated parameters:
        - 'period': Time to complete one revolution (s)
        - 'frequency': Number of revolutions per second (Hz)
        - 'angular_velocity': Angular velocity (rad/s)
        - 'centripetal_acceleration': Centripetal acceleration (m/s²)
        - 'centripetal_force': Centripetal force (N)
    """
    # Calculate angular velocity (ω)
    angular_velocity = velocity / radius
    
    # Calculate period (T)
    period = 2 * np.pi / angular_velocity
    
    # Calculate frequency (f)
    frequency = 1 / period
    
    # Calculate centripetal acceleration (a_c)
    centripetal_acceleration = velocity**2 / radius
    
    # Calculate centripetal force (F_c)
    centripetal_force = mass * centripetal_acceleration
    
    return {
        'period': period,
        'frequency': frequency,
        'angular_velocity': angular_velocity,
        'centripetal_acceleration': centripetal_acceleration,
        'centripetal_force': centripetal_force
    }

def calculate_tension_force(mass, gravity, angle):
    """
    Calculate the tension force in a string or rod for a conical pendulum.
    
    Parameters:
    -----------
    mass : float
        Mass of the object (kg)
    gravity : float
        Acceleration due to gravity (m/s²)
    angle : float
        Angle between the string and the vertical (radians)
    
    Returns:
    --------
    float
        Tension force in the string or rod (N)
    """
    # From the vertical force balance: T*cos(θ) = mg
    tension = mass * gravity / np.cos(angle)
    return tension

def visualize_conical_pendulum(length, angle, mass=1.0, gravity=9.8, save_path=None):
    """
    Visualize a conical pendulum system with its key parameters.
    
    Parameters:
    -----------
    length : float
        Length of the string or rod (m)
    angle : float
        Angle between the string and the vertical (radians)
    mass : float, optional
        Mass of the object (kg), default is 1.0 kg
    gravity : float, optional
        Acceleration due to gravity (m/s²), default is 9.8 m/s²
    save_path : str, optional
        Path to save the figure, if None, the figure is not saved
    
    Returns:
    --------
    None
    """
    # Set up font for Chinese characters if needed
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Calculate velocity and other parameters
    velocity = calculate_conical_pendulum_velocity(length, angle, gravity)
    radius = length * np.sin(angle)
    height = length * np.cos(angle)
    tension = calculate_tension_force(mass, gravity, angle)
    
    # Create figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the pendulum rod
    ax.plot([0, 0], [0, 0], [0, -length], 'k--', alpha=0.3, linewidth=1)  # Vertical reference
    ax.plot([0, radius * np.cos(0)], [0, radius * np.sin(0)], [0, -height], 'b-', linewidth=2)  # Pendulum rod
    
    # Plot the circular path
    theta = np.linspace(0, 2*np.pi, 100)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = -height * np.ones_like(theta)
    ax.plot(x, y, z, 'r--', alpha=0.7)
    
    # Plot the mass
    ax.scatter([radius], [0], [-height], color='red', s=100*mass, label=f'Mass: {mass} kg')
    
    # Add force vectors
    # Gravity force
    ax.quiver(radius, 0, -height, 0, 0, -0.2*mass*gravity, color='green', label=f'Weight: {mass*gravity:.2f} N')
    
    # Tension force components
    tension_x = -tension * np.sin(angle) * np.cos(0)
    tension_y = -tension * np.sin(angle) * np.sin(0)
    tension_z = tension * np.cos(angle)
    ax.quiver(radius, 0, -height, 0.2*tension_x, 0.2*tension_y, 0.2*tension_z, color='blue', label=f'Tension: {tension:.2f} N')
    
    # Set labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Conical Pendulum\nVelocity: {velocity:.2f} m/s, Radius: {radius:.2f} m')
    
    # Add text with parameters
    params_text = (
        f"Length: {length} m\n"
        f"Angle: {angle*180/np.pi:.1f}°\n"
        f"Velocity: {velocity:.2f} m/s\n"
        f"Radius: {radius:.2f} m\n"
        f"Height: {height:.2f} m\n"
        f"Tension: {tension:.2f} N"
    )
    ax.text2D(0.05, 0.05, params_text, transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Add legend
    ax.legend()
    
    # Adjust view
    ax.view_init(elev=20, azim=30)
    
    # Save figure if path is provided
    if save_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()

def animate_conical_pendulum(length, angle, duration=5, fps=30, save_path=None):
    """
    Create an animation of a conical pendulum.
    
    Parameters:
    -----------
    length : float
        Length of the string or rod (m)
    angle : float
        Angle between the string and the vertical (radians)
    duration : float, optional
        Duration of the animation in seconds, default is 5 seconds
    fps : int, optional
        Frames per second, default is 30
    save_path : str, optional
        Path to save the animation, if None, the animation is not saved
    
    Returns:
    --------
    None
    """
    # Set up font for Chinese characters if needed
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Calculate parameters
    velocity = calculate_conical_pendulum_velocity(length, angle, 9.8)
    radius = length * np.sin(angle)
    height = length * np.cos(angle)
    angular_velocity = velocity / radius  # rad/s
    
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Initialize pendulum rod and mass
    rod_line, = ax.plot([], [], [], 'b-', linewidth=2)
    mass_point, = ax.plot([], [], [], 'ro', markersize=10)
    
    # Plot the circular path
    theta = np.linspace(0, 2*np.pi, 100)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = -height * np.ones_like(theta)
    ax.plot(x, y, z, 'r--', alpha=0.5)
    
    # Plot the vertical reference
    ax.plot([0, 0], [0, 0], [0, -length], 'k--', alpha=0.3)
    
    # Set labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Conical Pendulum Animation\nVelocity: {velocity:.2f} m/s')
    
    # Set axis limits
    ax.set_xlim(-length, length)
    ax.set_ylim(-length, length)
    ax.set_zlim(-length, 0.2)
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Add text with parameters
    params_text = ax.text2D(0.05, 0.05, "", transform=ax.transAxes, 
                           bbox=dict(facecolor='white', alpha=0.8))
    
    def init():
        rod_line.set_data([], [])
        rod_line.set_3d_properties([])
        mass_point.set_data([], [])
        mass_point.set_3d_properties([])
        params_text.set_text("")
        return rod_line, mass_point, params_text
    
    def update(frame):
        # Calculate current angle
        current_angle = angular_velocity * frame / fps
        
        # Calculate position
        x = radius * np.cos(current_angle)
        y = radius * np.sin(current_angle)
        z = -height
        
        # Update rod
        rod_line.set_data([0, x], [0, y])
        rod_line.set_3d_properties([0, z])
        
        # Update mass
        mass_point.set_data([x], [y])
        mass_point.set_3d_properties([z])
        
        # Update text
        time = frame / fps
        params_text.set_text(
            f"Time: {time:.2f} s\n"
            f"Angle: {angle*180/np.pi:.1f}°\n"
            f"Velocity: {velocity:.2f} m/s\n"
            f"Angular position: {current_angle % (2*np.pi):.2f} rad"
        )
        
        return rod_line, mass_point, params_text
    
    # Create animation
    frames = int(duration * fps)
    ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, interval=1000/fps)
    
    # Save animation if path is provided
    if save_path:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        ani.save(save_path, writer='pillow', fps=fps)
    
    plt.tight_layout()
    plt.show()
    
    return ani

def main():
    """
    Main function to demonstrate the use of the circular motion solver tools.
    """
    print("Conical Pendulum (Tetherball) Problem Solver")
    print("--------------------------------------------")
    
    # Problem parameters
    m = 0.5  # Mass of the tetherball (kg)
    L = 2.0  # Length of the rope (m)
    theta = np.radians(30)  # Angle with vertical (30 degrees)
    g = 9.8  # Acceleration due to gravity (m/s²)
    
    # Calculate velocity
    velocity = calculate_conical_pendulum_velocity(L, theta, g)
    print(f"\nCalculated velocity: {velocity:.4f} m/s")
    
    # Calculate radius of the circular path
    radius = L * np.sin(theta)
    print(f"Radius of circular path: {radius:.4f} m")
    
    # Calculate other circular motion parameters
    params = calculate_circular_motion_parameters(velocity, radius, m)
    print("\nCircular Motion Parameters:")
    print(f"Period: {params['period']:.4f} s")
    print(f"Frequency: {params['frequency']:.4f} Hz")
    print(f"Angular velocity: {params['angular_velocity']:.4f} rad/s")
    print(f"Centripetal acceleration: {params['centripetal_acceleration']:.4f} m/s²")
    print(f"Centripetal force: {params['centripetal_force']:.4f} N")
    
    # Calculate tension in the rope
    tension = calculate_tension_force(m, g, theta)
    print(f"\nTension in the rope: {tension:.4f} N")
    
    # Verify the formula
    formula_result = np.sqrt(g * L * np.sin(theta)**2 / np.cos(theta))
    print(f"\nVerification using the formula v = √(gL·sin²θ/cosθ): {formula_result:.4f} m/s")
    
    # Create visualization
    print("\nCreating visualization...")
    visualize_conical_pendulum(L, theta, m, g, save_path="./images/conical_pendulum.png")
    
    # Optional: Create animation
    # Uncomment the following line to create an animation
    # animate_conical_pendulum(L, theta, duration=5, save_path="./images/conical_pendulum_animation.gif")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()