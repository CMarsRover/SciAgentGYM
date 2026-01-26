# Filename: friction_dynamics_solver.py

import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Union, Tuple, Dict, Optional, List

def convert_speed(speed: float, from_unit: str, to_unit: str) -> float:
    """
    Convert speed between different units.
    
    Parameters:
    -----------
    speed : float
        The speed value to convert
    from_unit : str
        The original unit ('m/s', 'km/h', 'mph', 'ft/s')
    to_unit : str
        The target unit ('m/s', 'km/h', 'mph', 'ft/s')
    
    Returns:
    --------
    float
        The converted speed value in the target unit
    
    Examples:
    ---------
    >>> convert_speed(55, 'mph', 'm/s')
    24.5872
    """
    # Conversion factors to m/s
    to_ms = {
        'm/s': 1.0,
        'km/h': 1/3.6,
        'mph': 0.44704,
        'ft/s': 0.3048
    }
    
    # Conversion factors from m/s
    from_ms = {
        'm/s': 1.0,
        'km/h': 3.6,
        'mph': 1/0.44704,
        'ft/s': 1/0.3048
    }
    
    # Convert to m/s first, then to target unit
    speed_ms = speed * to_ms[from_unit]
    return speed_ms * from_ms[to_unit]

def calculate_acceleration(initial_velocity: float, final_velocity: float, time: float, 
                          units: str = 'm/s') -> float:
    """
    Calculate constant acceleration based on initial velocity, final velocity, and time.
    
    Parameters:
    -----------
    initial_velocity : float
        The starting velocity (in specified units)
    final_velocity : float
        The ending velocity (in specified units)
    time : float
        The time taken for the velocity change (in seconds)
    units : str, optional
        The units of the velocities ('m/s', 'km/h', 'mph', 'ft/s')
        
    Returns:
    --------
    float
        The constant acceleration in m/s²
    
    Examples:
    ---------
    >>> calculate_acceleration(0, 55, 24.0, 'mph')
    1.024
    """
    # Convert velocities to m/s if needed
    if units != 'm/s':
        initial_velocity = convert_speed(initial_velocity, units, 'm/s')
        final_velocity = convert_speed(final_velocity, units, 'm/s')
    
    # Calculate acceleration
    acceleration = (final_velocity - initial_velocity) / time
    
    return acceleration

def calculate_friction_coefficient(mass: float, acceleration: float, 
                                  angle: float = 0.0, gravity: float = 9.8) -> Dict[str, float]:
    """
    Calculate the minimum coefficient of friction needed to prevent an object from sliding.
    
    This function calculates both static and kinetic friction coefficients needed to prevent
    an object from sliding on a surface during acceleration. It accounts for inclined surfaces.
    
    Parameters:
    -----------
    mass : float
        The mass of the object in kg
    acceleration : float
        The acceleration of the system in m/s²
    angle : float, optional
        The angle of inclination in degrees (0 for horizontal surface)
    gravity : float, optional
        The gravitational acceleration in m/s² (default: 9.8)
    
    Returns:
    --------
    Dict[str, float]
        Dictionary containing 'static' and 'kinetic' friction coefficients
    
    Examples:
    ---------
    >>> calculate_friction_coefficient(789, 1.024)
    {'static': 0.104, 'kinetic': 0.104}
    """
    # Convert angle to radians
    angle_rad = np.radians(angle)
    
    # Calculate the normal force
    normal_force = mass * gravity * np.cos(angle_rad)
    
    # Calculate the force due to acceleration
    acceleration_force = mass * acceleration
    
    # Calculate the force due to gravity along the incline
    gravity_force = mass * gravity * np.sin(angle_rad)
    
    # Calculate the total force that needs to be counteracted by friction
    total_force = acceleration_force + gravity_force
    
    # Calculate the minimum coefficient of static friction
    static_friction = total_force / normal_force
    
    # For this problem, the kinetic friction would be the same minimum value
    # In reality, kinetic friction is typically less than static friction
    kinetic_friction = static_friction
    
    return {
        'static': round(static_friction, 3),
        'kinetic': round(kinetic_friction, 3)
    }

def analyze_cargo_stability(cargo_mass: float, vehicle_mass: float, 
                           acceleration_profile: Dict[str, Union[float, str]],
                           surface_properties: Dict[str, float] = None,
                           visualize: bool = False) -> Dict[str, Union[float, bool]]:
    """
    Analyze the stability of cargo on a moving vehicle.
    
    This function determines whether cargo will remain stable on a vehicle during
    acceleration, braking, or turning, based on friction properties.
    
    Parameters:
    -----------
    cargo_mass : float
        The mass of the cargo in kg
    vehicle_mass : float
        The mass of the vehicle in kg
    acceleration_profile : Dict
        Dictionary containing acceleration information:
        - 'initial_speed': float - Starting speed
        - 'final_speed': float - Ending speed
        - 'time': float - Time taken for speed change
        - 'units': str - Speed units ('m/s', 'km/h', 'mph')
        - 'direction': str - 'forward', 'braking', or 'turning'
    surface_properties : Dict, optional
        Dictionary containing surface properties:
        - 'friction_coefficient': float - Known coefficient of friction
        - 'angle': float - Inclination angle in degrees
    visualize : bool, optional
        Whether to generate visualization of the analysis
    
    Returns:
    --------
    Dict
        Dictionary containing analysis results:
        - 'required_friction': float - Required coefficient of friction
        - 'is_stable': bool - Whether cargo will remain stable
        - 'safety_factor': float - Ratio of available to required friction
    
    Examples:
    ---------
    >>> analyze_cargo_stability(
    ...     789, 9750,
    ...     {'initial_speed': 0, 'final_speed': 55, 'time': 24.0, 'units': 'mph', 'direction': 'forward'},
    ...     {'friction_coefficient': 0.15, 'angle': 0}
    ... )
    {'required_friction': 0.104, 'is_stable': True, 'safety_factor': 1.44}
    """
    # Extract acceleration profile information
    initial_speed = acceleration_profile.get('initial_speed', 0)
    final_speed = acceleration_profile.get('final_speed', 0)
    time = acceleration_profile.get('time', 0)
    units = acceleration_profile.get('units', 'm/s')
    direction = acceleration_profile.get('direction', 'forward')
    
    # Calculate acceleration
    acceleration = calculate_acceleration(initial_speed, final_speed, time, units)
    
    # Adjust acceleration sign based on direction
    if direction == 'braking':
        acceleration = -acceleration
    
    # Set default surface properties if not provided
    if surface_properties is None:
        surface_properties = {'friction_coefficient': 0.0, 'angle': 0.0}
    
    angle = surface_properties.get('angle', 0.0)
    available_friction = surface_properties.get('friction_coefficient', 0.0)
    
    # Calculate required friction coefficient
    friction_result = calculate_friction_coefficient(cargo_mass, acceleration, angle)
    required_friction = friction_result['static']
    
    # Determine if cargo is stable
    is_stable = available_friction >= required_friction
    
    # Calculate safety factor
    safety_factor = available_friction / required_friction if required_friction > 0 else float('inf')
    safety_factor = round(safety_factor, 2)
    
    # Create visualization if requested
    if visualize:
        create_stability_visualization(cargo_mass, vehicle_mass, acceleration, 
                                      required_friction, available_friction)
    
    return {
        'required_friction': required_friction,
        'is_stable': is_stable,
        'safety_factor': safety_factor
    }

def create_stability_visualization(cargo_mass: float, vehicle_mass: float, 
                                  acceleration: float, required_friction: float,
                                  available_friction: float = None) -> None:
    """
    Create a visualization of cargo stability analysis.
    
    Parameters:
    -----------
    cargo_mass : float
        The mass of the cargo in kg
    vehicle_mass : float
        The mass of the vehicle in kg
    acceleration : float
        The acceleration in m/s²
    required_friction : float
        The required coefficient of friction
    available_friction : float, optional
        The available coefficient of friction
    
    Returns:
    --------
    None
        Saves the visualization to a file
    """
    # Set up plot with proper font support
    plt.figure(figsize=(10, 6))
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Create directory for images if it doesn't exist
    os.makedirs('./images', exist_ok=True)
    
    # Plot required friction vs acceleration
    accelerations = np.linspace(0, acceleration * 2, 100)
    frictions = [calculate_friction_coefficient(cargo_mass, a)['static'] for a in accelerations]
    
    plt.plot(accelerations, frictions, 'b-', linewidth=2, label='所需摩擦系数')
    plt.axvline(x=acceleration, color='r', linestyle='--', label=f'当前加速度 ({acceleration:.2f} m/s²)')
    plt.axhline(y=required_friction, color='g', linestyle='--', label=f'所需摩擦系数 ({required_friction:.3f})')
    
    if available_friction is not None:
        plt.axhline(y=available_friction, color='purple', linestyle='-', 
                   label=f'可用摩擦系数 ({available_friction:.3f})')
    
    plt.xlabel('加速度 (m/s²)')
    plt.ylabel('摩擦系数')
    plt.title('货物稳定性分析')
    plt.grid(True)
    plt.legend()
    
    # Save the figure
    plt.savefig('./images/cargo_stability_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """
    Main function to demonstrate the use of the friction dynamics solver.
    """
    # Problem parameters
    cargo_mass = 789  # kg
    vehicle_mass = 9750  # kg
    initial_speed = 0  # mph
    final_speed = 55  # mph
    time = 24.0  # seconds
    
    # Calculate acceleration
    acceleration = calculate_acceleration(initial_speed, final_speed, time, 'mph')
    print(f"Acceleration: {acceleration:.3f} m/s²")
    
    # Calculate required friction coefficient
    friction_result = calculate_friction_coefficient(cargo_mass, acceleration)
    print(f"Required static friction coefficient: {friction_result['static']}")
    
    # Perform full stability analysis
    stability_result = analyze_cargo_stability(
        cargo_mass, 
        vehicle_mass,
        {
            'initial_speed': initial_speed,
            'final_speed': final_speed,
            'time': time,
            'units': 'mph',
            'direction': 'forward'
        },
        {
            'friction_coefficient': 0.15,  # Example available friction
            'angle': 0
        },
        visualize=True
    )
    
    print("\nStability Analysis Results:")
    print(f"Required friction coefficient: {stability_result['required_friction']}")
    print(f"Is cargo stable? {'Yes' if stability_result['is_stable'] else 'No'}")
    print(f"Safety factor: {stability_result['safety_factor']}")
    
    # The answer to the specific problem
    print("\nAnswer to the problem:")
    print(f"The minimum coefficient of static friction needed is {friction_result['static']}")

if __name__ == "__main__":
    main()