import numpy as np
import matplotlib.pyplot as plt
import torch



#================FORWARD DIFFUSION========================

# Forward diffusion for N steps in 1D.
def forward_diffusion_1D(x0, noise_strength_fn, t0, nsteps, dt):
    """
    Parameters:
    - x0: Initial sample value (scalar)
    - noise_strength_fn: Function of time, outputs scalar noise strength
    - t0: Initial time
    - nsteps: Number of diffusion steps
    - dt: Time step size

    Returns:
    - x: Trajectory of sample values over time
    - t: Corresponding time points for the trajectory
    """

    # Initialize the trajectory array
    x = np.zeros(nsteps + 1)
    
    # Set the initial sample value
    x[0] = x0

    # Generate time points for the trajectory
    t = t0 + np.arange(nsteps + 1) * dt

    # Perform Euler-Maruyama time steps for diffusion simulation
    for i in range(nsteps):

        # Get the noise strength at the current time
        noise_strength = noise_strength_fn(t[i])

        # Generate a random normal variable
        random_normal = np.random.randn()

        # Update the trajectory using Euler-Maruyama method
        x[i + 1] = x[i] + random_normal * noise_strength

    # Return the trajectory and corresponding time points
    return x, t


# Example noise strength function: always equal to 1
def noise_strength_constant(t):
    """
    Example noise strength function that returns a constant value (1).

    Parameters:
    - t: Time parameter (unused in this example)

    Returns:
    - Constant noise strength (1)
    """
    return 1


#================REVERSE DIFFUSION========================

# Reverse diffusion for N steps in 1D.
def reverse_diffusion_1D(x0, noise_strength_fn, score_fn, T, nsteps, dt):
    """
    Parameters:
    - x0: Initial sample value (scalar)
    - noise_strength_fn: Function of time, outputs scalar noise strength
    - score_fn: Score function
    - T: Final time
    - nsteps: Number of diffusion steps
    - dt: Time step size

    Returns:
    - x: Trajectory of sample values over time
    - t: Corresponding time points for the trajectory
    """

    # Initialize the trajectory array
    x = np.zeros(nsteps + 1)
    
    # Set the initial sample value
    x[0] = x0

    # Generate time points for the trajectory
    t = np.arange(nsteps + 1) * dt

    # Perform Euler-Maruyama time steps for reverse diffusion simulation
    for i in range(nsteps):

        # Calculate noise strength at the current time
        noise_strength = noise_strength_fn(T - t[i])

        # Calculate the score using the score function
        score = score_fn(x[i], 0, noise_strength, T - t[i])

        # Generate a random normal variable
        random_normal = np.random.randn()

        # Update the trajectory using the reverse Euler-Maruyama method
        x[i + 1] = x[i] + score * noise_strength**2 * dt + noise_strength * random_normal * np.sqrt(dt)

    # Return the trajectory and corresponding time points
    return x, t

# Example score function: always equal to 1
def score_simple(x, x0, noise_strength, t):
    """
    Parameters:
    - x: Current sample value (scalar)
    - x0: Initial sample value (scalar)
    - noise_strength: Scalar noise strength at the current time
    - t: Current time

    Returns:
    - score: Score calculated based on the provided formula
    """

    # Calculate the score using the provided formula
    score = - (x - x0) / ((noise_strength**2) * t)

    # Return the calculated score
    return score


def test_forward_diffusion():
    # Number of diffusion steps
    nsteps = 100

    # Initial time
    t0 = 0

    # Time step size
    dt = 0.1

    # Noise strength function
    noise_strength_fn = noise_strength_constant

    # Initial sample value
    x0 = 0

    # Number of tries for visualization
    num_tries = 5

    # Setting larger width and smaller height for the plot
    plt.figure(figsize=(15, 5))

    # Loop for multiple trials
    for i in range(num_tries):

        # Simulate forward diffusion
        x, t = forward_diffusion_1D(x0, noise_strength_fn, t0, nsteps, dt)

        # Plot the trajectory
        plt.plot(t, x, label=f'Trial {i+1}')  # Adding a label for each trial

    # Labeling the plot
    plt.xlabel('Time', fontsize=20)
    plt.ylabel('Sample Value ($x$)', fontsize=20)

    # Title of the plot
    plt.title('Forward Diffusion Visualization', fontsize=20)

    # Adding a legend to identify each trial
    plt.legend()

    # Show the plot
    plt.show()

def test_reverse_diffusion():
    # Number of reverse diffusion steps
    nsteps = 100

    # Initial time for reverse diffusion
    t0 = 0

    # Time step size for reverse diffusion
    dt = 0.1

    # Function defining constant noise strength for reverse diffusion
    noise_strength_fn = noise_strength_constant

    # Example score function for reverse diffusion
    score_fn = score_simple

    # Initial sample value for reverse diffusion
    x0 = 0

    # Final time for reverse diffusion
    T = 11

    # Number of tries for visualization
    num_tries = 5

    # Setting larger width and smaller height for the plot
    plt.figure(figsize=(15, 5))

    # Loop for multiple trials
    for i in range(num_tries):
        # Draw from the noise distribution, which is diffusion for time T with noise strength 1
        x0 = np.random.normal(loc=0, scale=T)

        # Simulate reverse diffusion
        x, t = reverse_diffusion_1D(x0, noise_strength_fn, score_fn, T, nsteps, dt)

        # Plot the trajectory
        plt.plot(t, x, label=f'Trial {i+1}')  # Adding a label for each trial

    # Labeling the plot
    plt.xlabel('Time', fontsize=20)
    plt.ylabel('Sample Value ($x$)', fontsize=20)

    # Title of the plot
    plt.title('Reverse Diffusion Visualized', fontsize=20)

    # Adding a legend to identify each trial
    plt.legend()

    # Show the plot
    plt.show()


if __name__ == "__main__":
    # Test the forward diffusion function
    test_forward_diffusion()
    # Test the reverse diffusion function
    test_reverse_diffusion()