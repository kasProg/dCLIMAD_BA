import torch
import matplotlib.pyplot as plt


def add_multiplicative_noise(E, a, b, c, seed=None):
    """
    Adds multiplicative noise to a trend signal of the form aE + b.

    Parameters:
    E (torch.Tensor): The input variable (positive).
    a (float): The coefficient for the linear term.
    b (float): The intercept.
    c (float): The scaling parameter for the noise.
    seed (int, optional): Seed for reproducibility. Default is None.

    Returns:
    torch.Tensor: The noise-added signal.
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Calculate the trend signal
    signal = a * E + b

    # Compute the multiplicative noise
    noise = c * (E - E.min()) * torch.randn_like(E)

    # Return the noise-added signal
    noise_added_signal = signal + noise
    return noise, noise_added_signal


# Example usage
E = torch.linspace(1.0, 10.0, 100)  # Define the E variable (must be positive)
a = 2.0  # Coefficient for the linear term
b = 5.0  # Intercept
c = 0.2  # Noise scaling parameter
seed = 42  # For reproducibility

# Generate noise and noise-added signal
noise, noise_added_signal = add_multiplicative_noise(E, a, b, c, seed)

# Plot E vs noise-added signal
plt.figure(figsize=(8, 6))
plt.plot(E.numpy(), (a * E + b).numpy(), label='Original Signal', linestyle='--')
plt.plot(E.numpy(), noise_added_signal.numpy(), label='Noise-Added Signal')
plt.xlabel('E')
plt.ylabel('Signal')
plt.title('E vs Noise-Added Signal')
plt.legend()
plt.grid(True)
plt.show()