import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt

L = 1.0  # Rod length
T = 0.1  # Total time
alpha = 0.01  # Thermal diffusivity

Nx = 100  # Space points
Nt = 1000  # Time steps
dx = L / (Nx - 1)
dt = T / Nt

# Stability condition (CFL)
if alpha * dt / dx ** 2 > 0.5:
    raise ValueError("ERROR")

x = np.linspace(0, L, Nx)
u = np.zeros((Nt + 1, Nx))
u[0, :] = np.sin(np.pi * x)

for n in range(Nt):
    for i in range(1, Nx - 1):
        u[n + 1, i] = u[n, i] + alpha * dt / dx ** 2 * (u[n, i + 1] - 2 * u[n, i] + u[n, i - 1])

noise_amplitude = 0.01
u_noisy = u[-1] + noise_amplitude * np.random.randn(Nx)

plt.figure(figsize=(12, 8))
plt.plot(x, u[-1], label='exact $u(x, T)$')
plt.plot(x, u_noisy, '--', label='noisy $u(x, T)$')
plt.legend()
plt.xlabel('x')
plt.ylabel('temperature')
plt.title('temperature distribution at T')
plt.grid()
plt.show()

# Gradient descent algorithm + Tikhonov regularization
eta_values = [0.01, 0.1, 1.0]  # Different learning rates
lambda_val = 0.01  # Regularization parameter
max_iter = 1000  # Number of iterations

plt.figure(figsize=(12, 8))

for eta in eta_values:
    f = np.zeros(Nx)
    for k in range(max_iter):
        grad = f - u_noisy + lambda_val * f
        f = f - eta * grad
    plt.plot(x, f, label=f'η = {eta}')

plt.plot(x, u[0], '--', label='true initial condition')
plt.legend()
plt.xlabel('x')
plt.ylabel('temperature')
plt.title('recovered initial condition using  Gradient descent algorithm')
plt.grid()
plt.show()
