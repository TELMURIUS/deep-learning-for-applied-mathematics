import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt

L = 1.0      # Rod length
T = 0.1      # Total simulation time
alpha = 0.01 # Thermal diffusivity

Nx = 100      # Number of spatial points
Nt = 1000    # Number of time steps
dx = L / (Nx - 1)
dt = T / Nt

# Stability condition (CFL)
stability = alpha * dt / dx**2
if stability > 0.5:
    raise ValueError("CFL condition violated: reduce dt or increase dx")

x = np.linspace(0, L, Nx)
u = np.zeros((Nt + 1, Nx))
u[0, :] = np.sin(np.pi * x)

for n in range(Nt):
    for i in range(1, Nx - 1):
        u[n+1, i] = u[n, i] + stability * (u[n, i+1] - 2 * u[n, i] + u[n, i-1])

noise_amplitude = 0.01
u_noisy = u[-1] + noise_amplitude * np.random.randn(Nx)

plt.figure(figsize=(12, 8))
plt.plot(x, u[-1, :], label='Exact $u(x, T)$')
plt.plot(x, u_noisy, '--', label='Noisy $u(x, T)$')
plt.xlabel('x')
plt.ylabel('Temperature')
plt.title('Temperature distribution at time T')
plt.legend()
plt.grid()
plt.show()

lambdas = [0.001, 0.01, 0.1, 1, 10, 1000]
identity_matrix = np.eye(Nx)

plt.figure(figsize=(12, 8))
for lam in lambdas:
    u_recovered = np.linalg.solve(identity_matrix + lam * identity_matrix, u_noisy)
    plt.plot(x, u_recovered, label=f'λ = {lam}')

plt.plot(x, u[0], '--', label='True Initial Condition')
plt.xlabel('x')
plt.ylabel('Temperature')
plt.title('Recovered Initial Condition (Tikhonov Regularization)')
plt.legend()
plt.grid()
plt.show()
