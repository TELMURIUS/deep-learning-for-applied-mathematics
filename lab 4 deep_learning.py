import numpy as np
import torch
import matplotlib
matplotlib.use('TkAgg')
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


def initial_condition(x):
    return np.sin(np.pi * x)

def solve_burgers(u0, Nx, Nt, dx, dt, mu):
    u = np.copy(u0)
    u_new = np.copy(u)
    for _ in range(Nt):
        u_new[1:-1] = u[1:-1] - dt * u[1:-1] * (u[1:-1] - u[:-2]) / dx \
                      + mu * dt / dx**2 * (u[2:] - 2 * u[1:-1] + u[:-2])
        u_new[0] = u_new[-1] = 0
        u[:] = u_new
    return u


#parameters
x_max = 1
Nx = 101
Nt = 100
dx = x_max / (Nx - 1)
dt = 0.001
mu = 0.01

x = np.linspace(0, x_max, Nx)


u0_true = initial_condition(x)
u_T = solve_burgers(u0_true, Nx, Nt, dx, dt, mu)


u_noisy = u_T + 0.05 * np.random.randn(Nx)


class SimpleNN(nn.Module):
    def __init__(self, n):
        super(SimpleNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(n, 50),
            nn.ReLU(),
            nn.Linear(50, n)
        )

    def forward(self, x):
        return self.fc(x)

model = SimpleNN(Nx)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

tensor_x = torch.tensor(u_noisy, dtype=torch.float32).unsqueeze(0)
tensor_y = torch.tensor(u0_true, dtype=torch.float32).unsqueeze(0)

loss_history = []
n_epoch = 500
for epoch in range(n_epoch):
    optimizer.zero_grad()
    output = model(tensor_x)
    loss = loss_fn(output, tensor_y)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
    if epoch % 10 == 0:
        print(f'epoch {epoch}, error: {loss.item():.6f}')

plt.plot(loss_history)
plt.xlabel('epoch')
plt.ylabel('error')
plt.title('learning learning')
plt.show()

u0_pred = model(tensor_x).detach().numpy().flatten()

plt.plot(x, u0_true, label='true', color= 'red')
plt.plot(x, u0_pred, label='predition', color = 'blue')
plt.legend()
plt.xlabel('x')
plt.ylabel('u(x,0)')
plt.title('initial condition recovery')
plt.show()
