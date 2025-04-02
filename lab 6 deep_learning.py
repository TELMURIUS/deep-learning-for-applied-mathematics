import numpy as np
import torch
import matplotlib
matplotlib.use('TkAgg')
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Parameters
alpha = 0.01
T1 = 0.1
Nx = 100
Nt = 1000

x = np.linspace(0, 1, Nx)

#initial condition
u_true = np.sin(np.pi * x)


def solve_heat_equation(Nx, Nt, alpha, T1):
    dx = 1 / (Nx - 1)
    dt = T1 / Nt
    r = alpha * dt / dx ** 2

    u = np.sin(np.pi * x)
    for _ in range(Nt):
        u_new = u.copy()
        for i in range(1, Nx - 1):
            u_new[i] = u[i] + r * (u[i - 1] - 2 * u[i] + u[i + 1])
        u = u_new
    return u


u_t1 = solve_heat_equation(Nx, Nt, alpha, T1)

u_t1_tensor = torch.tensor(u_t1, dtype=torch.float32).view(1, 1, Nx)
u_true_tensor = torch.tensor(u_true, dtype=torch.float32).view(1, 1, Nx)


class CNN_InverseHeat(nn.Module):
    def __init__(self):
        super(CNN_InverseHeat, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(32, 16, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=5, padding=2)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



model = CNN_InverseHeat()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


num_epochs = 500
loss_history = []
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(u_t1_tensor)
    loss = criterion(output, u_true_tensor)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
    if epoch % 10 == 0:
        print(f'epoch {epoch}, loss: {loss.item():}')


plt.plot(loss_history)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('loss curve')
plt.show()

predicted_u0 = model(u_t1_tensor).detach().numpy().flatten()

plt.plot(x, u_true, label='true initial condition', color='blue')
plt.plot(x, predicted_u0, label='recovered initial condition (CNN)', color='red')
plt.legend()
plt.show()
