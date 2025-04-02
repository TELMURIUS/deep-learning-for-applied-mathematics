import numpy as np
import torch
import matplotlib
matplotlib.use('TkAgg')
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


#parameters
alpha = 0.01
T1 = 0.1
N_u = 100
N_f = 1000


x_u = torch.linspace(0, 1, N_u).view(-1, 1)
t_u = T1 * torch.ones_like(x_u)
u_true = torch.sin(np.pi * x_u)


x_f = torch.rand(N_f, 1)
t_f = torch.rand(N_f, 1) * T1


class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 1)
        )

    def forward(self, x, t):
        return self.net(torch.cat((x, t), dim=1))


def loss_function(model, x_u, t_u, u_true, x_f, t_f):
    u_pred = model(x_u, t_u)
    loss_data = torch.mean((u_pred - u_true) ** 2)

    x_f.requires_grad, t_f.requires_grad = True, True
    u_f = model(x_f, t_f)
    u_t = torch.autograd.grad(u_f, t_f, torch.ones_like(u_f), create_graph=True)[0]
    u_x = torch.autograd.grad(u_f, x_f, torch.ones_like(u_f), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_f, torch.ones_like(u_x), create_graph=True)[0]
    loss_pde = torch.mean((u_t - alpha * u_xx) ** 2)

    x_bc = torch.tensor([[0.0], [1.0]])
    t_bc = torch.rand(2, 1) * T1
    u_bc = model(x_bc, t_bc)
    loss_bc = torch.mean(u_bc ** 2)

    return loss_data + loss_pde + loss_bc


model = PINN()
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_history = []
for epoch in range(500):
    optimizer.zero_grad()
    loss = loss_function(model, x_u, t_u, u_true, x_f, t_f)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

plt.plot(loss_history, label="loss during learning")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('learning learning')
plt.show()


x_test = torch.linspace(0, 1, 100).view(-1, 1)
t_test = torch.zeros_like(x_test)
u_pred = model(x_test, t_test).detach().numpy()


plt.plot(x_test.numpy(), np.sin(np.pi * x_test.numpy()), label="true u(x,0)", color='blue')
plt.plot(x_test.numpy(), u_pred, label="recovered u(x,0)",color='red')
plt.legend()
plt.show()
