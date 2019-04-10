import torch
import matplotlib.pyplot as plt
import torch.nn as nn

N_SAMPLES = 20
N_HIDDEN = 300

x = torch.unsqueeze(torch.linspace(-1, 1, N_SAMPLES), 1)
y = x + 0.3 * torch.normal(torch.zeros(N_SAMPLES, 1), torch.ones(N_SAMPLES, 1))

test_x = torch.unsqueeze(torch.linspace(-1, 1, N_SAMPLES), 1)
test_y = test_x + 0.3 * torch.normal(torch.zeros(N_SAMPLES, 1), torch.ones(N_SAMPLES, 1))

plt.scatter(x.data.numpy(), y.data.numpy(), c = 'magenta', s = 50, alpha = 0.5, label = 'train')
plt.scatter(test_x.data.numpy(), test_y.data.numpy(), c = 'cyan', s = 50, alpha = 0.5, label = 'test')
plt.legend(loc = 'upper left')
plt.ylim(-2.5, 2.5)
plt.show()

net_overfitting = nn.Sequential(
        nn.Linear(1, N_HIDDEN),
        nn.ReLU(),
        nn.Linear(N_HIDDEN, N_HIDDEN),
        nn.ReLU(),
        nn.Linear(N_HIDDEN, 1),
        )


net_dropout = nn.Sequential(
        nn.Linear(1, N_HIDDEN),
        nn.Dropout(0.5),
        nn.ReLU(),
        nn.Linear(N_HIDDEN, N_HIDDEN),
        nn.Dropout(0.5),
        nn.ReLU(),
        nn.Linear(N_HIDDEN, 1),
        )

print(net_overfitting)
print(net_dropout)

optimizer_overfit = torch.optim.Adam(net_overfitting.parameters(), lr = 0.01)
optimizer_dropout = torch.optim.Adam(net_dropout.parameters(), lr = 0.01)
loss_func = nn.MSELoss()

plt.ion()

for i in range(500):
    pred_overfit = net_overfitting(x)
    pred_dropout = net_dropout(x)
    loss_overfit = loss_func(pred_overfit, y)
    loss_dropout = loss_func(pred_dropout, y)

    optimizer_overfit.zero_grad()
    loss_overfit.backward()
    optimizer_overfit.step()

    optimizer_dropout.zero_grad()
    loss_dropout.backward()
    optimizer_dropout.step()


    if i % 10 == 0:
        net_overfitting.eval()
        net_dropout.eval()
        plt.cla()

        test_pred_overfit = net_overfitting(test_x)
        test_pred_dropout = net_dropout(test_x)
        plt.scatter(x.data.numpy(), y.data.numpy(), c = 'magenta', s = 50, alpha = 0.3, label = 'train')
        plt.scatter(test_x.data.numpy(), test_y.data.numpy(), c = 'cyan', s = 50, alpha = 0.3, label = 'test')
        plt.plot(test_x.data.numpy(), test_pred_overfit.data.numpy(), 'r-', lw = 3, label = 'overfitting')
        plt.plot(test_x.data.numpy(), test_pred_dropout.data.numpy(), 'b-', lw = 3, label = 'dropout')
        plt.text(0, -1.2, 'overfit loss = %.4f'%loss_func(test_pred_overfit, test_y).data.numpy(), fontdict = {'size':20, 'color':'red'})
        plt.text(0, -1.5, 'dropout loss = %.4f'%loss_func(test_pred_dropout, test_y).data.numpy(), fontdict = {'size':20, 'color':'red'})
        plt.legend(loc = 'upper left')
        plt.ylim(-2.5, 2.5)
        plt.pause(0.1)

    net_overfitting.train()
    net_dropout.train()

plt.ioff()
plt.show()

