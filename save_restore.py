import torch

torch.manual_seed(0)


x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim = 1)
y = x.pow(2) + 0.2 * torch.rand(x.size())


def save():
    net1 = torch.nn.Sequential(
            torch.nn.Linear(1, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 1),
            )

    optimizer = torch.optim.SGD(net1.parameters(), lr = 0.2)
    loss_func = torch.nn.MSELoss()

    for t in range(100):
        prediction = net1(x)
        loss = loss_func(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(net1, "net.pkl")
    torch.save(net1.state_dict(), "net_params.pkl")


def restore_net():
    net2 = torch.load("net.pkl")
    print(net2)
    prediction = net2(x)

def restore_param():
    net3 = torch.nn.Sequential(
            torch.nn.Linear(1, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 1),
            )

    net3.load_state_dict(torch.load("net_params.pkl"))
    prediction = net3(x)

# save()
restore_net()
restore_param()
