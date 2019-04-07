import torch
from torch import nn
import torchvision.datasets as Datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


Epoch = 1
Batch_size = 64
Time_step = 28
Input_size = 28
Lr = 0.01


train_data = Datasets.MNIST(
        root = "./mnist/",
        train = True,
        transform = transforms.ToTensor(),
        download = False,
        )

print(train_data.train_data.size())
print(train_data.train_labels.size())
plt.imshow(train_data.train_data[0].numpy(), cmap = "gray")
plt.title("%i"%train_data.targets[0])
plt.show()

train_loader = torch.utils.data.DataLoader(dataset = train_data,
        batch_size = Batch_size,
        shuffle = True)

test_data = Datasets.MNIST(
        root = "./mnist/",
        train = False,
        transform = transforms.ToTensor(),
        )

test_x = test_data.test_data.type(torch.FloatTensor)[:2000] / 255.
test_y = test_data.test_labels.numpy()[:2000]

class LSTM(nn.Module):

    def __init__(self):
        super(LSTM, self).__init__()

        self.lstm = nn.LSTM(
                input_size = Input_size,
                hidden_size = 64,
                num_layers = 1,
                batch_first = True,
                )
        self.out = nn.Linear(64, 10)


    def forward(self, x):
        r_out, (h_n, h_c) = self.lstm(x, None)
        out = self.out(r_out[:, -1,:])
        return out

lstm = LSTM()
print(lstm)

optimizer = torch.optim.Adam(lstm.parameters(), lr = Lr)
loss_func = nn.CrossEntropyLoss()

for epoch in range(Epoch):
    for step, (b_x, b_y) in enumerate(train_loader):
        b_x = b_x.view(-1, 28, 28)
        output = lstm(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_output = lstm(test_x)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuarcy = float((pred_y == test_y).astype(int).sum()) / float(test_y.size)
            print("Epoch: ", epoch, " | train loss:%.4f"%loss.item(), " |test accuarcy: %.2f"%accuarcy)


test_output = lstm(test_x[:10].view(-1, 28, 28))
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y, "prediction number")
print(test_y[:10], "real number")
