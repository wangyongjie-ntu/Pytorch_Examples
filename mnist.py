import torch
import os
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

EPOCH = 1
batch_size = 50
lr = 0.001
download_mnist = False

if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
        download_mnist = True

train_data = torchvision.datasets.MNIST(
        root =  './mnist/',
        train = True,
        transform = torchvision.transforms.ToTensor(),
        download = download_mnist,
        )

print(train_data.data.size())
print(train_data.targets.size())
plt.imshow(train_data.data[0].numpy(), cmap = "gray")
plt.title("%i"%train_data.targets[0])
plt.show()

train_loader = Data.DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True)

test_data = torchvision.datasets.MNIST(root = './mnist/', train = False)
test_x = torch.unsqueeze(test_data.data, dim = 1).type(torch.FloatTensor)[:2000]/255.
test_y = test_data.targets[:2000]

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
                nn.Conv2d(
                    in_channels = 1,
                    out_channels = 16,
                    kernel_size = 5,
                    stride =  1,
                    padding = 2,
                    ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 2),
                )
        self.conv2 = nn.Sequential(
                nn.Conv2d(16, 32, 5, 1, 2),
                nn.ReLU(),
                nn.MaxPool2d(2),
                )

        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x



cnn = CNN()
print(cnn)

optimizer = torch.optim.Adam(cnn.parameters(), lr = lr)
loss_func = nn.CrossEntropyLoss()

from matplotlib import cm
try:
    from sklearn.manifold import TSNE;
    HAS_SK = True
except:
    HAS_SK = False
    print("Install sklean for layer visualization")


def plot_with_labels(lowWeights, labels):
    plt.cla()
    X, Y = lowWeights[:, 0], lowWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / 9));
        plt.text(x, y, s, backgroundcolor = c, fontsize = 9)
        plt.xlim(X.min(), X.max())
        plt.ylim(Y.min(), Y.max())
        plt.title("Visualize last layer")
        plt.show()
        plt.pause(0.01)


plt.ion()

for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):

        output = cnn(b_x)[0]
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_output, last_layer = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.numpy()

            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))

            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

            if HAS_SK:
                tsne = TSNE(perplexity = 30, n_components = 2, n_iter = 5000)
                plot_only = 500
                low_dim_embs = tsne.fit_transform(last_layer.data.numpy()[:plot_only, :])
                labels = test_y.numpy()[:plot_only]
                plot_with_labels(low_dim_embs, labels)



plt.ioff()

test_output, _ = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy()

print(pred_y, "prediction number")
print(test_y[:10].numpy(), "real number")

