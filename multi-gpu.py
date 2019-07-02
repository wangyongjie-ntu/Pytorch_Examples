#Filename:	multi-gpu.py
#Author:	Wang Yongjie
#Email:		yongjie.wang@ntu.edu.sg
#Date:		Sen 01 Jul 2019 04:04:13  +08

import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import os
from torch.autograd import Variable

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

train_loader = Data.DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True)

test_data = torchvision.datasets.MNIST(root = './mnist/', train = False)
test_x = torch.unsqueeze(test_data.data, dim = 1).type(torch.FloatTensor)[:2000]/255.
test_y = test_data.targets[:2000]

class MNIST(nn.Module):

    def __init__(self):
        super(MNIST, self).__init__()

        self.conv1 = nn.Sequential(
                nn.Conv2d(1, 16, 5, 1, 2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 2),
                )
        self.conv2 = nn.Sequential(
                nn.Conv2d(16, 32, 5, 1, 2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = 2),
                )

        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.fc(x)
        return output


MnistMulti = nn.DataParallel(MNIST()).cuda()
optimizer = torch.optim.Adam(MnistMulti.parameters(), lr = lr)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):
        b_x = Variable(b_x.cuda(async = True))
        b_y = Variable(b_y.cuda(async = True))
        output = MnistMulti(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pred = torch.max(output, 1)[1]
        acc = torch.sum(pred == b_y).item() / batch_size
        print(acc)

 
