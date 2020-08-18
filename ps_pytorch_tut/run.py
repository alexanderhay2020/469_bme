from __future__ import print_function

import torch
import torchvision

import matplotlib.pyplot as plt
from torchvision import transforms, datasets

train = datasets.MNIST("",
                       train=True,
                       download=True,
                       transform = transforms.Compose([transforms.ToTensor()]))

test = datasets.MNIST("",
                       train=False,
                       download=True,
                       transform = transforms.Compose([transforms.ToTensor()]))

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)

# for data in trainset:
#     print(data)
#     break
#
# x,y = data[0][0], data[1][0]
#
# print(y)
#
# plt.imshow(x.view(28,28))
# # plt.imshow(data[0][0])
# # print(data[0][0])
# plt.show()

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 64)   # fc1 = fully connected layer 1, 28x28=784
        self.fc2 = nn.Linear(64, 64)    # 64 inputs, 64 outputs
        self.fc3 = nn.Linear(64, 64)    # 64 inputs, 64 outputs
        self.fc4 = nn.Linear(64, 10)    # 64 inputs, 10 outputs (numbers 0-9)

    def forward(self, x):
        x = F.relu(self.fc1(x))         # pass information through network, with relu sctivation function
        x = F.relu(self.fc2(x))         # activation function only runs on output information
        x = F.relu(self.fc3(x))
        x = self.fc4(x)                 # we only want 1 neuron to ultimately fire

        return x

net = Net()
print(net)
