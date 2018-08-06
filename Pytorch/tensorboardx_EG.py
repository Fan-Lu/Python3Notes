#####################################################
#   Tensorboard Data Visualization
#
#   Pytorch Version: 0.3.1
#   Usage:  python tensorboard_EG.py
#           tensorboard --logdir='./logs' --port=6006
######################################################

import torch
import torchvision.utils as vutils
import numpy as np
import torchvision.models as models
from torchvision import datasets
from tensorboardX import SummaryWriter
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision
from torchvision import transforms
from logger import Logger
from torch.autograd import Variable

# profit = pd.read_csv('result_saved/train_loss_profit.csv')
# profit = profit.values
# profit = profit[:, 1:402]
# profit = np.mean(profit, 1)
#
# loss = pd.read_csv('result_saved/train_loss.csv')
# loss = loss.values
# loss = loss[:, 1:402]
# loss = np.mean(loss, 1)
#
# args = GetParameters()
#
# demand = pd.read_csv('result_saved/train_demd.csv')
# demand = demand.values
# demand = demand[4950:5000, 1:402]
#
# a = demand.reshape(-1, 1)
# sns.distplot(a, rug=True, hist=True)
# plt.xlabel('Demand value')
# plt.ylabel('Probability')
# plt.title('Demand Probability Density Distribution')
# plt.savefig('/home/exx/Lab/rl4b/MatlabPlot/' + 'density')
# plt.show()

# Device configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MNIST dataset
dataset = torchvision.datasets.MNIST(root='../../data',
                                     train=True,
                                     transform=transforms.ToTensor(),
                                     download=True)

# Data loader
data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=100,
                                          shuffle=True)


# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size=784, hidden_size=500, num_classes=10):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


model = NeuralNet().cuda()

logger = Logger('./logs')

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

data_iter = iter(data_loader)
iter_per_epoch = len(data_loader)
total_step = 50000

# Start training
for step in range(total_step):

    # Reset the data_iter
    if (step + 1) % iter_per_epoch == 0:
        data_iter = iter(data_loader)

    # Fetch images and labels
    images, labels = next(data_iter)
    images, labels = Variable(images).view(images.size(0), -1).cuda(), Variable(labels).cuda()

    # Forward pass
    outputs = model(images)
    loss = criterion(outputs, labels)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Compute accuracy
    _, argmax = torch.max(outputs, 1)
    accuracy = (labels == argmax.squeeze()).float().mean()

    if (step + 1) % 100 == 0:
        print('Step [{}/{}], Loss: {:.4f}, Acc: {:.2f}'
              .format(step + 1, total_step, loss.cpu().data.numpy()[0], accuracy.cpu().data.numpy()[0]))

        # ================================================================== #
        #                        Tensorboard Logging                         #
        # ================================================================== #

        # 1. Log scalar values (scalar summary)
        info = {'loss': loss.cpu().data.numpy()[0], 'accuracy': accuracy.cpu().data.numpy()[0]}

        for tag, value in info.items():
            logger.scalar_summary(tag, value, step + 1)

        # 2. Log values and gradients of the parameters (histogram summary)
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            logger.histo_summary(tag, value.data.cpu().numpy(), step + 1)
            logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), step + 1)

        # 3. Log training images (image summary)
        info = {'images': images.view(-1, 28, 28)[:10].cpu().data.numpy()}

        for tag, images in info.items():
            logger.image_summary(tag, images, step + 1)