import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
import numpy as np
import os
from dataset import MyDataset
from model.model import Model


train_data = MyDataset()
print('size of train_data: %d' % len(train_data))
val_data = MyDataset(size=500, random_offset=33333)
print('size of val_data: %d' % len(val_data))
test_data = MyDataset(size=500, random_offset=99999)
print('size of test_data: %d' % len(test_data))

trainloader = DataLoader(train_data, batch_size=128, shuffle=True)
valloader = DataLoader(train_data, batch_size=128, shuffle=True)
testloader = DataLoader(train_data, batch_size=1, shuffle=True)

model = Model()
# training
for i in range(100):
    loss = 0
    for _, data in enumerate(trainloader):
        input, label = data
        model.set_input(input, label)
        model.optimize()
        loss += model.loss.item()
    print('epoch: %d;' % (i + 1), 'loss: %.2f' % (loss / len(trainloader)))
model.save_net()
print('saving network...')

#val
loss = 0
for i, data in enumerate(valloader):
    input, label = data
    model.set_input(input, label)
    model.test('/content/drive/My Drive/Colab Notebooks/lab/model.pkl')
    loss += model.loss.item()
print('Val_MSE: %.3f' % (loss / len(valloader)))

# test
loss = 0
for i, data in enumerate(testloader):
    input, label = data
    model.set_input(input, label)
    model.test('/content/drive/My Drive/Colab Notebooks/lab/model.pkl')
    loss += model.loss.item()
print('Test_MSE: %.3f' % (loss / len(testloader)))


def show(label, pred, input):
    X = np.arange(40)
    L = label[0][0] * X + label[0][1]
    P = pred[0][0] * X + pred[0][1]
    plt.figure(figsize=(5, 5))
    plt.plot(X, 40 - L, color="yellow", linewidth=3.0)
    plt.plot(X, 40 - P, color="red", linewidth=3.0)

    plt.imshow(input[0][0])
    plt.show()


# test
loss = 0
for i, data in enumerate(testloader):
    input, label = data
    model.set_input(input, label)
    pred = model.test('/content/drive/My Drive/Colab Notebooks/lab/model_3.pkl')
    show(label.cpu().detach().numpy(), pred.cpu().detach().numpy(), input.cpu().detach().numpy())

