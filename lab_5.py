# -*- coding: utf-8 -*-

import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
import numpy as np
import easydict
import os

seed = 7
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

class MyDataset(Dataset):
  def __init__(self, size=5000, dim=40, random_offset=0):
        super(MyDataset, self).__init__()
        self.size = size
        self.dim = dim
        self.random_offset = random_offset

  def __getitem__(self, index):
      if index >= len(self):
          raise IndexError("{} index out of range".format(self.__class__.__name__))

      rng_state = torch.get_rng_state()
      torch.manual_seed(index + self.random_offset)

      while True:
        img = torch.zeros(self.dim, self.dim)
        dx = torch.randint(-10,10,(1,),dtype=torch.float)
        dy = torch.randint(-10,10,(1,),dtype=torch.float)
        c = torch.randint(-20,20,(1,), dtype=torch.float)

        params = torch.cat((dy/dx, c))
        xy = torch.randint(0,img.shape[1], (20, 2), dtype=torch.float)
        xy[:,1] = xy[:,0] * params[0] + params[1]

        xy.round_()
        xy = xy[ xy[:,1] > 0 ]
        xy = xy[ xy[:,1] < self.dim ]
        xy = xy[ xy[:,0] < self.dim ]

        for i in range(xy.shape[0]):
          x, y = xy[i][0], self.dim - xy[i][1]
          img[int(y), int(x)]=1
        if img.sum() > 2:
          break

      torch.set_rng_state(rng_state)
      return img.unsqueeze(0), params

  def __len__(self):
      return self.size

class Network1(nn.Module):
  def __init__(self):
    super(Network1, self).__init__()
    self.conv = nn.Conv2d(1, 48, (3,3), 1, padding=1)
    self.relu = nn.ReLU()
    self.fc1 = nn.Linear(76800, 128)
    self.fc2 = nn.Linear(128, 2)

  def forward(self, x):
    x = self.relu(self.conv(x))
    x = x.view(x.shape[0],-1)
    x = self.relu(self.fc1(x))
    outputs = self.fc2(x)

    return outputs

class Network2(nn.Module):
  def __init__(self):
    super(Network2, self).__init__()
    self.conv1 = nn.Conv2d(1, 48, (3,3), 1, padding=1)
    self.conv2 = nn.Conv2d(48, 48, (3,3), 1, padding=1)
    self.maxpool = nn.AdaptiveMaxPool2d(1)
    self.relu = nn.ReLU()
    self.fc1 = nn.Linear(48, 128)
    self.fc2 = nn.Linear(128, 2)

  def forward(self, x):
    x = self.relu(self.conv1(x))
    x = self.relu(self.conv2(x))
    x = self.maxpool(x)
    x = x.view(x.size(0), -1)
    x = self.relu(self.fc1(x))
    outputs = self.fc2(x)

    return outputs

class Network3(nn.Module):
  def __init__(self):
    super(Network3, self).__init__()
    self.conv1 = nn.Conv2d(3, 48, (3,3), 1, padding=1)
    self.conv2 = nn.Conv2d(48, 48, (3,3), 1, padding=1)
    self.maxpool = nn.AdaptiveMaxPool2d(1)
    self.relu = nn.ReLU()
    self.fc1 = nn.Linear(48, 128)
    self.fc2 = nn.Linear(128, 2)

  def forward(self, x):
    idxx = torch.repeat_interleave(torch.arange(-20, 20, dtype=torch.float).unsqueeze(0)/40.0,
                                                repeats=40, 
                                                dim=0).to(x.device)
    idxy = idxx.clone().t()
    idx = torch.stack([idxx, idxy]).unsqueeze(0)
    idx = torch.repeat_interleave(idx, repeats=x.shape[0], dim=0) 
    x = torch.cat([x, idx], dim=1)

    x = self.relu(self.conv1(x))
    x = self.relu(self.conv2(x))
    x = self.maxpool(x)
    x = x.view(x.size(0), -1)
    x = self.relu(self.fc1(x))
    outputs = self.fc2(x)

    return outputs

class Model():
  def __init__(self):
    self.classifier = Network3()
    self.loss_function = nn.MSELoss()
    self.optimizer = optim.Adam(self.classifier.parameters())
    self.loss = torch.zeros(1, requires_grad=False)
    self.device = 'cuda: 0' if torch.cuda.is_available() else 'cpu'
    self.classifier = self.classifier.to(self.device)
  
  def set_input(self, input, label):
    self.input, self.label = input.to(device), label.to(device)

  def optimize(self):
    pred = self.classifier(self.input)
    self.optimizer.zero_grad()
    self.loss = self.loss_function(pred, self.label)
    self.loss.backward()
    self.optimizer.step()
  
  def test(self, path = None):
    self.classifier = self.classifier.eval()
    if path:
      self.classifier.load_state_dict(torch.load(path))
    pred = self.classifier(self.input)
    self.loss = self.loss_function(pred, self.label)

    return pred
  
  def save_net(self):
    save_path = os.path.join('/content/drive/My Drive/Colab Notebooks/lab', 'model'+'.pkl')
    torch.save(self.classifier.state_dict(), save_path)

  # def get_current_error(self):

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
  print('epoch: %d;'% (i+1), 'loss: %.2f'%(loss/len(trainloader)))
model.save_net()
print('saving network...')

val
loss = 0
for i, data in enumerate(valloader):
  input, label = data
  model.set_input(input, label)
  model.test('/content/drive/My Drive/Colab Notebooks/lab/model.pkl')
  loss += model.loss.item()
print('Val_MSE: %.3f' % (loss/len(valloader)))

# test
loss = 0
for i, data in enumerate(testloader):
  input, label = data
  model.set_input(input, label)
  model.test('/content/drive/My Drive/Colab Notebooks/lab/model.pkl')
  loss += model.loss.item()
print('Test_MSE: %.3f' % (loss/len(testloader)))

def show(label, pred, input):
    X = np.arange(40)
    L = label[0][0] * X + label[0][1]
    P = pred[0][0] * X + pred[0][1]
    plt.figure(figsize = (5,5))
    plt.plot(X, 40-L, color="yellow", linewidth=3.0)
    plt.plot(X, 40-P, color="red", linewidth=3.0)
    
    plt.imshow(input[0][0])
    plt.show()

# test
loss = 0
for i, data in enumerate(testloader):
  input, label = data
  model.set_input(input, label)
  pred = model.test('/content/drive/My Drive/Colab Notebooks/lab/model_3.pkl')
  show(label.cpu().detach().numpy(), pred.cpu().detach().numpy(), input.cpu().detach().numpy())
