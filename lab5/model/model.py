import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn
from torch import optim
import numpy as np
import os
from network.network1 import Network1
from network.network2 import Network2
from network.network3 import Network3

class Model():
  def __init__(self):
    self.classifier = Network3()
    self.loss_function = nn.MSELoss()
    self.optimizer = optim.Adam(self.classifier.parameters())
    self.loss = torch.zeros(1, requires_grad=False)
    self.device = 'cuda: 0' if torch.cuda.is_available() else 'cpu'
    self.classifier = self.classifier.to(self.device)
  
  def set_input(self, input, label):
    self.input, self.label = input.to(self.device), label.to(self.device)

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
