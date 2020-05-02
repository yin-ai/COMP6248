import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.models import resnet50
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.optim as optim

def network():

  net = resnet50(pretrained = True)
  net.fc = nn.Linear(2048, len(train_dataset.classes))
  freeze_num = 3
  net_iter = net.named_children().__iter__()

  # freeze_layer
  while freeze_num:
      _, child = net_iter.__next__()
      for _, param in child.named_parameters():
          param.requires_grad = False
      freeze_num -= 1  

  return net

class Model():
    def __init__(self):
      self.device = 'cuda: 0' if torch.cuda.is_available() else 'cpu'
      self.classifier = network().to(self.device)
      self.classifier.load_state_dict(torch.load('/content/drive/My Drive/data/checkpoint/model_1.pkl'))
      self.loss_function = nn.CrossEntropyLoss()
      self.optim = optim.Adam(self.classifier.parameters(),
                              lr=0.001,
                              betas=(0.9, 0.999),
                              weight_decay=0)
      
    def set_input(self, input, label):
      self.input, self.label = input.to(self.device), label.to(self.device)
    
    def optimizer(self):
      self.output = self.classifier(self.input)
      self.optim.zero_grad()
      self.loss = self.loss_function(self.output, self.label)
      self.loss.backward()
      self.optim.step()
    
    def test(self, pretrained = False):
      self.classifier = self.classifier.eval()
      if pretrained:
        path = os.path.join('/content/drive/My Drive/data/checkpoint', 'model'+'.pkl')
        self.classifier.load_state_dict(torch.load(path))
      self.output = self.classifier(self.input)
      self.loss = self.loss_function(self.output, self.label)

    def get_current_error(self):
      _, pred = torch.max(self.output.detach(), dim=1)
      correct_mask = torch.eq(pred, self.label).type(torch.FloatTensor)
      self.accuracy = torch.mean(correct_mask)
      
      return self.accuracy.item(), self.loss.item()
  
    def print(self, epoch, i, iter_):
      print('Epoch: %d(%d/%d) current_accuracy: %.3f current_Train_loss: %.3f' % (epoch, i, iter_, self.accuracy, self.loss))

    def save_net(self):
      save_path = os.path.join('/content/drive/My Drive/data/checkpoint', 'model'+'.pkl')
      torch.save(self.classifier.state_dict(), save_path)
