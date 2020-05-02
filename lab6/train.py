from model import Model
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.models import resnet50
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.optim as optim

batch_size=128
image_size=(224, 224)
data_path = '/content/drive/My Drive/data/'
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor()  # convert to tensor
])

train_dataset = ImageFolder(os.path.join(data_path, "train"), transform)
print('len of trainset: %d' % (len(train_dataset)))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


val_dataset = ImageFolder(os.path.join(data_path, "valid"), transform)
print('len of valset: %d' % (len(val_dataset)))
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

test_dataset = ImageFolder(os.path.join(data_path, "test"), transform)
print('len of testset: %d' % (len(test_dataset)))
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

model = Model()
#train
for epoch in range(1, 21):
    accuracy, loss = 0, 0
    for i, data in enumerate(train_loader):
      input, label = data
      model.set_input(input, label)
      model.optimizer()
      temp_accuracy, temp_loss = model.get_current_error()
      accuracy, loss = accuracy+temp_accuracy, loss+temp_loss
      if i%5 == 0:
        model.print(epoch, i, len(train_loader))
    print('Epoch: %d Average_Accuracy: %.3f Average_Train_loss: %.3f' % (epoch, 
                                                        (accuracy/len(train_loader)), 
                                                        (loss/len(train_loader))))  
    model.save_net()
    print('Saving network...') 

#val
accuracy, loss = 0, 0
for _, data in enumerate(val_loader):
  input, label = data
  model.set_input(input, label)
  model.test(pretrained = True)
  temp_accuracy, temp_loss = model.get_current_error()
  accuracy, loss = accuracy+temp_accuracy, loss+temp_loss
print('val_Accuracy: %.3f val_loss: %.3f' % ( accuracy/len(val_loader), 
                                                      loss/len(val_loader)))

#test
accuracy, loss = 0, 0
for i, data in enumerate(test_loader):
  input, label = data
  model.set_input(input, label)
  model.test(pretrained = True)
  temp_accuracy, temp_loss = model.get_current_error()
  accuracy, loss = accuracy+temp_accuracy, loss+temp_loss

print('test_Accuracy: %.3f Test_loss: %.3f' % (accuracy/len(test_loader), 
                                          loss/len(test_loader)))
