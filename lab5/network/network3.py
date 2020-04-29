from torch import nn
import torch

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
