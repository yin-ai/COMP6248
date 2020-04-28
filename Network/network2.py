from torch import nn

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
