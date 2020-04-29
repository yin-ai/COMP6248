from torch import nn

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
