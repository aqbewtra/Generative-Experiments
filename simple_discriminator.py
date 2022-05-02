import torch.nn as nn
from torch import flatten

class SimpleDiscriminator(nn.Module):
    def __init__(self, img_shape=(1, 28, 28)):
        super(SimpleDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=img_shape[0], out_channels=5, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=10, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5, padding=2)

        features = 320

        self.lin1 = nn.Linear(in_features=features, out_features=int(features/2))
        self.lin2 = nn.Linear(in_features=int(features/2), out_features=int(features/4))
        self.lin3 = nn.Linear(in_features=int(features/4), out_features=1)

        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=5, stride=2, padding=2)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.max_pool(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.max_pool(x)
        x = self.relu(x)
        x = flatten(x, start_dim=1)
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        x = self.lin3(x)
        return x.squeeze_()

if __name__ == "__main__":
    from torch import rand
    x = rand((8, 1, 28, 28))

    disc = SimpleDiscriminator()
    out = disc(x)
    print(out.shape)

        
