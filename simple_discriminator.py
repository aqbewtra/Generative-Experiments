import torch.nn as nn
from torch import flatten

class SimpleDiscriminator(nn.Module):
    def __init__(self, in_features=28*28):
        super(SimpleDiscriminator, self).__init__()
        # self.conv1 = nn.Conv2d(in_channels=img_shape[0], out_channels=5, kernel_size=5, padding=2)
        # self.conv2 = nn.Conv2d(in_channels=5, out_channels=10, kernel_size=5, padding=2)
        # self.conv3 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5, padding=2)

        self.in_features = in_features

        self.lin1 = nn.Linear(in_features=self.in_features, out_features=self.in_features*10)
        self.lin2 = nn.Linear(in_features=self.in_features*10, out_features=self.in_features*10*2)
        self.lin3 = nn.Linear(in_features=self.in_features*10*2, out_features=1)

        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=5, stride=2, padding=2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        x = self.lin3(x)
        x = x.squeeze_()
        # return self.sigmoid(x)
        return x

if __name__ == "__main__":
    import torch
    from torch import rand
    import time

    batch_size = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    disc = SimpleDiscriminator(in_features=28*28).to(device)

    times = []
    for i in range(1000):
        x = rand((batch_size, 28, 28), device=device)
        start = time.time()
        out = disc(x)
        t = time.time() - start
        print(t, out.shape)
        times.append(t)
    
    print("Average Time: ", sum(times)/len(times))


        
