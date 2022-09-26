import torch.nn as nn
from torch import flatten
import torch.nn.functional as F

class SimpleDiscriminator(nn.Module):
    # For a batch of (128,28,28) --> (128,) classifications, avg. batch time = 0.01396514892578125 s
    def __init__(self, in_features=28*28, dropout=.2, return_logits=True):
        super(SimpleDiscriminator, self).__init__()

        self.dropout = dropout
        self.in_features = in_features
        self.return_logits=return_logits

        self.lin1 = nn.Linear(in_features=28*28, out_features=1024)
        self.lin2 = nn.Linear(in_features=self.lin1.out_features, out_features=self.lin1.out_features//2)
        self.lin3 = nn.Linear(in_features=self.lin2.out_features, out_features=self.lin2.out_features//2)
        self.lin4 = nn.Linear(in_features=self.lin3.out_features, out_features=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.lin1(x)
        x = self.relu(x)
        x = F.dropout(x, self.dropout)
        x = self.lin2(x)
        x = self.relu(x)
        x = F.dropout(x, self.dropout)
        x = self.lin3(x)
        x = self.relu(x)
        x = F.dropout(x, self.dropout)
        x = self.lin4(x)
        x = x.squeeze_()
        if self.return_logits: return x
        else: return self.sigmoid(x)


if __name__ == "__main__":
    import torch
    from torch import rand
    import time

    batch_size = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    disc = SimpleDiscriminator(in_features=28*28).to(device)

    times = []
    for i in range(25):
        x = rand((batch_size, 28, 28), device=device)
        start = time.time()
        out = disc(x)
        t = time.time() - start
        print(t, out.shape)
        times.append(t)
    
    print("Average Time: ", sum(times)/len(times))


        
