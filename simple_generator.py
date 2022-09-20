from torch import nn

class SimpleGenerator(nn.Module):
    def __init__(self, input_dim, features=512, out_features=28*28, img_shape=(1,28,28)):
        super(SimpleGenerator, self).__init__()
        
        self.input_dim = input_dim
        self.features=features
        self.out_features = out_features
        self.img_shape = img_shape

        self.lin1 = nn.Linear(in_features=self.input_dim, out_features=self.features)
        self.lin2 = nn.Linear(in_features=self.lin1.out_features, out_features=self.lin1.out_features*2)
        self.lin3 = nn.Linear(in_features=self.lin2.out_features, out_features=self.lin2.out_features*2)
        self.lin4 = nn.Linear(in_features=self.lin3.out_features, out_features=out_features)

        self.ln1 = nn.LayerNorm(self.lin1.out_features)
        self.ln2 = nn.LayerNorm(self.lin2.out_features)
        self.ln3 = nn.LayerNorm(self.lin3.out_features)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, z):
        z = self.lin1(z)
        z = self.ln1(z)
        z = self.relu(z)

        z = self.lin2(z)
        z = self.ln2(z)
        z = self.relu(z)

        z = self.lin3(z)
        z = self.ln3(z)
        z = self.relu(z)
        
        z = self.lin4(z)
        z = self.sigmoid(z)
        z = z.view(-1, *self.img_shape)
        return z

if __name__ == "__main__":
    import torch
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1000
    img_shape = (1, 28, 28)
    z_dim = 100
    g = SimpleGenerator(input_dim=100, features=512, out_features=28*28, img_shape=img_shape).to(device)

    times = []
    for i in range(1):
        start = time.time()
        z = torch.rand((batch_size, z_dim), device=g.lin1.weight.device)
        fake_batch = g(z)
        t = time.time() - start
        print(time.time() - start, fake_batch.shape)
        times.append(t)
    print("Average Time: ", sum(times)/len(times))


        
