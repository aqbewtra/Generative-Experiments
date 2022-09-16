from torch import nn

class SimpleGenerator(nn.Module):
    def __init__(self, input_dim, out_features, img_shape, batch_size=None):
        super(SimpleGenerator, self).__init__()
        
        self.input_dim = input_dim
        self.out_features = out_features
        self.batch_size = batch_size
        self.img_shape = img_shape

        self.lin1 = nn.Linear(in_features=self.input_dim, out_features=256)
        self.lin2 = nn.Linear(in_features=self.lin1.out_features, out_features=self.lin1.out_features*2)
        self.lin3 = nn.Linear(in_features=self.lin2.out_features, out_features=self.lin2.out_features*2)
        self.lin4 = nn.Linear(in_features=self.lin3.out_features, out_features=out_features)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        z = self.lin1(z)
        z = self.relu(z)
        z = self.lin2(z)
        z = self.relu(z)
        z = self.lin3(z)
        z = self.relu(z)
        z = self.lin4(z)
        z = self.sigmoid(z)
        
        z = z.view(self.batch_size, *self.img_shape)

        return z

if __name__ == "__main__":
    import torch
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1000
    img_shape = (1, 28, 28)
    z_dim = 100
    g = SimpleGenerator(input_dim=100, out_features=28*28, img_shape=img_shape, batch_size=batch_size).to(device)

    times = []
    for i in range(100):
        start = time.time()
        z = torch.rand((batch_size, z_dim), device=g.lin1.weight.device)
        fake_batch = g(z)
        t = time.time() - start
        print(time.time() - start, fake_batch.shape)
        times.append(t)
    print("Average Time: ", sum(times)/len(times))


        
