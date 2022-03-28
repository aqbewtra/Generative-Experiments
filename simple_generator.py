from torch import nn

class SimpleGenerator(nn.Module):
    def __init__(self, input_shape=(10,10,10), output_shape=(1,28,28), batch_size=None):
        super(SimpleGenerator, self).__init__()
        
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.batch_size = batch_size

        self.lin1 = nn.Linear(in_features=self.input_shape[0], out_features=self.output_shape[0])
        self.lin2 = nn.Linear(in_features=self.input_shape[1], out_features=self.output_shape[1])
        self.lin3 = nn.Linear(in_features=self.input_shape[2], out_features=self.output_shape[2])

        self.conv1 = nn.Conv2d(in_channels=output_shape[0], out_channels=output_shape[0], kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=output_shape[0], out_channels=output_shape[0], kernel_size=3, padding=1)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, z):
        z = self.lin1(z)
        z = self.relu(z)
        z = z.reshape((self.batch_size, self.output_shape[0], self.input_shape[2], self.input_shape[1]))
        z = self.lin2(z)
        z = self.relu(z)
        z = z.reshape((self.batch_size, self.output_shape[0], self.output_shape[1], self.input_shape[2]))
        z = self.lin3(z)
        if len(z.shape) < 4: z = z.unsqueeze_(dim=0)
        z = self.conv1(z)
        z = self.relu(z)
        z = self.conv2(z)

        return z

if __name__ == "__main__":
    img_shape = (1, 28, 28)
    z_shape = (10, 10, 10)
    batch_size = 16

    gen = SimpleGenerator(input_shape=z_shape, output_shape=img_shape, batch_size=batch_size)
    import torch

    z = torch.rand((batch_size, *z_shape))
    img = gen(z)
    print(img.shape)