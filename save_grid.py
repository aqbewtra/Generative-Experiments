import torch
from torchvision.utils import make_grid
from matplotlib import pyplot as plt

def save_grid(bchw, path):
    # bchw = tensor with shape = (batch_size, channels, height, width)
    grid_x = make_grid(bchw, nrow=int(bchw.shape[0]**(1/2)))
    plt.imshow(grid_x.permute(1, 2, 0))
    plt.savefig(path)
    return

if __name__ == "__main__":
    size = (16,3,28,28)
    x = torch.rand(*size)

    save_grid(x, 'samples/sample_grid')

    grid_x = make_grid(x, nrow=4)
    print(grid_x.shape)
    plt.imshow(grid_x.permute(1, 2, 0))
    plt.show()



