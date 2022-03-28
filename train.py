from torch.utils.data import DataLoader, ConcatDataset
from torch import rand, cat, ones, zeros, argmax
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

from simple_discriminator import SimpleDiscriminator
from simple_generator import SimpleGenerator


img_shape = (1, 28, 28)
z_shape = (10, 10, 10)

batch_size = 16

gen = SimpleGenerator(input_shape=z_shape, output_shape=img_shape, batch_size=batch_size)
disc = SimpleDiscriminator(img_shape=img_shape)

train_dataset = MNIST('/Users/avibewtra/anaconda3/envs/data/MNIST', train=True, download=False, transform=transforms.PILToTensor())
test_dataset = MNIST('/Users/avibewtra/anaconda3/envs/data/MNIST', train=False, download=False, transform=transforms.PILToTensor())

loader = DataLoader(ConcatDataset([train_dataset, test_dataset]), batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)

for batch_id, (imgs, MNIST_label) in enumerate(loader):
    # print(batch_id, imgs.shape, label)
    z = rand((batch_size, *z_shape))
    imgs_gen = gen(z)

    x = cat((imgs, imgs_gen), dim=0)
    y = disc(x)
    disc_label = cat((ones((batch_size,)), zeros((batch_size,))), dim=0)
    print(disc_label.shape)
    print(argmax(y, dim=1).shape)
    break 
