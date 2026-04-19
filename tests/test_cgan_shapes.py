import torch
from lab2.vanilla_gan.models import Generator, Discriminator

def test_cgan_shapes():
    z_dim = 16
    h_dim = 32
    x_dim = 28 * 28
    batch_size = 4
    num_classes = 10

    G = Generator(z_dim, h_dim, x_dim)
    D = Discriminator(x_dim, h_dim)

    z = torch.randn(batch_size, z_dim)
    labels = torch.randint(0, num_classes, (batch_size,))

    g_out = G(z, labels)
    assert g_out.shape == (batch_size, x_dim)

    d_out = D(g_out, labels)
    assert d_out.shape == (batch_size, 1)

