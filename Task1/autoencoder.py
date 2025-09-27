import torch
import torch.nn as nn

def create_conv2d_blocks(in_ch, base, layer_count, k=3, s=2, p=1):
    layer = nn.Sequential()
    for i in range(layer_count):
        in_c  = in_ch if i == 0 else base * (2 ** (i - 1))
        out_c = base * (2 ** i)
        layer.add_module(f'conv2d_{i+1}', nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False))
        layer.add_module(f'batchnorm2d_{i+1}', nn.BatchNorm2d(out_c))
        layer.add_module(f'relu_{i+1}', nn.ReLU(inplace=True))
    return layer

def create_deconv2d_block(start_ch, base, layer_count, k=4, s=2, p=1):
    layers = nn.Sequential()
    for i in range(layer_count):
        in_c  = base * (2 ** (layer_count - 1 - i))
        out_c = base * (2 ** (layer_count - 2 - i)) if i < layer_count - 1 else base
        if i == 0 and in_c != start_ch:
            in_c = start_ch
        layers.add_module(f'deconv2d_{i+1}', nn.ConvTranspose2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False))
        layers.add_module(f'batchnorm2d_{i+1}', nn.BatchNorm2d(out_c))
        layers.add_module(f'relu_{i+1}', nn.ReLU(inplace=True))
    return layers

class Encoder2D(nn.Module):
    def __init__(self, in_channels: int, input_shape: tuple, latent_dim: int, base: int = 32, layer_count: int = 4,
                 k=3, s=2, p=1):
        super().__init__()
        C, H, W = input_shape
        assert C == in_channels, "input_shape[0] and in_channels should be same."

        self.enc = create_conv2d_blocks(in_channels, base, layer_count, k=k, s=s, p=p)

        hL, wL = H, W
        for _ in range(layer_count):
            hL = (hL + 2*p - k) // s + 1
            wL = (wL + 2*p - k) // s + 1

        last_ch = base * (2 ** (layer_count - 1))
        self._feat_shape = (last_ch, hL, wL)
        self.fc = nn.Linear(last_ch * hL * wL, latent_dim)

    def forward(self, x):
        x = self.enc(x)                  # (N, last_ch, hL, wL)
        x = x.flatten(1)                 # (N, last_ch*hL*wL)
        z = self.fc(x)                   # (N, latent_dim)
        return z


class Decoder2D(nn.Module):
    def __init__(self, out_channels: int, feat_shape: tuple, latent_dim: int, base: int = 32, layer_count: int = 4):
        super().__init__()
        cL, hL, wL = feat_shape
        self.fc = nn.Linear(latent_dim, cL * hL * wL)

        self.dec = create_deconv2d_block(start_ch=cL, base=base, layer_count=layer_count)

        self.to_out = nn.Sequential(
            nn.Conv2d(base, out_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self._feat_shape = feat_shape

    def forward(self, z):
        x = self.fc(z)
        cL, hL, wL = self._feat_shape
        x = x.view(-1, cL, hL, wL)      # artık (N, cL, hL, wL)
        x = self.dec(x)
        x = self.to_out(x)
        return x

class AutoEncoder2D(nn.Module):
    def __init__(self, input_shape: tuple, latent_dim: int, base: int = 32, layer_count: int = 4):
        super().__init__()
        in_ch = input_shape[0]
        self.encoder = Encoder2D(in_channels=in_ch, input_shape=input_shape,
                                 latent_dim=latent_dim, base=base, layer_count=layer_count)
        self.decoder = Decoder2D(out_channels=in_ch, feat_shape=self.encoder._feat_shape,
                                 latent_dim=latent_dim, base=base, layer_count=layer_count)

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out
