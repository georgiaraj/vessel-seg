import torch
from torch import nn

class UNet(nn.Module):

    def __init__(self, num_layers, init_filters):
        super().__init__()

        num_channels = 3
        num_filters = init_filters
        self.encoder_layers = []
        self.decoder_layers = []
        
        for _ in range(num_layers):
            self.encoder_layers.append(nn.Conv2d(num_channels, num_filters, 3, stride=2))
            num_channels = num_filters
            num_filters *= 2

        for _ in range(num_layers):
            self.decoder_layers.append(nn.ConvTranspose2d(num_filters, num_channels, 3, stride=2))
            num_filters = num_channels
            num_channels /= 2    
        
    def forward(self, images):

        for enc in self.encoder_layers:
            x = enc(x)
            x = torch.relu(x)

        for dec in self.decoder_layers:
            x = dec(x)
            x = torch.relu(x)

        return x

    
    
