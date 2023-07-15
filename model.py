import pdb
import torch
from torch import nn

class UNet(nn.Module):

    def __init__(self, num_layers, init_filters):
        super().__init__()

        num_channels = 3
        num_filters = init_filters
        encoders = []
        decoders = []

        for _ in range(num_layers):
            print(f'Adding encoder layer with {num_channels} channels and {num_filters} filters')
            encoders.append(nn.Conv2d(num_channels, num_filters, 3, stride=2))
            num_channels = num_filters
            num_filters *= 2

        self.encoder_layers = nn.ModuleList(encoders)

        for _ in range(num_layers):
            num_filters = num_channels
            num_channels /= 2
            print(f'Adding decoder layer with {num_filters} channels and {num_channels} filters')
            decoders.append(nn.ConvTranspose2d(int(num_filters), int(num_channels), 3,
                                                          stride=2))

        self.decoder_layers = nn.ModuleList(decoders)

        # Ensure that each pixel predicts just one of the possible classes
        self.softmax = nn.Softmax(dim=3)

    def forward(self, x):

        for enc in self.encoder_layers:
            x = enc(x)
            x = torch.relu(x)

        for dec in self.decoder_layers:
            x = dec(x)
            x = torch.relu(x)

        x = self.softmax(x)

        return x
