import pdb
import torch
from torch import nn

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3):
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)


class UNet(nn.Module):

    def __init__(self, num_layers, init_filters):
        super().__init__()

        num_channels = 3
        num_filters = init_filters
        encoders = []
        decoders = []

        for _ in range(num_layers):
            print(f'Adding encoder layer with {num_channels} channels and {num_filters} filters')
            encoders.append(DoubleConv(num_channels, num_filters, 3))
            num_channels = num_filters
            num_filters *= 2

        self.encoder_layers = nn.ModuleList(encoders)

        for _ in range(num_layers):
            num_filters = num_channels
            num_channels /= 2
            padding = 0 if num_filters > init_filters else 1
            print(f'Adding decoder layer with {num_filters} channels and {num_channels} filters')
            decoders.append(DoubleConv(int(num_filters), int(num_channels), 3)

        self.decoder_layers = nn.ModuleList(decoders)

        # Ensure that each pixel predicts just one of the possible classes
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        inputs = []
        for enc in self.encoder_layers:
            x = torch.maxpool2(x)
            x = enc(x)
            inputs.append(x)

        for dec, inp in zip(self.decoder_layers, inputs[::-1]):
            x = torch.nn.functional.interpolate(x, scale_factor=2,
                                                mode='bilinear', align_corners=True)
            x = torch.concat)[dec(x), inp]
            x = torch.relu(x)

        x = self.softmax(x)

        return x
