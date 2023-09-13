import pdb
import torch
from torch import nn
from torch.nn import functional as F

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel=3):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        return x

class UNet(nn.Module):

    def __init__(self, num_layers, init_filters):
        super().__init__()

        num_channels = 3
        num_filters = init_filters // 2
        encoders = []
        decoders = []

        self.first_conv = DoubleConv(num_channels, num_filters)

        for _ in range(num_layers):
            num_channels = num_filters
            num_filters *= 2
            print(f'Adding encoder layer with {num_channels} channels and {num_filters} filters')
            encoders.append(DoubleConv(num_channels, num_filters, kernel=3))

        self.encoder_layers = nn.ModuleList(encoders)

        for _ in range(num_layers):
            padding = 0 if num_filters > init_filters else 1
            print(f'Adding decoder layer with {num_filters} channels and {num_channels} filters')
            decoders.append(DoubleConv(int(num_filters), int(num_channels), kernel=3))
            num_filters = num_channels
            num_channels //= 2

        self.decoder_layers = nn.ModuleList(decoders)

        # Ensure that each pixel predicts just one of the possible classes
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        x = self.first_conv(x)

        inputs = []
        for enc in self.encoder_layers:
            inputs.append(x)
            x = F.max_pool2d(x, kernel_size=2)
            x = enc(x)

        for dec, inp in zip(self.decoder_layers, inputs[::-1]):
            x = F.interpolate(x, scale_factor=2,
                                          mode='bilinear', align_corners=True)
            x2 = dec(x)
            padding = (inp.shape[2] - x2.shape[2]) // 2
            x2 = F.pad(x2, [padding] * 4)
            pdb.set_trace()
            x = torch.concat([x2, inp])

        x = self.softmax(x)

        return x
