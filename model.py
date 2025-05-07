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


class UNetDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, kernel=kernel)

    def forward(self, x, skip_input):
        x = self.upconv(x)
        padding = (skip_input.shape[2] - x.shape[2]) // 2
        extra_padding = (skip_input.shape[2] - x.shape[2]) % 2
        x = F.pad(x, (padding, padding + extra_padding,
                      padding, padding + extra_padding))
        x = torch.cat((x, skip_input), dim=1)
        return self.conv(x)


class UNet(nn.Module):

    def __init__(self, num_layers, init_filters, num_classes=4):
        super().__init__()

        num_channels = 3
        num_filters = init_filters
        encoders = []
        decoders = []

        self.first_conv = DoubleConv(num_channels, num_filters)

        for _ in range(num_layers):
            num_channels = num_filters
            num_filters *= 2
            print(f'Adding encoder layer with {num_channels} channels and {num_filters} filters')
            encoders.append(DoubleConv(num_channels, num_filters, kernel=3))

        self.encoder_layers = nn.ModuleList(encoders)

        num_channels *= 2
        num_filters = num_filters // 2

        for i in range(num_layers):
            print(f'Adding decoder layer with {num_channels} channels and {num_filters} filters')
            decoders.append(UNetDecoder(int(num_channels), int(num_filters), kernel=3))
            num_channels = num_filters
            num_filters //= 2

        self.decoder_layers = nn.ModuleList(decoders)

        self.final_conv = nn.Conv2d(num_channels, num_classes, kernel_size=1)

        # Ensure that each pixel predicts just one of the possible classes
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        original_size = x.size()[2:]
        x = self.first_conv(x)

        inputs = []
        for enc in self.encoder_layers:
            inputs.append(x)
            x = F.max_pool2d(x, kernel_size=2)
            x = enc(x)

        for dec, inp in zip(self.decoder_layers, inputs[::-1]):
            x = dec(x, inp)

        x = self.final_conv(x)
        # Padding to ensure that the output size matches the input size
        padding = (original_size[0] - x.shape[2]) // 2
        extra_padding = (original_size[0] - x.shape[2]) % 2
        x = F.pad(x, (padding, padding + extra_padding,
                      padding, padding + extra_padding))
        x = self.softmax(x)
        return x
