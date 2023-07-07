from model import UNet
from datasets import data

def train_unet():
    model = UNet(3, 8)

    print(model)

