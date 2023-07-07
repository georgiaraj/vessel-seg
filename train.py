import argparse

from model import UNet
from datasets import data


def get_args():
    parser = argparse.ArgumentParser('Vessel Segmentation')
    parser.add_argument('data_dir', type=str, help='Directory for the data')
    parser.add_argument('--train-videos', default=None, nargs='+', type=str,
                        help='Videos to use for train. If empty all are used.')
    parser.add_argument('--test-videos', default=None, nargs='+', type=str,
                        help='Videos to use for test. If empty all are used.')
    return parser.parse_args()
    
    
if __name__ == '__main__':

    args = get_args()
    
    model = UNet(3, 8)

    print(model)

    dataset = data['vessel_data'](args.data_dir)

    print(dataset)
