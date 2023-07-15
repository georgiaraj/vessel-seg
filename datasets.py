import pdb
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.v2 import Resize, ConvertImageDtype, ToDtype

class VesselSegDataset(Dataset):

    def __init__(self, root_dir, videos=None, imsize=480, transforms=None):
        self.root_dir = Path(root_dir)
        if videos is None:
            # Use all videos that are found
            videos = os.listdir(self.root_dir)
        self.videos = [self.root_dir / vid for vid in videos
                       if os.path.isdir(self.root_dir / vid) and os.path.exists(self.root_dir / vid)]
        if len(self.videos) == 0:
            raise RuntimeError('No videos available to create dataset')

        self.image_list = []
        for vid in self.videos:
            self.image_list.extend(self._get_image_list(vid))

        if len(self.image_list) == 0:
            raise RuntimeError(f"No images found in videos {' '.join(self.videos)}")

        self.label_list = [im.replace('images', 'labels') for im in self.image_list]
        print(f'Number of images added to dataset: {self.__len__()}')

        self.image_transforms = [ConvertImageDtype(torch.float32), Resize(imsize)]
        self.label_transforms = [ToDtype(torch.long), Resize(imsize)]

        if transforms:
            self.image_transforms += transforms
            self.label_transforms += transforms

        self.image_transforms = torch.nn.Sequential(*self.image_transforms)
        self.label_transforms = torch.nn.Sequential(*self.label_transforms)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = self.image_transforms(read_image(self.image_list[idx], mode=ImageReadMode.RGB))
        labels = self.label_transforms(read_image(self.label_list[idx]))
        return image, labels.squeeze()

    @staticmethod
    def _get_image_list(video):
        image_path = video / 'images'
        images = [str(image_path / im) for im in os.listdir(image_path) if im.endswith('.png')]
        image_list = [im for im in images if os.path.exists(im)]

        if len(image_list) == 0:
            raise Warning(f'No images found in video directory {video}')

        print(f'Adding {len(image_list)} images for video {video}')

        return image_list


data = {'vessel_data': VesselSegDataset}
