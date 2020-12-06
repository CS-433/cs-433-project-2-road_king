import os
from os.path import splitext
from os import listdir
import numpy as np

import torch
import torchvision
import PIL


class BaseDataset(torch.utils.data.Dataset):
    """
    Base dataset for loading images and masks
    optional preprocessing
    """
    def __init__(self, imgs_dir, masks_dir, image_set="all", split_ratio=0.85, preprocess=None, colorJitter=True,
                 verbose=True):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.preprocess = preprocess
        self.colorJitter = colorJitter

        ids = [splitext(file)[0] for file in listdir(imgs_dir)
               if not file.startswith('.')]
        ids.sort()
        n = len(ids)
        assert image_set in ["all", "train", "val"]
        if image_set == "train":
            ids = ids[:int(n * split_ratio)]
        elif image_set == "val":
            ids = ids[int(n * split_ratio):]

        self.ids = ids
        if verbose:
            print(f'Creating {image_set} dataset with {len(self.ids)} original examples')
            print(f'image directory: {imgs_dir}')
            print(f'mask directory； {masks_dir}')

    def __getitem__(self, i):
        idx = self.ids[i]
        assert "satImage" in idx
        image_path = os.path.join(self.imgs_dir, f'{idx}.png')
        mask_path = os.path.join(self.masks_dir, f'{idx}.png')
        with open(image_path, 'rb') as f1:
            img = PIL.Image.open(f1).convert('RGB')
        with open(mask_path, 'rb') as f2:
            mask = PIL.Image.open(f2).convert('L')
        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'
        img = torchvision.transforms.ToTensor()(img)
        mask = torchvision.transforms.ToTensor()(mask)
        mask = (mask > 0.5).float()
        if self.colorJitter:
            c_jitter = torchvision.transforms.Compose([
                torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                torchvision.transforms.RandomGrayscale(p=0.05),
            ])
            img = c_jitter(img)

        if self.preprocess is not None:
            T = torch.cat([img, mask], dim=0)
            T = self.preprocess(T)
            img = T[:3, :, :]
            mask = T[3, :, :]

        mask = (mask > 0.5).float()
        mask = torch.unsqueeze(mask, dim=0)
        return {
            'image': img,
            'mask': mask,
            "ID": idx

        }

    def __len__(self):

        return len(self.ids)
