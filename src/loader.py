import os
from os.path import splitext
from os import listdir
import torch
import torchvision
import PIL
from rotate import *


class BaseDataset(torch.utils.data.Dataset):
    """
    Base dataset for loading training/validation images and masks
    optional preprocessing
    """

    def __init__(self, imgs_dir, masks_dir, image_set="all", split_ratio=0.98, preprocess=None, color_jitter=True,
                 rotation=True, verbose=True):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.preprocess = preprocess
        self.color_jitter = color_jitter
        self.rotation = rotation
        ids = [splitext(file)[0] for file in listdir(imgs_dir)
               if not file.startswith('.')]
        ids.sort()
        n = len(ids)
        assert image_set in ["all", "train", "val"], "image set should be train,val or all"
        assert split_ratio < 1, "split ratio should be within (0,1)"
        if image_set == "train":
            ids = ids[:int(n * split_ratio)]
        elif image_set == "val":
            ids = ids[int(n * split_ratio):]
        self.ids = ids
        if verbose:
            print(f'Creating {image_set} dataset with {len(self.ids)} original examples')
            print(f'image directory: {imgs_dir}')
            print(f'mask directory: {masks_dir}')

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
        # optional color jitter on images
        if self.color_jitter:
            c_jitter = torchvision.transforms.Compose([
                torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                torchvision.transforms.RandomGrayscale(p=0.05),
            ])
            img = c_jitter(img)
        ts = torch.cat([img, mask], dim=0)
        # optional rotation on both images and masks
        if self.rotation:
            ts = random_rotate(ts, prob=0.875)
            img = ts[:3, :, :]
            mask = ts[3, :, :]
        # optional other preprocessing on both images and masks
        if self.preprocess is not None:
            ts = self.preprocess(ts)
            img = ts[:3, :, :]
            mask = ts[3, :, :]
        mask = (mask > 0.5).float()
        if len(mask.size()) < 3:
            mask = torch.unsqueeze(mask, dim=0)
        return {
            'image': img,
            'mask': mask,
            "ID": idx
        }

    def __len__(self):
        return len(self.ids)


class TestDataset(torch.utils.data.Dataset):
    """
    Dataset for loading test images
    """

    def __init__(self, test_dir, num_imgs=50, to_numpy=False):
        self.test_dir = test_dir
        self.num_imgs = num_imgs
        ids = []
        for i in range(1, num_imgs + 1):
            idx = os.path.join("test_%d" % i, "test_%d.png" % i)
            ids.append(idx)
        self.ids = ids
        self.to_numpy = to_numpy

    def __getitem__(self, i):
        image_path = os.path.join(self.test_dir, self.ids[i])
        with open(image_path, 'rb') as f:
            img = PIL.Image.open(f).convert('RGB')
        img = torchvision.transforms.ToTensor()(img)
        if self.to_numpy:
            img = img.numpy()

        return {"image": img,
                "ID": self.ids[i]
                }

    def __len__(self):
        return len(self.ids)
