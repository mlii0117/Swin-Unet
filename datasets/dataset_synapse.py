import os
import random
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class TestGenerator(object):

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample

class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, annotation, split, mask, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.data_dir = base_dir
        self.df = pd.read_csv(annotation)
        self.df = self.df[self.df['split'] == split]
        self.df.reset_index(drop=True, inplace=True)
        self.mask = mask

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        info = self.df.iloc[idx]
        image_path = info['img']
        mask_path = info[self.mask]
        # read the images and labels
        image = Image.open(os.path.join(self.data_dir, image_path)).convert('L')
        label = Image.open(os.path.join(self.data_dir, mask_path))
        # change image to numpy
        image = np.array(image)
        label = np.array(label)
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        # data_type	data_id	data_ind
        sample['case_name'] = str(info['data_id'])+'_'+str(info['data_ind'])+'_'+str(info['data_type'])
        return sample

